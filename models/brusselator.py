"""Brusselator model.

This module owns the model-specific configuration, physics, data generation,
and learned-ODE wrapper code for this system.  It is the source of truth for
this model.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from torch import Tensor

from .base import (
    BasePhysics2D,
    CKFConfig,
    DataGenConfig,
    ExperimentConfig,
    IntegratorConfig,
    NNConfig,
    PlotConfig,
    TrainConfig,
    _to_torch,
)

from .base import LearnedODE3D

@dataclass
class BrusselatorSystemConfig:
    """Brusselator constants, initial conditions, and state bounds."""
    # System parameters
    A: float = 1.0  # Input concentration of reactant
    B: float = 3.0  # Input concentration of reactant (B > 1 + A?? for oscillations)
    C: float = 1.0  # Coupling constant for z dynamics

    # Initial states
    x0_true: float = 1.5
    y0_true: float = 2.5
    z0_true: float = 1.5

    x0_prior: float = 1.5
    y0_prior: float = 2.5
    z0_prior: float = 1.5

    # State bounds
    x_min: float = 0.25
    x_max: float = 6.0
    y_min: float = 0.25
    y_max: float = 6.0
    z_min: float = 0.25
    z_max: float = 6.0

class BrusselatorPhysics(torch.jit.ScriptModule):
    """
    Brusselator chemical oscillator model.

    This class keeps the full mechanistic RHS and measurement model together.

    State: y = [x, y, z]
    - x: reactive species 1 (autocatalytic)
    - y: reactive species 2
    - z: slower-responding species

    Measurement: h(y) = [y, z] (x not measured)

    Dynamics:
        dx/dt = A + x²y - Bx - x
        dy/dt = Bx - x²y
        dz/dt = C(x - z)
    """

    __constants__ = []

    def __init__(
        self,
        sys_cfg: BrusselatorSystemConfig,
        integrator_cfg: IntegratorConfig,
        ckf_cfg: CKFConfig,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()

        # Noise covariances
        Q_diag = list(ckf_cfg.Q_diag)
        R_diag = list(ckf_cfg.R_diag)
        self.register_buffer("Q", torch.diag(torch.tensor(Q_diag, dtype=torch.float32)))
        self.register_buffer("R", torch.diag(torch.tensor(R_diag, dtype=torch.float32)))

        # Integrator parameters
        self.register_buffer("default_dt", torch.tensor(integrator_cfg.default_dt))
        self.register_buffer("max_dt", torch.tensor(integrator_cfg.max_dt))
        self.register_buffer("min_dt", torch.tensor(integrator_cfg.min_dt))

        # System parameters
        self.register_buffer("A", torch.tensor(sys_cfg.A))
        self.register_buffer("B", torch.tensor(sys_cfg.B))
        self.register_buffer("C", torch.tensor(sys_cfg.C))

        # State bounds
        self.register_buffer("X_MIN", torch.tensor(sys_cfg.x_min))
        self.register_buffer("X_MAX", torch.tensor(sys_cfg.x_max))
        self.register_buffer("Y_MIN", torch.tensor(sys_cfg.y_min))
        self.register_buffer("Y_MAX", torch.tensor(sys_cfg.y_max))
        self.register_buffer("Z_MIN", torch.tensor(sys_cfg.z_min))
        self.register_buffer("Z_MAX", torch.tensor(sys_cfg.z_max))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Clamp all state variables to physical bounds"""
        x = torch.clamp(y[..., 0:1], self.X_MIN, self.X_MAX)
        y_var = torch.clamp(y[..., 1:2], self.Y_MIN, self.Y_MAX)
        z = torch.clamp(y[..., 2:3], self.Z_MIN, self.Z_MAX)
        return torch.cat([x, y_var, z], dim=-1)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        """
        Brusselator dynamics.

        State: [x, y, z]
        Returns: [dx/dt, dy/dt, dz/dt]
        """
        x = y[..., 0:1]
        y_var = y[..., 1:2]
        z = y[..., 2:3]

        # Math form: dx/dt=A+x^2*y-B*x-x, dy/dt=B*x-x^2*y, dz/dt=C*(x-z).
        # Brusselator equations
        # dx/dt = A + x²y - Bx - x
        dx = self.A + (x ** 2) * y_var - self.B * x - x

        # dy/dt = Bx - x²y
        dy = self.B * x - (x ** 2) * y_var

        # dz/dt = C(x - z)
        dz = self.C * (x - z)

        return torch.cat([dx, dy, dz], dim=-1)

    @torch.jit.script_method
    def _rk4_single(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Single RK4 step from the current state vector."""
        k1 = self.rhs_true(state)
        k2 = self.rhs_true(state + 0.5 * dt * k1)
        k3 = self.rhs_true(state + 0.5 * dt * k2)
        k4 = self.rhs_true(state + dt * k3)
        y_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clip_states(y_new)

    @torch.jit.script_method
    def rk4_step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
        """RK4 integration with adaptive sub-stepping"""
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            return y.squeeze(0) if single else y

        # Handle negative dt
        sign = 1.0
        if total_dt < 0.0:
            sign = -1.0
            total_dt = -total_dt

        max_dt = float(self.max_dt.item())
        min_dt = float(self.min_dt.item())

        # Enforce minimum step
        if min_dt > 0.0 and total_dt < min_dt:
            total_dt = min_dt

        # Adaptive sub-stepping
        n_sub = 1
        if max_dt > 0.0 and total_dt > max_dt:
            n_sub = int(total_dt / max_dt + 0.999999)

        h = sign * (total_dt / float(n_sub))

        # Integrate
        out = y
        for _ in range(n_sub):
            out = self._rk4_single(out, h)

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: torch.Tensor, dt: float) -> torch.Tensor:
        """Propagate batch of states (for CKF sigma points)"""
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: torch.Tensor) -> torch.Tensor:
        """
        Measurement model: observe [y, z].

        x is not measured (will be learned by neural network).

        Args:
            X: States [..., 3] = [x, y, z]

        Returns:
            Measurements [..., 2] = [y, z]
        """
        # Concatenate observed species only; x is the hidden state supervised through smoothing.
        return torch.cat([X[:, 1:2], X[:, 2:3]], dim=-1)

def make_data_brusselator(exp: ExperimentConfig, device: torch.device):
    """
    Brusselator chemical oscillator:
      state: [x, y, z]
      measurement: [y, z] + noise  -> shape (T, 2)
      (x is not measured)

    Creates the synthetic trajectory used for CKF/RTS and hybrid training.
    """
    sys_cfg: BrusselatorSystemConfig = exp.system_cfg  # type: ignore
    tc = exp.train_cfg
    dc = exp.data_cfg

    # Initial conditions
    x0 = sys_cfg.x0_true
    y0 = sys_cfg.y0_true
    z0 = sys_cfg.z0_true
    y0_np = np.array([x0, y0, z0], dtype=np.float64)

    # Define the Brusselator dynamics for scipy
    def brusselator_rhs(t, y):
        """Brusselator ODE system"""
        x, y_var, z = y

        # Same state order as rhs_true: derivatives are for [x, y, z].
        # Brusselator equations
        # dx/dt = A + x²y - Bx - x
        dx = sys_cfg.A + (x ** 2) * y_var - sys_cfg.B * x - x

        # dy/dt = Bx - x²y
        dy = sys_cfg.B * x - (x ** 2) * y_var

        # dz/dt = C(x - z)
        dz = sys_cfg.C * (x - z)

        return [dx, dy, dz]

    # Integrate using scipy
    from scipy.integrate import solve_ivp

    max_step = dc.max_step_scipy if dc.max_step_scipy > 0 else np.inf
    sol = solve_ivp(
        brusselator_rhs,
        t_span=(0.0, tc.T_train),
        y0=y0_np,
        method='RK45',
        max_step=max_step,
        dense_output=True,
    )

    # Create uniform time grid if requested
    if dc.uniform_dt > 0:
        n_pts = int(tc.T_train / dc.uniform_dt) + 1
        t_np = np.linspace(0.0, tc.T_train, n_pts)
        y_np = sol.sol(t_np).T  # (T, 3)
    else:
        t_np = sol.t
        y_np = sol.y.T  # (T, 3)

    # Convert to torch
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_np, dtype=torch.float32, device=device)  # (T, 3)

    x_true = y_true[:, 0]
    y_true_var = y_true[:, 1]
    z_true = y_true[:, 2]

    # Per-channel proportional noise: noise_std = meas_noise_std * |channel_rms|
    torch.manual_seed(dc.seed)
    y_scale = y_true_var.abs().mean().clamp(min=1e-6)
    z_scale = z_true.abs().mean().clamp(min=1e-6)
    y_meas = y_true_var + tc.meas_noise_std * y_scale * torch.randn_like(y_true_var)
    z_meas = z_true + tc.meas_noise_std * z_scale * torch.randn_like(z_true)

    # Stack measurements [y, z] shape (T, 2)
    meas = torch.stack([y_meas, z_meas], dim=1)

    # Return: t, meas, and all true states
    return t, meas, x_true, y_true_var, z_true

LearnedODEBrusselator = LearnedODE3D

def make_experiment() -> ExperimentConfig:
    # Central preset for this model: physics, filter noise, NN size, training, and plotting.
    return ExperimentConfig(system_name='brusselator', system_cfg=BrusselatorSystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.005 / 3.0, max_dt=0.01 / 3.0, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(0.1, 0.01, 0.01), R_diag=(0.001, 0.001), P0_diag=(0.5, 0.1, 0.1)), nn_cfg=NNConfig(in_dim=3, out_dim=1, hidden_dims=(48, 48, 48, 48), activation='tanh', input_scale=(1.0, 1.0, 1.0), lr=0.001, weight_decay=1e-06, grad_clip=5.0), train_cfg=TrainConfig(T_train=50.0, meas_noise_std=0.05, n_epochs=50, min_rollout_horizon=1, max_rollout_horizon=1, stride=3, use_pseudo_latent=True, pseudo_weight=2.0), plot_cfg=PlotConfig(plot_stride=30, show_every=5, max_rollouts=15), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.0, seed=1234), state_labels=('x', 'y', 'z'), title_prefix='Brusselator', meas_idx=(1, 2), pseudo_idx=(0,), uses_pin=False)

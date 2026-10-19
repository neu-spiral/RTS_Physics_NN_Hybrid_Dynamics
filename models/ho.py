"""Harmonic oscillator model.

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

from .base import LearnedODE2D

@dataclass
class HOSystemConfig:
    """Harmonic oscillator constants, initial conditions, and state bounds."""
    omega: float = 1.0  # Reduced from 2?? to keep v_max ??? 1.0 (v_max = omega * x0)
    x0_true: float = 1.0
    v0_true: float = 0.0
    x0_prior: float = 1.0
    v0_prior: float = 0.0
    x_min: float = -2.5
    x_max: float = 2.5
    v_min: float = -2.0  # Reduced from -5.0 to match new omega
    v_max: float = 2.0   # Reduced from 5.0 to match new omega

class HOPhysics(BasePhysics2D):
    """
    Harmonic Oscillator: d²x/dt² + ω²x = 0

    This physics class is used by CKF/RTS and by the hybrid wrapper for the known ODE terms.

    State: y = [x, v]
    Dynamics:
        dx/dt = v
        dv/dt = -ω²x
    """

    def __init__(
        self,
        sys_cfg: HOSystemConfig,
        integrator_cfg: IntegratorConfig,
        ckf_cfg: CKFConfig,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__(integrator_cfg, ckf_cfg)

        # System parameters
        self.register_buffer("omega", torch.tensor(sys_cfg.omega))

        # State bounds
        self.register_buffer("X_MIN", torch.tensor(sys_cfg.x_min))
        self.register_buffer("X_MAX", torch.tensor(sys_cfg.x_max))
        self.register_buffer("V_MIN", torch.tensor(sys_cfg.v_min))
        self.register_buffer("V_MAX", torch.tensor(sys_cfg.v_max))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Clamp position and velocity to physical bounds"""
        x = torch.clamp(y[..., 0:1], self.X_MIN, self.X_MAX)
        v = torch.clamp(y[..., 1:2], self.V_MIN, self.V_MAX)
        return torch.cat([x, v], dim=-1)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        """
        Harmonic oscillator dynamics.

        dx/dt = v
        dv/dt = -ω²x
        """
        x = y[..., 0:1]
        v = y[..., 1:2]
        # Math form: dx/dt = v, dv/dt = -omega^2*x.
        dx = v
        dv = -(self.omega * self.omega) * x
        return torch.cat([dx, dv], dim=-1)

def integrate_ho_adaptive(
    y0_np,
    t0,
    T,
    omega,
    max_step_scipy=0.0,
    uniform_dt=0.0,
    return_torch=True,
    device=None,
):
    """
    Integrate HO:
        x' = v
        v' = -omega^2 x

    y0_np : [x0, v0]
    """
    y0_np = np.asarray(y0_np, dtype=np.float64)

    def rhs(t, y):
        x, v = y[0], y[1]
        # Same state order as rhs_true: first derivative is for x, second is for v.
        dx = v
        dv = - (omega ** 2) * x
        return np.array([dx, dv], dtype=np.float64)

    t_span = (float(t0), float(T))

    kwargs = dict(method="RK45", rtol=1e-6, atol=1e-9)
    if max_step_scipy > 0.0:
        kwargs["max_step"] = max_step_scipy
    if uniform_dt > 0.0:
        N = int((T - t0) / uniform_dt + 0.5) + 1
        kwargs["t_eval"] = np.linspace(t0, T, N, endpoint=True)

    sol = solve_ivp(rhs, t_span, y0_np, **kwargs)
    t_np = sol.t
    y_np = sol.y.T  # (T, 2)

    return _to_torch(t_np, y_np, return_torch=return_torch, device=device)

def make_data_ho(exp: ExperimentConfig, device: torch.device):
    """
    Harmonic oscillator:
      state: [x, v]
      measurement: x + noise  -> shape (T,)

    Creates the synthetic trajectory used for CKF/RTS and hybrid training.
    """
    sys_cfg: HOSystemConfig = exp.system_cfg  # type: ignore
    tc = exp.train_cfg
    dc = exp.data_cfg

    y0_np = np.array([sys_cfg.x0_true, sys_cfg.v0_true], dtype=np.float64)

    t, y_true = integrate_ho_adaptive(
        y0_np=y0_np,
        t0=0.0,
        T=tc.T_train,
        omega=sys_cfg.omega,
        max_step_scipy=dc.max_step_scipy,
        uniform_dt=dc.uniform_dt,
        return_torch=True,
        device=device,
    )
    # y_true: (T, 2) or (T, 1, 2)
    if y_true.dim() == 3:
        y_true = y_true[:, 0, :]  # (T,2)

    x_true = y_true[:, 0]
    v_true = y_true[:, 1]

    # Per-channel proportional noise: noise_std = meas_noise_std * |channel_rms|
    # so SNR = channel_rms / noise_std = 1 / meas_noise_std, consistent across channels
    torch.manual_seed(dc.seed)
    x_scale = x_true.abs().mean().clamp(min=1e-6)
    x_meas = x_true + tc.meas_noise_std * x_scale * torch.randn_like(x_true)

    return t, x_meas, x_true, v_true

LearnedODEHO = LearnedODE2D

def make_experiment() -> ExperimentConfig:
    # Central preset for this model: physics, filter noise, NN size, training, and plotting.
    return ExperimentConfig(system_name='ho', system_cfg=HOSystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.01, max_dt=0.02, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(0.001, 0.1), R_diag=(0.0001,), P0_diag=(0.01, 0.5)), nn_cfg=NNConfig(in_dim=2, out_dim=1, hidden_dims=(48, 48, 48), activation='tanh', input_scale=(1.0, 1.0), lr=0.001, weight_decay=0.0, grad_clip=0.5), train_cfg=TrainConfig(T_train=30.0, meas_noise_std=0.0005, n_epochs=50, min_rollout_horizon=5, max_rollout_horizon=20, stride=1, use_pseudo_latent=True, pseudo_weight=10.0), plot_cfg=PlotConfig(plot_stride=30, show_every=10, max_rollouts=15), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.01, seed=1234), state_labels=('x', 'v'), title_prefix='HO', meas_idx=(0,), pseudo_idx=(1,), uses_pin=False)

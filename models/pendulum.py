"""Pendulum model.

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
class PendulumSystemConfig:
    """Pendulum constants, initial conditions, and state bounds."""
    g: float = 9.81
    L: float = 1.0
    theta0_true: float = 0.5
    omega0_true: float = 0.0
    theta0_prior: float = 0.5
    omega0_prior: float = 0.0
    theta_min: float = -math.pi
    theta_max: float = math.pi
    omega_min: float = -10.0
    omega_max: float = 10.0

class PendulumPhysics(BasePhysics2D):
    """
    Nonlinear Pendulum: d²θ/dt² + (g/L)sin(θ) = 0

    State: y = [θ, ω]
    This physics class is used by CKF/RTS and by the hybrid wrapper for the known ODE terms.

    Dynamics:
        dθ/dt = ω
        dω/dt = -(g/L)sin(θ)
    """

    def __init__(
        self,
        sys_cfg: PendulumSystemConfig,
        integrator_cfg: IntegratorConfig,
        ckf_cfg: CKFConfig,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__(integrator_cfg, ckf_cfg)

        # System parameters
        self.register_buffer("g_over_L", torch.tensor(sys_cfg.g / sys_cfg.L))

        # State bounds
        self.register_buffer("TH_MIN", torch.tensor(sys_cfg.theta_min))
        self.register_buffer("TH_MAX", torch.tensor(sys_cfg.theta_max))
        self.register_buffer("OM_MIN", torch.tensor(sys_cfg.omega_min))
        self.register_buffer("OM_MAX", torch.tensor(sys_cfg.omega_max))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Clamp angle and angular velocity to physical bounds"""
        th = torch.clamp(y[..., 0:1], self.TH_MIN, self.TH_MAX)
        om = torch.clamp(y[..., 1:2], self.OM_MIN, self.OM_MAX)
        return torch.cat([th, om], dim=-1)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        """
        Pendulum dynamics.

        dθ/dt = ω
        dω/dt = -(g/L)sin(θ)
        """
        th = y[..., 0:1]
        om = y[..., 1:2]
        # Math form: dtheta/dt = omega, domega/dt = -(g/L)*sin(theta).
        dth = om
        dom = -self.g_over_L * torch.sin(th)
        return torch.cat([dth, dom], dim=-1)

def integrate_pendulum_adaptive(
    y0_np,
    t0,
    T,
    g,
    L,
    max_step_scipy=0.0,
    uniform_dt=0.0,
    return_torch=True,
    device=None,
):
    """
    Simple pendulum:
        theta' = omega
        omega' = - (g/L) sin(theta)

    y0_np : [theta0, omega0]
    """
    y0_np = np.asarray(y0_np, dtype=np.float64)

    def rhs(t, y):
        th, om = y[0], y[1]
        # Same state order as rhs_true: first derivative is for theta, second is for omega.
        dth = om
        dom = - (g / L) * math.sin(th)
        return np.array([dth, dom], dtype=np.float64)

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

def make_data_pendulum(exp: ExperimentConfig, device: torch.device):
    """
    Pendulum:
      state: [theta, omega]
      measurement: theta + noise  -> shape (T,)

    Creates the synthetic trajectory used for CKF/RTS and hybrid training.
    """
    sys_cfg: PendulumSystemConfig = exp.system_cfg  # type: ignore
    tc = exp.train_cfg
    dc = exp.data_cfg

    y0_np = np.array([sys_cfg.theta0_true, sys_cfg.omega0_true], dtype=np.float64)

    t, y_true = integrate_pendulum_adaptive(
        y0_np=y0_np,
        t0=0.0,
        T=tc.T_train,
        g=sys_cfg.g,
        L=sys_cfg.L,
        max_step_scipy=dc.max_step_scipy,
        uniform_dt=dc.uniform_dt,
        return_torch=True,
        device=device,
    )
    # y_true: (T, 2) or (T, 1, 2)
    if y_true.dim() == 3:
        y_true = y_true[:, 0, :]

    th_true = y_true[:, 0]
    om_true = y_true[:, 1]

    # Per-channel proportional noise: noise_std = meas_noise_std * |channel_rms|
    torch.manual_seed(dc.seed)
    th_scale = th_true.abs().mean().clamp(min=1e-6)
    th_meas = th_true + tc.meas_noise_std * th_scale * torch.randn_like(th_true)

    return t, th_meas, th_true, om_true

LearnedODEPendulum = LearnedODE2D

def make_experiment() -> ExperimentConfig:
    # Central preset for this model: physics, filter noise, NN size, training, and plotting.
    return ExperimentConfig(system_name='pendulum', system_cfg=PendulumSystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.01, max_dt=0.02, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(1e-06, 1e-06), R_diag=(0.001,), P0_diag=(0.01, 0.01)), nn_cfg=NNConfig(in_dim=2, out_dim=1, hidden_dims=(48, 48), activation='tanh', input_scale=(1.0, 1.0), lr=0.05, weight_decay=1e-06, grad_clip=5.0), train_cfg=TrainConfig(T_train=5.0, meas_noise_std=0.01, n_epochs=50, min_rollout_horizon=3, max_rollout_horizon=30, stride=1, use_pseudo_latent=True, pseudo_weight=0.1), plot_cfg=PlotConfig(plot_stride=30, show_every=5, max_rollouts=15), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.0, seed=1234), state_labels=('theta', 'omega'), title_prefix='Pendulum', meas_idx=(0,), pseudo_idx=(1,), uses_pin=False)

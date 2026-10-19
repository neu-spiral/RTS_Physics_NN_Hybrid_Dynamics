"""Hodgkin-Huxley model.

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

from .base import LearnedODE4D

@dataclass
class HodgkinHuxleySystemConfig:
    """Hodgkin-Huxley constants, initial conditions, and state bounds."""
    # Membrane capacitance (??F/cm??)
    C_m: float = 1.0

    # Maximum conductances (mS/cm??)
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3

    # Reversal potentials (mV)
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387

    # External current (??A/cm??) - always on
    I_ext: float = 10.0

    # Initial states (resting state at V=-65 mV)
    V0_true: float = -65.0
    m0_true: float = 0.0529  # Computed from alpha/beta at rest
    h0_true: float = 0.5961
    n0_true: float = 0.3177

    V0_prior: float = -65.0
    m0_prior: float = 0.5  # Uninformative prior for unmeasured state
    h0_prior: float = 0.5961
    n0_prior: float = 0.3177

    # State bounds
    V_min: float = -100.0
    V_max: float = 60.0
    gating_min: float = 0.0
    gating_max: float = 1.0

class HodgkinHuxleyPhysics(torch.jit.ScriptModule):
    """
    Hodgkin-Huxley neuron model.

    This class owns the full conductance-based RHS and measurement model.

    State: y = [V, m, h, n]
    - V: Membrane potential (mV)
    - m: Sodium activation gating variable
    - h: Sodium inactivation gating variable
    - n: Potassium activation gating variable

    Measurement: h(y) = [V, h, n] (all except m)

    Dynamics:
        dV/dt = (I_ext - I_Na - I_K - I_L) / C_m
        dm/dt = α_m(V)(1-m) - β_m(V)m
        dh/dt = α_h(V)(1-h) - β_h(V)h
        dn/dt = α_n(V)(1-n) - β_n(V)n

    where:
        I_Na = g_Na * m³ * h * (V - E_Na)
        I_K = g_K * n⁴ * (V - E_K)
        I_L = g_L * (V - E_L)
    """

    __constants__ = []

    def __init__(
        self,
        sys_cfg: HodgkinHuxleySystemConfig,
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
        self.register_buffer("C_m", torch.tensor(sys_cfg.C_m))
        self.register_buffer("g_Na", torch.tensor(sys_cfg.g_Na))
        self.register_buffer("g_K", torch.tensor(sys_cfg.g_K))
        self.register_buffer("g_L", torch.tensor(sys_cfg.g_L))
        self.register_buffer("E_Na", torch.tensor(sys_cfg.E_Na))
        self.register_buffer("E_K", torch.tensor(sys_cfg.E_K))
        self.register_buffer("E_L", torch.tensor(sys_cfg.E_L))
        self.register_buffer("I_ext", torch.tensor(sys_cfg.I_ext))

        # State bounds
        self.register_buffer("V_MIN", torch.tensor(sys_cfg.V_min))
        self.register_buffer("V_MAX", torch.tensor(sys_cfg.V_max))
        self.register_buffer("GATING_MIN", torch.tensor(sys_cfg.gating_min))
        self.register_buffer("GATING_MAX", torch.tensor(sys_cfg.gating_max))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Clamp voltage and gating variables to physical bounds"""
        V = torch.clamp(y[..., 0:1], self.V_MIN, self.V_MAX)
        m = torch.clamp(y[..., 1:2], self.GATING_MIN, self.GATING_MAX)
        h = torch.clamp(y[..., 2:3], self.GATING_MIN, self.GATING_MAX)
        n = torch.clamp(y[..., 3:4], self.GATING_MIN, self.GATING_MAX)
        return torch.cat([V, m, h, n], dim=-1)

    @torch.jit.script_method
    def _alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium activation rate"""
        return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

    @torch.jit.script_method
    def _beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium deactivation rate"""
        return 4.0 * torch.exp(-(V + 65.0) / 18.0)

    @torch.jit.script_method
    def _alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium inactivation rate"""
        return 0.07 * torch.exp(-(V + 65.0) / 20.0)

    @torch.jit.script_method
    def _beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium de-inactivation rate"""
        return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

    @torch.jit.script_method
    def _alpha_n(self, V: torch.Tensor) -> torch.Tensor:
        """Potassium activation rate"""
        return 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))

    @torch.jit.script_method
    def _beta_n(self, V: torch.Tensor) -> torch.Tensor:
        """Potassium deactivation rate"""
        return 0.125 * torch.exp(-(V + 65.0) / 80.0)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        """
        Hodgkin-Huxley dynamics.

        State: [V, m, h, n]
        Returns: [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        V = y[..., 0:1]
        m = y[..., 1:2]
        h = y[..., 2:3]
        n = y[..., 3:4]

        # Math form: dV/dt=(Iext-INa-IK-IL)/Cm, dg/dt=alpha_g(V)*(1-g)-beta_g(V)*g for g in {m,h,n}.
        # Gating variable dynamics
        dm = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dh = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dn = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n

        # Ionic currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Membrane potential dynamics
        dV = (self.I_ext - I_Na - I_K - I_L) / self.C_m

        return torch.cat([dV, dm, dh, dn], dim=-1)

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
        Measurement model: observe [V, h, n].

        m is not measured (will be learned by neural network).

        Args:
            X: States [..., 4] = [V, m, h, n]

        Returns:
            Measurements [..., 3] = [V, h, n]
        """
        # Concatenate only measured channels; m is intentionally left for the smoother/NN.
        return torch.cat([X[:, 0:1], X[:, 2:3], X[:, 3:4]], dim=-1)

def make_data_hh(exp: ExperimentConfig, device: torch.device):
    """
    Hodgkin-Huxley neuron:
      state: [V, m, h, n]
      measurement: [V, h, n] + noise  -> shape (T, 3)
      (m is not measured)

    Creates the synthetic trajectory used for CKF/RTS and hybrid training.
    """
    sys_cfg: HodgkinHuxleySystemConfig = exp.system_cfg  # type: ignore
    tc = exp.train_cfg
    dc = exp.data_cfg

    # Initial conditions at resting state
    V0 = sys_cfg.V0_true
    m0 = sys_cfg.m0_true
    h0 = sys_cfg.h0_true
    n0 = sys_cfg.n0_true
    y0_np = np.array([V0, m0, h0, n0], dtype=np.float64)

    # Define the HH dynamics for scipy
    def hh_rhs(t, y):
        """Hodgkin-Huxley ODE system"""
        V, m, h, n = y

        # Same state order as rhs_true: derivatives are for [V, m, h, n].
        # Rate functions (numerically safe versions to avoid exp overflow)
        # alpha_m: 0.1*(V+40)/(1-exp(-(V+40)/10))
        x_m = (V + 40.0) / 10.0
        # Use expm1 and clip to avoid overflow for large |x_m|
        if abs(x_m) > 1e-7:
            x_m_clip = np.clip(x_m, -50.0, 50.0)
            denom_m = -np.expm1(-x_m_clip)  # 1 - exp(-x) = -expm1(-x)
            alpha_m = x_m_clip / denom_m
        else:
            alpha_m = 1.0
        beta_m = 4.0 * np.exp(np.clip(-(V + 65.0) / 18.0, -500, 500))
        alpha_h = 0.07 * np.exp(np.clip(-(V + 65.0) / 20.0, -500, 500))
        beta_h = 1.0 / (1.0 + np.exp(np.clip(-(V + 35.0) / 10.0, -500, 500)))
        # alpha_n: 0.01*(V+55)/(1-exp(-(V+55)/10))
        x_n = (V + 55.0) / 10.0
        if abs(x_n) > 1e-7:
            x_n_clip = np.clip(x_n, -50.0, 50.0)
            denom_n = -np.expm1(-x_n_clip)
            alpha_n = 0.1 * x_n_clip / denom_n
        else:
            alpha_n = 0.1
        beta_n = 0.125 * np.exp(np.clip(-(V + 65.0) / 80.0, -500, 500))

        # Gating dynamics
        dm = alpha_m * (1.0 - m) - beta_m * m
        dh = alpha_h * (1.0 - h) - beta_h * h
        dn = alpha_n * (1.0 - n) - beta_n * n

        # Ionic currents
        I_Na = sys_cfg.g_Na * (m ** 3) * h * (V - sys_cfg.E_Na)
        I_K = sys_cfg.g_K * (n ** 4) * (V - sys_cfg.E_K)
        I_L = sys_cfg.g_L * (V - sys_cfg.E_L)

        # Membrane potential dynamics
        dV = (sys_cfg.I_ext - I_Na - I_K - I_L) / sys_cfg.C_m

        return [dV, dm, dh, dn]

    # Integrate using scipy
    from scipy.integrate import solve_ivp

    max_step = dc.max_step_scipy if dc.max_step_scipy > 0 else np.inf
    sol = solve_ivp(
        hh_rhs,
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
        y_np = sol.sol(t_np).T  # (T, 4)
    else:
        t_np = sol.t
        y_np = sol.y.T  # (T, 4)

    # Convert to torch
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_np, dtype=torch.float32, device=device)  # (T, 4)

    V_true = y_true[:, 0]
    m_true = y_true[:, 1]
    h_true = y_true[:, 2]
    n_true = y_true[:, 3]

    # Per-channel proportional noise: noise_std = meas_noise_std * |channel_rms|
    torch.manual_seed(dc.seed)
    V_scale = V_true.abs().mean().clamp(min=1e-6)
    h_scale = h_true.abs().mean().clamp(min=1e-6)
    n_scale = n_true.abs().mean().clamp(min=1e-6)
    V_meas = V_true + tc.meas_noise_std * V_scale * torch.randn_like(V_true)
    h_meas = h_true + tc.meas_noise_std * h_scale * torch.randn_like(h_true)
    n_meas = n_true + tc.meas_noise_std * n_scale * torch.randn_like(n_true)

    # Stack measurements [V, h, n] shape (T, 3)
    meas = torch.stack([V_meas, h_meas, n_meas], dim=1)

    # Return: t, meas, and all true states
    return t, meas, V_true, m_true, h_true, n_true

LearnedODEHH = LearnedODE4D

def make_experiment() -> ExperimentConfig:
    # Central preset for this model: physics, filter noise, NN size, training, and plotting.
    return ExperimentConfig(system_name='hh', system_cfg=HodgkinHuxleySystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.00125, max_dt=0.0025, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(1000.0, 1.0, 0.1, 0.1), R_diag=(0.01, 0.001, 0.001), P0_diag=(100.0, 0.5, 0.1, 0.1)), nn_cfg=NNConfig(in_dim=4, out_dim=1, hidden_dims=(48, 48), activation='tanh', input_scale=(0.01, 1.0, 1.0, 1.0), lr=0.01, weight_decay=1e-06, grad_clip=10.0), train_cfg=TrainConfig(T_train=50.0, meas_noise_std=0.05, n_epochs=50, min_rollout_horizon=3, max_rollout_horizon=50, stride=5, use_pseudo_latent=True, pseudo_weight=1.0), plot_cfg=PlotConfig(plot_stride=20, show_every=5, max_rollouts=20), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.0, seed=1234), state_labels=('V', 'm', 'h', 'n'), title_prefix='HH', meas_idx=(0, 2, 3), pseudo_idx=(1,), uses_pin=False)

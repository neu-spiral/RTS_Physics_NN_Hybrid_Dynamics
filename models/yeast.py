"""Yeast glycolysis model.

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

@dataclass
class YeastSystemConfig:
    """Yeast glycolysis constants, burn-in settings, and initial states."""
    # Initial conditions before burn-in
    x0_true: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    x0_prior: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    burn_t0: float = 0.0
    burn_t1: float = 200.0
    # Oscillatory regime parameters
    c1: float = 8.0
    c2: float = -100.0
    c3: float = 13.6769
    d1: float = 200.0
    d2: float = 13.6769
    d3: float = -6.0
    d4: float = 6.0
    e1: float = 6.0
    e2: float = -64.0
    e3: float = 6.0
    e4: float = 16.0
    f1: float = 64.0
    f2: float = -13.0
    f3: float = 13.0
    f4: float = -16.0
    f5: float = -100.0
    g1: float = 1.3
    g2: float = -3.1
    h1: float = -200.0
    h2: float = 13.6769
    h3: float = 128.0
    h4: float = -1.28
    h5: float = -32.0
    j1: float = 6.0
    j2: float = -18.0
    j3: float = -100.0

class YeastPhysics(torch.jit.ScriptModule):
    """
    Physics model for the 7-state yeast glycolysis oscillator.

    This class is the mechanistic teacher used by CKF/RTS and by the hybrid
    wrapper before one biochemical derivative is replaced.

    Implements the Goldbeter-type biochemical oscillator that describes
    sustained oscillations in the glycolytic pathway of yeast cells. The
    autocatalytic feedback loop at the phosphofructokinase (PFK) step drives
    the periodic behavior, making this a classic benchmark for nonlinear
    state estimation and hybrid modeling.

    States: [x1, x2, x3, x4, x5, x6, x7]
      x1: Substrate (glucose-6-phosphate)
      x2: PFK product (fructose-1,6-bisphosphate)
      x3: Downstream glycolytic intermediate (DHAP/G3P)
      x4: Further downstream intermediate (1,3-BPG)  — may be LEARNED BY NN
      x5: Pyruvate or downstream product
      x6: Allosteric regulator / ADP-ATP ratio
      x7: Redox cofactor (NADH)

    One state (default x4, replaced_index=3) can be replaced by a learned
    neural ODE component during hybrid training. The remaining states follow
    the analytical kinetic equations. The measurement model observes all
    states except the replaced one (default: indices [0,1,2,4,5,6]).

    Noise matrices:
      Q: Process noise covariance (7×7 diagonal)
      R: Measurement noise covariance (6×6 diagonal, measured states only)

    This class is compiled as a TorchScript module for efficient execution
    during CKF sigma-point propagation and multi-step rollout training.
    """

    __constants__ = []

    def __init__(
        self,
        sys_cfg,
        integrator_cfg: IntegratorConfig,
        ckf_cfg: CKFConfig,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()

        # Noise covariances (7-state system)
        Q_diag = list(ckf_cfg.Q_diag)
        R_diag_full = list(ckf_cfg.R_diag)
        self.replaced_index: int = int(getattr(sys_cfg, "replaced_index", 3))  # x4 is learned
        meas_idx = list(getattr(sys_cfg, "meas_idx", [0, 1, 2, 4, 5, 6]))
        # If R_diag is per-state (len 7), filter to measured order; otherwise assume it already matches meas_idx length.
        if len(R_diag_full) == 7:
            R_diag = [R_diag_full[i] for i in meas_idx]
        elif len(R_diag_full) == len(meas_idx):
            R_diag = R_diag_full
        else:
            raise ValueError(f"Yeast R_diag length {len(R_diag_full)} does not match meas_idx {len(meas_idx)}")

        self.register_buffer("Q", torch.diag(torch.tensor(Q_diag, dtype=torch.float32)))
        self.register_buffer("R", torch.diag(torch.tensor(R_diag, dtype=torch.float32)))

        # Integrator parameters
        self.register_buffer("default_dt", torch.tensor(integrator_cfg.default_dt))
        self.register_buffer("max_dt", torch.tensor(integrator_cfg.max_dt))
        self.register_buffer("min_dt", torch.tensor(integrator_cfg.min_dt))
        self.n_substeps: int = int(integrator_cfg.n_substeps)

        # System parameters
        for name in [
            "c1", "c2", "c3", "d1", "d2", "d3", "d4",
            "e1", "e2", "e3", "e4",
            "f1", "f2", "f3", "f4", "f5",
            "g1", "g2",
            "h1", "h2", "h3", "h4", "h5",
            "j1", "j2", "j3",
        ]:
            self.register_buffer(name, torch.tensor(getattr(sys_cfg, name), dtype=torch.float32))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        # Clip states.
        return torch.clamp(y, -1e3, 1e3)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        # Support the rhs true step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        x1 = y[:, 0:1]
        x2 = y[:, 1:2]
        x3 = y[:, 2:3]
        x4 = y[:, 3:4]
        x5 = y[:, 4:5]
        x6 = y[:, 5:6]
        x7 = y[:, 6:7]

        denom_c = 1.0 + self.c3 * (x6 ** 4)
        denom_d = 1.0 + self.d2 * (x6 ** 4)
        denom_h = 1.0 + self.h2 * (x6 ** 4)

        # Math form: output is [dx1/dt, dx2/dt, dx3/dt, dx4/dt, dx5/dt, dx6/dt, dx7/dt].
        dx1 = self.c1 + (self.c2 * x1 * x6) / denom_c
        dx2 = (self.d1 * x1 * x6) / denom_d + self.d3 * x2 - self.d4 * x2 * x7
        dx3 = self.e1 * x2 + self.e2 * x3 + self.e3 * x2 * x7 + self.e4 * x3 * x6
        dx4 = self.f1 * x3 + self.f2 * x4 + self.f3 * x5 + self.f4 * x3 * x6 + self.f5 * x4 * x7
        dx5 = self.g1 * x4 + self.g2 * x5
        dx6 = self.h3 * x3 + self.h5 * x6 + self.h4 * x3 * x6 + (self.h1 * x1 * x6) / denom_h
        dx7 = self.j1 * x2 + self.j2 * x2 * x7 + self.j3 * x4 * x7

        out = torch.cat([dx1, dx2, dx3, dx4, dx5, dx6, dx7], dim=-1)
        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def _rk4_single(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        # Single RK4 step from the current state vector.
        k1 = self.rhs_true(state)
        k2 = self.rhs_true(state + 0.5 * dt * k1)
        k3 = self.rhs_true(state + 0.5 * dt * k2)
        k4 = self.rhs_true(state + dt * k3)
        y_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clip_states(y_new)

    @torch.jit.script_method
    def rk4_step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
        # Support the rk4 step step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            return y.squeeze(0) if single else y

        sign = 1.0
        if total_dt < 0.0:
            sign = -1.0
            total_dt = -total_dt

        max_dt = float(self.max_dt.item())
        min_dt = float(self.min_dt.item())

        if min_dt > 0.0 and total_dt < min_dt:
            total_dt = min_dt

        n_sub = 1
        if max_dt > 0.0 and total_dt > max_dt:
            n_sub = int(total_dt / max_dt + 0.999999)

        h = sign * (total_dt / float(n_sub))

        out = y
        for _ in range(n_sub):
            out = self._rk4_single(out, h)

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: torch.Tensor, dt: float) -> torch.Tensor:
        # Propagate batch.
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: torch.Tensor) -> torch.Tensor:
        """
        Measurement model: observe [x1, x2, x3, x5, x6, x7] (omit x4).
        """
        # Keep measured biochemical states in measurement order while omitting learned x4.
        return torch.cat([X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 4:5], X[:, 5:6], X[:, 6:7]], dim=-1)

def make_data_yeast(exp: ExperimentConfig, device: torch.device):
    """
    Yeast glycolysis oscillator:
      state: [x1..x7]
      measurement: all except x4 (x4 is learned)

    Runs burn-in first, then returns the training trajectory and noisy measurements.
    """
    sys_cfg: YeastSystemConfig = exp.system_cfg  # type: ignore
    tc = exp.train_cfg
    dc = exp.data_cfg

    y0_np = np.array(sys_cfg.x0_true, dtype=np.float64)

    params = dict(
        c1=sys_cfg.c1, c2=sys_cfg.c2, c3=sys_cfg.c3,
        d1=sys_cfg.d1, d2=sys_cfg.d2, d3=sys_cfg.d3, d4=sys_cfg.d4,
        e1=sys_cfg.e1, e2=sys_cfg.e2, e3=sys_cfg.e3, e4=sys_cfg.e4,
        f1=sys_cfg.f1, f2=sys_cfg.f2, f3=sys_cfg.f3, f4=sys_cfg.f4, f5=sys_cfg.f5,
        g1=sys_cfg.g1, g2=sys_cfg.g2,
        h1=sys_cfg.h1, h2=sys_cfg.h2, h3=sys_cfg.h3, h4=sys_cfg.h4, h5=sys_cfg.h5,
        j1=sys_cfg.j1, j2=sys_cfg.j2, j3=sys_cfg.j3,
    )

    def yeast_rhs(t, x):
        # Support the yeast rhs step in this module.
        x1, x2, x3, x4, x5, x6, x7 = x

        denom_c = 1.0 + params["c3"] * (x6 ** 4)
        denom_d = 1.0 + params["d2"] * (x6 ** 4)
        denom_h = 1.0 + params["h2"] * (x6 ** 4)

        # Same state order as rhs_true: derivatives are for [x1, x2, x3, x4, x5, x6, x7].
        dx1 = params["c1"] + (params["c2"] * x1 * x6) / denom_c
        dx2 = (params["d1"] * x1 * x6) / denom_d + params["d3"] * x2 - params["d4"] * x2 * x7
        dx3 = params["e1"] * x2 + params["e2"] * x3 + params["e3"] * x2 * x7 + params["e4"] * x3 * x6
        dx4 = params["f1"] * x3 + params["f2"] * x4 + params["f3"] * x5 + params["f4"] * x3 * x6 + params["f5"] * x4 * x7
        dx5 = params["g1"] * x4 + params["g2"] * x5
        dx6 = (params["h3"] * x3 + params["h5"] * x6 + params["h4"] * x3 * x6 + (params["h1"] * x1 * x6) / denom_h)
        dx7 = params["j1"] * x2 + params["j2"] * x2 * x7 + params["j3"] * x4 * x7

        return [dx1, dx2, dx3, dx4, dx5, dx6, dx7]

    # Burn-in to reach steady oscillation
    burn = solve_ivp(
        yeast_rhs,
        (float(sys_cfg.burn_t0), float(sys_cfg.burn_t1)),
        y0_np,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not burn.success:
        raise RuntimeError(burn.message)
    y0_steady = burn.y[:, -1]

    # Main simulation over training horizon
    max_step = dc.max_step_scipy if dc.max_step_scipy > 0 else np.inf
    kwargs = dict(method="LSODA", rtol=1e-8, atol=1e-10, max_step=max_step)
    if dc.uniform_dt > 0:
        n_pts = int(tc.T_train / dc.uniform_dt) + 1
        kwargs["t_eval"] = np.linspace(0.0, tc.T_train, n_pts)

    sol = solve_ivp(
        yeast_rhs,
        (0.0, tc.T_train),
        y0_steady,
        **kwargs,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t_np = sol.t
    y_np = sol.y.T  # (T, 7)

    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_np, dtype=torch.float32, device=device)  # (T, 7)

    x1_true, x2_true, x3_true, x4_true, x5_true, x6_true, x7_true = y_true.T

    # Measurements: all except x4
    # Use state-relative noise: noise_std = meas_noise_std * state_mean
    # This ensures comparable SNR across states (x5 range is only 0.0025)
    torch.manual_seed(dc.seed)
    meas_list = []
    yeast_meas_states = (x1_true, x2_true, x3_true, x5_true, x6_true, x7_true)
    # x5 has a very small dynamic range; keep its measurement noise very low so
    # rollouts do not drift/shift on x5.
    yeast_noise_scale = (1.0, 1.0, 1.0, 0.03, 1.0, 1.0)
    for state, rel_scale in zip(yeast_meas_states, yeast_noise_scale):
        noise_std = tc.meas_noise_std * rel_scale * state.abs().mean().item()
        meas_list.append(state + noise_std * torch.randn_like(state))
    meas = torch.stack(meas_list, dim=1)  # (T, 6)

    return t, meas, x1_true, x2_true, x3_true, x4_true, x5_true, x6_true, x7_true

YEAST_STATE_NAMES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
YEAST_DEPENDENCIES = {
    0: [0, 5],             # dx1/dt depends on x1, x6
    1: [0, 1, 5, 6],       # dx2/dt depends on x1, x2, x6, x7
    2: [1, 2, 5, 6],       # dx3/dt depends on x2, x3, x6, x7
    3: [2, 3, 4, 5, 6],    # dx4/dt depends on x3, x4, x5, x6, x7 (learned)
    4: [3, 4],             # dx5/dt depends on x4, x5
    5: [0, 2, 5],          # dx6/dt depends on x1, x3, x6
    6: [1, 3, 6],          # dx7/dt depends on x2, x4, x7
}

def get_yeast_input_mask(replaced_index: int) -> torch.Tensor:
    # Build the dependency mask so the NN only sees states used by that derivative.
    if replaced_index not in YEAST_DEPENDENCIES:
        raise ValueError(f"Invalid replaced_index {replaced_index}. Must be 0-6.")
    mask = torch.zeros(7)
    for dep_idx in YEAST_DEPENDENCIES[replaced_index]:
        mask[dep_idx] = 1.0
    return mask

class LearnedODEYeast(torch.jit.ScriptModule):
    """
    Hybrid physics + neural ODE for the 7-state yeast glycolysis oscillator.

    This class exists because yeast needs a dependency-masked NN input instead
    of the generic full-state learned wrapper.

    This class combines the known analytical dynamics of the yeast glycolysis
    model with a small learned MLP that replaces the derivative of one state
    whose first-principles equations are uncertain or unknown. The intent is to
    keep the well-understood biochemical structure intact while giving the model
    the flexibility to correct the part that is hard to derive from first
    principles.

    Architecture:
      - physics.rhs_true(state) provides dy/dt for all seven states analytically.
      - latent_model(masked_state_features) overrides the derivative of the replaced state.
      - input_mask (from get_yeast_input_mask) zeroes out states that do not
        causally influence the replaced derivative, enforcing a sparse,
        biologically-motivated input structure.

    The standard CKF/RTS interface is fully implemented:
      - f_full(state):        full hybrid RHS (physics + learned component)
      - rk4_step(state, dt):  RK4 integration with adaptive sub-stepping
      - rk4_rollout(state, dt, steps): multi-step rollout used during training
      - propagate_batch(X, dt): batched propagation for CKF sigma points
      - h(X):                 measurement model (all states except replaced)

    Args:
        physics: Frozen YeastPhysics module providing the analytical equations.
        latent_model: Small MLP that outputs d{x_replaced}/dt from masked inputs.
        max_dt: Maximum allowed RK4 sub-step size (s). Stiff regimes may
            require values well below the CKF observation interval.
        min_dt: Minimum sub-step size; 0.0 disables the lower bound.
        replaced_index: Index of the state whose dynamics are learned
            (default 3, corresponding to x4 — the 1,3-BPG intermediate).
    """

    __constants__ = ['replaced_index', 'n_states', 'stiff_corrector_iters']

    def __init__(
        self,
        physics: torch.jit.ScriptModule,
        latent_model: nn.Module,
        max_dt: float,
        min_dt: float,
        replaced_index: int = 3,  # x4
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()
        self.physics = physics
        self.latent_model = latent_model
        self.replaced_index: int = int(replaced_index)
        self.n_states: int = 7

        mask = get_yeast_input_mask(replaced_index)
        self.register_buffer("input_mask", mask)

        deps = YEAST_DEPENDENCIES[replaced_index]
        dep_names = [YEAST_STATE_NAMES[i] for i in deps]
        print(f"[LearnedODEYeast] Learning d{YEAST_STATE_NAMES[replaced_index]}/dt")
        print(f"  Dependencies: {dep_names} (indices {deps})")

        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))

    @torch.jit.script_method
    def f_full(self, state: Tensor) -> Tensor:
        """Return hybrid yeast derivatives for the current ODE state."""
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        # Compute all biochemical kinetics first, then replace the uncertain state derivative.
        rhs_phys = self.physics.rhs_true(state)  # (..., 7)
        out = rhs_phys.clone()

        # Feed only the biologically relevant dependency states into the neural ODE term.
        masked_state_features = state * self.input_mask
        idx = self.replaced_index
        # The NN output is d{x_idx}/dt at the current RK stage.
        d_learned = self.latent_model(masked_state_features)
        # Hybrid RHS: keep six analytical derivatives and swap in the learned one.
        out[..., idx:idx+1] = d_learned

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def _rk4_single(self, y: Tensor, h: float) -> Tensor:
        # Support the rk4 single step in this module.
        hh = torch.tensor(h, dtype=y.dtype, device=y.device)
        k1 = self.f_full(y)
        y2 = self.physics.clip_states(y + 0.5 * hh * k1)
        k2 = self.f_full(y2)
        y3 = self.physics.clip_states(y + 0.5 * hh * k2)
        k3 = self.f_full(y3)
        y4 = self.physics.clip_states(y + hh * k3)
        k4 = self.f_full(y4)
        y_next = y + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.physics.clip_states(y_next)

    @torch.jit.script_method
    def rk4_step(self, y: Tensor, dt: float) -> Tensor:
        # Support the rk4 step step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            return y.squeeze(0) if single else y

        sign = 1.0
        if total_dt < 0.0:
            sign = -1.0
            total_dt = -total_dt

        max_dt = float(self.max_dt.item())
        min_dt = float(self.min_dt.item())

        if min_dt > 0.0 and total_dt < min_dt:
            total_dt = min_dt

        n_sub = 1
        if max_dt > 0.0 and total_dt > max_dt:
            n_sub = int(total_dt / max_dt + 0.999999)

        h = sign * (total_dt / float(n_sub))

        out = y
        for _ in range(n_sub):
            out = self._rk4_single(out, h)

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: Tensor, dt: float) -> Tensor:
        # Propagate batch.
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: Tensor) -> Tensor:
        # Map model states into measurement space.
        return self.physics.h(X)

    @property
    def Q(self) -> Tensor:
        # Return the process-noise covariance used by filtering.
        return self.physics.Q

    @property
    def R(self) -> Tensor:
        # Return the measurement-noise covariance used by filtering.
        return self.physics.R

def make_experiment() -> ExperimentConfig:
    # Central preset for this model: physics, filter noise, NN size, training, and plotting.
    return ExperimentConfig(system_name='yeast', system_cfg=YeastSystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.002, max_dt=0.01, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(0.00035, 0.00035, 0.00035, 0.005, 5e-05, 0.00035, 0.00015), R_diag=(2.7e-07, 5.7e-06, 9e-08, 1e-09, 5e-07, 1.2e-08), P0_diag=(0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01)), nn_cfg=NNConfig(in_dim=7, out_dim=1, hidden_dims=(48, 48), activation='tanh', input_scale=(1.0, 1.0, 3.3, 1.2, 2.8, 1.4, 9.0), input_shift=None, lr=0.003, weight_decay=1e-06, grad_clip=5.0), train_cfg=TrainConfig(T_train=1.6667, meas_noise_std=0.0005, n_epochs=100, min_rollout_horizon=1, max_rollout_horizon=1, stride=2, use_pseudo_latent=True, pseudo_weight=2.0), plot_cfg=PlotConfig(plot_stride=20, show_every=5, max_rollouts=10), data_cfg=DataGenConfig(max_step_scipy=0.01, uniform_dt=0.004, seed=1234), state_labels=('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'), title_prefix='Yeast', meas_idx=(0, 1, 2, 4, 5, 6), pseudo_idx=(3,), uses_pin=False)

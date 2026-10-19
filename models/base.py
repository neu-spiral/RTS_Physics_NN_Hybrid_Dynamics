"""Shared base classes and utilities for ODE model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


torch.set_default_dtype(torch.float32)

@dataclass
class IntegratorConfig:
    """ODE step-size settings shared by physics and hybrid wrappers."""
    default_dt: float = 0.01
    max_dt: float = 0.02
    min_dt: float = 0.0
    n_substeps: int = 1

@dataclass
class CKFConfig:
    """Filter covariance settings kept next to each model preset."""
    Q_diag: tuple   # Process noise covariance (diagonal)
    R_diag: tuple   # Measurement noise covariance (diagonal)
    P0_diag: tuple  # Initial state uncertainty (diagonal)

@dataclass
class NNConfig:
    """Learned-derivative MLP settings for a model preset."""
    in_dim: int
    out_dim: int
    hidden_dims: tuple = (32, 32)
    activation: str = "tanh"
    input_scale: tuple = (1.0, 1.0)
    input_shift: tuple = None  # Optional: shift inputs before scaling
    lr: float = 5e-2
    weight_decay: float = 1e-6
    grad_clip: float = 5.0

@dataclass
class TrainConfig:
    """Rollout-training settings used by the main experiment scripts."""
    T_train: float = 5.0
    meas_noise_std: float = 0.05
    meas_snr: Optional[float] = None
    n_epochs: int = 10  # Changed from 100 for faster experiments
    min_rollout_horizon: int = 1
    max_rollout_horizon: int = 200
    stride: int = 1
    use_pseudo_latent: bool = True
    pseudo_weight: float = 0.1
    full_state_weight: float = 0.0

@dataclass
class PlotConfig:
    """Plot cadence and rollout-display settings."""
    plot_stride: int = 30
    show_every: int = 5
    max_rollouts: int = 15

@dataclass
class DataGenConfig:
    """Synthetic data-generation settings."""
    max_step_scipy: float = 0.0
    uniform_dt: float = 0.0
    keep_adaptive_grid: bool = False
    scipy_method: str = ""
    seed: int = 1234

@dataclass
class ExperimentConfig:
    """One object carrying all model, filter, NN, data, and plot settings."""
    system_name: str
    system_cfg: object
    integrator_cfg: IntegratorConfig
    ckf_cfg: CKFConfig
    nn_cfg: NNConfig
    train_cfg: TrainConfig
    plot_cfg: PlotConfig
    data_cfg: DataGenConfig

    # Metadata for training/plotting
    state_labels: tuple
    title_prefix: str
    meas_idx: tuple  # Which states are measured
    pseudo_idx: tuple  # Which states get pseudo loss
    uses_pin: bool = False  # Whether system uses Pin(t) drive



def _to_torch(t_np, y_np, return_torch=True, device=None):
    """
    t_np : (T,)
    y_np : (T, state_dim)
    returns t_torch, y_torch with shape (T, 1, state_dim)
    """
    t_np = t_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    if not return_torch:
        return t_np, y_np

    if device is None:
        device = torch.device("cpu")

    t_t = torch.tensor(t_np, device=device)
    y_t = torch.tensor(y_np, device=device).unsqueeze(1)  # (T, 1, D)
    return t_t, y_t

class BasePhysics2D(torch.jit.ScriptModule):
    """
    Base class for 2D dynamical systems (HO and Pendulum).

    This exists so small two-state systems share the same CKF and RK4 interface.

    State: y = [s0, s1]
    Measurement: h(y) = s0 (observe first state component only)

    Subclasses must implement:
    - rhs_true(state): System dynamics for the current state vector
    - clip_states(y): State bounds enforcement
    """

    __constants__ = []

    def __init__(self, integrator_cfg: IntegratorConfig, ckf_cfg: CKFConfig):
        # Initialize this object and store the configuration it needs later.
        super().__init__()

        # Noise covariances
        Q_diag = list(ckf_cfg.Q_diag)
        R_diag = list(ckf_cfg.R_diag)
        self.register_buffer("Q", torch.diag(torch.tensor(Q_diag)))
        self.register_buffer("R", torch.diag(torch.tensor(R_diag)))

        # Integrator parameters
        self.register_buffer("dt", torch.tensor(integrator_cfg.default_dt))
        self.register_buffer("max_dt", torch.tensor(integrator_cfg.max_dt))
        self.register_buffer("min_dt", torch.tensor(integrator_cfg.min_dt))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Enforce state bounds - must be implemented by subclass"""
        raise NotImplementedError

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        """System dynamics dy/dt = f(y) - must be implemented by subclass"""
        raise NotImplementedError

    @torch.jit.script_method
    def _rk4_single(self, state: torch.Tensor, h: float) -> torch.Tensor:
        """Single RK4 step from the current state vector."""
        hh = torch.tensor(h, dtype=state.dtype, device=state.device)
        k1 = self.rhs_true(state)
        k2 = self.rhs_true(state + 0.5 * hh * k1)
        k3 = self.rhs_true(state + 0.5 * hh * k2)
        k4 = self.rhs_true(state + hh * k3)
        y_next = state + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clip_states(y_next)

    @torch.jit.script_method
    def rk4_step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
        """
        RK4 integration with adaptive sub-stepping.

        Args:
            y: State (1D or 2D batch)
            dt: Time step (can be negative for backward integration)

        Returns:
            Updated state after time dt
        """
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = dt
        if total_dt == 0.0:
            return y.squeeze(0) if single else y

        # Handle negative dt
        sign = 1.0
        if total_dt < 0.0:
            sign = -1.0
            total_dt = -total_dt

        max_dt = float(self.max_dt.item())
        min_dt = float(self.min_dt.item())

        # Enforce minimum step size
        if min_dt > 0.0 and total_dt < min_dt:
            total_dt = min_dt

        # Adaptive sub-stepping
        n_sub = 1
        if max_dt > 0.0 and total_dt > max_dt:
            n_sub = int(total_dt / max_dt + 0.999999)
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt / float(n_sub)

        # Integrate with sub-steps
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
        """Measurement model: observe first state component"""
        return X[:, 0:1]

class LearnedODE2D(torch.jit.ScriptModule):
    """
    Generic 2D learned ODE:
      state = [s0, s1]
      s0' = physics-based (rhs_true)
      s1' = latent_model([s0, s1])

    Created when a two-state model keeps one derivative analytical and learns the other.
    """
    def __init__(
        self,
        physics: BasePhysics2D,
        latent_model: nn.Module,
        max_dt: float,
        min_dt: float,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()
        self.physics = physics
        self.latent_model = latent_model

        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))

    @torch.jit.script_method
    def f_full(self, state: Tensor) -> Tensor:
        """
        state: (..., 2)
            Current ODE state [s0, s1] at one RK stage.
        returns: (..., 2) [s0', s1']
        """
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        # Start from the analytical ODE so the known state keeps its trusted physics.
        rhs_phys = self.physics.rhs_true(state)  # (..., 2)
        ds0 = rhs_phys[..., 0:1]

        s0 = state[..., 0:1]
        s1 = state[..., 1:2]
        # The NN sees the current ODE state and supplies only the unknown derivative.
        state_features = torch.cat([s0, s1], dim=-1)  # (..., 2)
        ds1 = self.latent_model(state_features)       # (..., 1)

        # This is the hybrid RHS: physics for s0, learned dynamics for s1.
        out = torch.cat([ds0, ds1], dim=-1)
        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def _rk4_single(self, y: Tensor, h: float) -> Tensor:
        # Support the rk4 single step in this module.
        hh = torch.tensor(h, dtype=y.dtype, device=y.device)
        k1 = self.f_full(y)
        k2 = self.f_full(y + 0.5 * hh * k1)
        k3 = self.f_full(y + 0.5 * hh * k2)
        k4 = self.f_full(y + hh * k3)
        y_next = y + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y_next = self.physics.clip_states(y_next)
        return y_next

    @torch.jit.script_method
    def rk4_step(self, y: Tensor, dt: float) -> Tensor:
        # Support the rk4 step step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = dt
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
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt / float(n_sub)

        out = y
        for _ in range(n_sub):
            out = self._rk4_single(out, h)

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def rk4_rollout(self, y: Tensor, dt: float, steps: int) -> Tensor:
        # Support the rk4 rollout step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            out = y.unsqueeze(1).repeat(1, steps + 1, 1)
            return out[0] if single else out

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
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt / float(n_sub)

        B = y.shape[0]
        n = y.shape[1]
        out = torch.zeros((B, steps + 1, n), dtype=y.dtype, device=y.device)
        out[:, 0, :] = y

        yy = y
        for i in range(steps):
            for _ in range(n_sub):
                yy = self._rk4_single(yy, h)
            out[:, i + 1, :] = yy

        return out[0] if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: Tensor, dt: float) -> Tensor:
        """CKF interface: propagate batch of sigma points"""
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: Tensor) -> Tensor:
        """CKF interface: measurement model (delegate to physics)"""
        return self.physics.h(X)

    @property
    def Q(self) -> Tensor:
        """CKF interface: process noise covariance"""
        return self.physics.Q

    @property
    def R(self) -> Tensor:
        """CKF interface: measurement noise covariance"""
        return self.physics.R
class LearnedODE3D(torch.jit.ScriptModule):
    """
    Generic 3D learned ODE:
      state = [s0, s1, s2]
      Teacher: physics.rhs_true(state) -> (...,3)
      The derivative at `replaced_index` is replaced with NN(state).

    Example: For Brusselator [x, y, z], replace x dynamics (index 0)

    Created when a three-state model needs one learned derivative slot.
    """
    __constants__ = ['replaced_index']

    def __init__(
        self,
        physics: torch.jit.ScriptModule,
        latent_model: nn.Module,
        max_dt: float,
        min_dt: float,
        replaced_index: int = 0,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()
        self.physics = physics
        self.latent_model = latent_model
        self.replaced_index: int = int(replaced_index)

        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))

    @torch.jit.script_method
    def f_full(self, state: Tensor) -> Tensor:
        """
        state: (..., 3)
            Current ODE state [s0, s1, s2] at one RK stage.
        returns: (..., 3) hybrid derivatives
        """
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        # Evaluate the full teacher RHS first, then surgically replace one slot.
        rhs_phys = self.physics.rhs_true(state)  # (..., 3)
        out = rhs_phys.clone()

        # The latent model is used inside the ODE as a derivative model, not a state update.
        ds_rep = self.latent_model(state)  # (..., 1)
        idx = self.replaced_index
        # Keep every physics derivative except the configured learned component.
        out[..., idx:idx+1] = ds_rep

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def _rk4_single(self, y: Tensor, h: float) -> Tensor:
        # Support the rk4 single step in this module.
        hh = torch.tensor(h, dtype=y.dtype, device=y.device)
        k1 = self.f_full(y)
        k2 = self.f_full(y + 0.5 * hh * k1)
        k3 = self.f_full(y + 0.5 * hh * k2)
        k4 = self.f_full(y + hh * k3)
        y_next = y + (hh / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
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
    def rk4_rollout(self, y: Tensor, dt: float, steps: int) -> Tensor:
        # Support the rk4 rollout step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            out = y.unsqueeze(1).repeat(1, steps + 1, 1)
            return out[0] if single else out

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
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt / float(n_sub)

        B = y.shape[0]
        n = y.shape[1]
        out = torch.zeros((B, steps + 1, n), dtype=y.dtype, device=y.device)
        out[:, 0, :] = y

        yy = y
        for i in range(steps):
            for _ in range(n_sub):
                yy = self._rk4_single(yy, h)
            out[:, i + 1, :] = yy

        return out[0] if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: Tensor, dt: float) -> Tensor:
        """CKF interface: propagate batch of sigma points"""
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: Tensor) -> Tensor:
        """CKF interface: measurement model (delegate to physics)"""
        return self.physics.h(X)

    @property
    def Q(self) -> Tensor:
        """CKF interface: process noise covariance"""
        return self.physics.Q

    @property
    def R(self) -> Tensor:
        """CKF interface: measurement noise covariance"""
        return self.physics.R

class LearnedODE4D(torch.jit.ScriptModule):
    """
    Generic 4D learned ODE:
      state = [s0, s1, s2, s3]
      Teacher: physics.rhs_true(state) -> (...,4)
      The derivative at `replaced_index` is replaced with NN(state).

    Example: For Hodgkin-Huxley [V, m, h, n], replace m dynamics (index 1)

    Created when a four-state model needs one learned derivative slot.
    """
    __constants__ = ['replaced_index']

    def __init__(
        self,
        physics: torch.jit.ScriptModule,
        latent_model: nn.Module,
        max_dt: float,
        min_dt: float,
        replaced_index: int = 1,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()
        self.physics = physics
        self.latent_model = latent_model
        self.replaced_index: int = int(replaced_index)

        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))

    @torch.jit.script_method
    def f_full(self, state: Tensor) -> Tensor:
        """
        state: (..., 4)
            Current ODE state [s0, s1, s2, s3] at one RK stage.
        returns: (..., 4) hybrid derivatives
        """
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        # Evaluate the full teacher RHS first, then replace only the missing term.
        rhs_phys = self.physics.rhs_true(state)  # (..., 4)
        out = rhs_phys.clone()

        # The NN output is interpreted as d(state[idx])/dt at this RK stage.
        ds_rep = self.latent_model(state)  # (..., 1)
        idx = self.replaced_index
        # This splice is the actual hybrid ODE: analytical channels plus one learned channel.
        out[..., idx:idx+1] = ds_rep

        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def _rk4_single(self, y: Tensor, h: float) -> Tensor:
        # Support the rk4 single step in this module.
        hh = torch.tensor(h, dtype=y.dtype, device=y.device)
        k1 = self.f_full(y)
        k2 = self.f_full(y + 0.5 * hh * k1)
        k3 = self.f_full(y + 0.5 * hh * k2)
        k4 = self.f_full(y + hh * k3)
        y_next = y + (hh / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
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
    def rk4_rollout(self, y: Tensor, dt: float, steps: int) -> Tensor:
        # Support the rk4 rollout step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt = float(dt)
        if total_dt == 0.0:
            out = y.unsqueeze(1).repeat(1, steps + 1, 1)
            return out[0] if single else out

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
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt / float(n_sub)

        B = y.shape[0]
        n = y.shape[1]
        out = torch.zeros((B, steps + 1, n), dtype=y.dtype, device=y.device)
        out[:, 0, :] = y

        yy = y
        for i in range(steps):
            for _ in range(n_sub):
                yy = self._rk4_single(yy, h)
            out[:, i + 1, :] = yy

        return out[0] if single else out

    @torch.jit.script_method
    def propagate_batch(self, X: Tensor, dt: float) -> Tensor:
        """CKF interface: propagate batch of sigma points"""
        return self.rk4_step(X, dt)

    @torch.jit.script_method
    def h(self, X: Tensor) -> Tensor:
        """CKF interface: measurement model (delegate to physics)"""
        return self.physics.h(X)

    @property
    def Q(self) -> Tensor:
        """CKF interface: process noise covariance"""
        return self.physics.Q

    @property
    def R(self) -> Tensor:
        """CKF interface: measurement noise covariance"""
        return self.physics.R


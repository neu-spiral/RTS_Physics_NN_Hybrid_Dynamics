"""Template for adding a new ODE model.

Copy this file to models/<your_model>.py, then rename every NewModel/new_model
placeholder to your model name. After that, register the module in
models/__init__.py and export the useful classes/functions there.

Keep the Torch RHS in NewModelPhysics.rhs_true and the NumPy RHS inside
integrate_new_model_adaptive mathematically identical. The smoother/training
path uses the Torch version, while synthetic data generation uses the NumPy
version through solve_ivp.
"""

from __future__ import annotations

from dataclasses import dataclass

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
class NewModelSystemConfig:
    """State, bounds, and physical constants for the new model.

    This class is created so all model-specific knobs live in one typed config.
    """

    # True initial condition used for synthetic data generation.
    x0_true: float = 1.0
    z0_true: float = 0.0

    # Prior initial condition given to CKF/RTS. This can be intentionally
    # imperfect if you want the smoother to solve a harder estimation problem.
    x0_prior: float = 1.0
    z0_prior: float = 0.0

    # Bounds used by clip_states. Keep these wide enough for real dynamics but
    # finite enough to stop sigma points and rollouts from exploding.
    x_min: float = -10.0
    x_max: float = 10.0
    z_min: float = -10.0
    z_max: float = 10.0


class NewModelPhysics(BasePhysics2D):
    """Torch physics used by CKF/RTS, rollout plots, and learned ODE training.

    This class is the mechanistic model that the smoother trusts before any NN
    derivative is inserted.
    """

    def __init__(self, sys_cfg: NewModelSystemConfig, integrator_cfg: IntegratorConfig, ckf_cfg: CKFConfig):
        super().__init__(integrator_cfg, ckf_cfg)
        # Register constants as buffers so TorchScript and .to(device) move them
        # with the physics module.
        self.register_buffer("X_MIN", torch.tensor(sys_cfg.x_min))
        self.register_buffer("X_MAX", torch.tensor(sys_cfg.x_max))
        self.register_buffer("Z_MIN", torch.tensor(sys_cfg.z_min))
        self.register_buffer("Z_MAX", torch.tensor(sys_cfg.z_max))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        # Clamp each state channel independently. For unbounded states, use wide
        # limits instead of removing this guard.
        x = torch.clamp(y[..., 0:1], self.X_MIN, self.X_MAX)
        z = torch.clamp(y[..., 1:2], self.Z_MIN, self.Z_MAX)
        # Concatenate in state-vector order; CKF covariances and meas_idx depend on this order.
        return torch.cat([x, z], dim=-1)

    @torch.jit.script_method
    def rhs_true(self, y: torch.Tensor) -> torch.Tensor:
        # Full known ODE: y = [x, z]. Replace dx and dz with your equations.
        # If one derivative will be learned later, keep the physics version here
        # as the prior model used by the smoother.
        x = y[..., 0:1]
        z = y[..., 1:2]
        # Math form for this template: dx/dt = z, dz/dt = -x.
        dx = z
        dz = -x
        return torch.cat([dx, dz], dim=-1)


def integrate_new_model_adaptive(y0_np, t0, T, sys_cfg, max_step_scipy=0.0, uniform_dt=0.0, return_torch=True, device=None):
    """SciPy integrator used only to generate synthetic true trajectories."""
    y0_np = np.asarray(y0_np, dtype=np.float64)

    def rhs(_t, y):
        # Must match NewModelPhysics.rhs_true, but written in NumPy for solve_ivp.
        x, z = y
        # Same state order as rhs_true: first derivative is for x, second is for z.
        return np.array([z, -x], dtype=np.float64)

    kwargs = dict(method="RK45", rtol=1e-6, atol=1e-9)
    if max_step_scipy > 0.0:
        kwargs["max_step"] = max_step_scipy
    if uniform_dt > 0.0:
        # Force a regular output grid. Leave uniform_dt at 0.0 to keep the
        # adaptive solve_ivp time grid.
        n = int((T - t0) / uniform_dt + 0.5) + 1
        kwargs["t_eval"] = np.linspace(t0, T, n, endpoint=True)
    sol = solve_ivp(rhs, (float(t0), float(T)), y0_np, **kwargs)
    return _to_torch(sol.t, sol.y.T, return_torch=return_torch, device=device)


def make_data_new_model(exp: ExperimentConfig, device: torch.device):
    """Create synthetic measurements and true states for training/evaluation."""
    sys_cfg = exp.system_cfg
    tc = exp.train_cfg
    dc = exp.data_cfg
    t, y_true = integrate_new_model_adaptive(
        np.array([sys_cfg.x0_true, sys_cfg.z0_true], dtype=np.float64),
        0.0,
        tc.T_train,
        sys_cfg,
        max_step_scipy=dc.max_step_scipy,
        uniform_dt=dc.uniform_dt,
        return_torch=True,
        device=device,
    )
    if y_true.dim() == 3:
        y_true = y_true[:, 0, :]

    # Split true states so _choose_system can return the same simple tuple
    # pattern used by the built-in 2D examples.
    x_true = y_true[:, 0]
    z_true = y_true[:, 1]

    # Measurement example: observe x only, with relative Gaussian noise.
    # If your system measures more channels, stack them as (T, n_meas).
    torch.manual_seed(dc.seed)
    x_meas = x_true + tc.meas_noise_std * x_true.abs().mean().clamp(min=1e-6) * torch.randn_like(x_true)
    return t, x_meas, x_true, z_true


class LearnedODENewModel(torch.jit.ScriptModule):
    """Optional custom learned-ODE wrapper.

    For many 2D, 3D, or 4D models, use LearnedODE2D/LearnedODE3D/LearnedODE4D
    from models/base.py instead of writing a custom wrapper. Add custom logic
    only when your learned derivative needs a drive input, masks, residual
    terms, or special clipping.

    Create a class like this only when the generic wrappers are not enough.
    """

    def __init__(self, physics: torch.jit.ScriptModule, latent_model: nn.Module, max_dt: float, min_dt: float):
        super().__init__()
        # The wrapper should expose physics and latent_model because training,
        # plotting, and checkpoint loading expect those attributes.
        self.physics = physics
        self.latent_model = latent_model
        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))


def make_experiment() -> ExperimentConfig:
    """Default experiment preset loaded by models/__init__.py."""
    return ExperimentConfig(
        system_name="new_model",  # CLI/model key; also used in result folders.
        system_cfg=NewModelSystemConfig(),  # Initial states, bounds, and model constants.
        integrator_cfg=IntegratorConfig(
            default_dt=0.01,  # Nominal RK step for training/CKF propagation.
            max_dt=0.02,  # Largest internal RK step; smaller is safer but slower.
        ),
        ckf_cfg=CKFConfig(
            Q_diag=(1e-4, 1e-4),  # Process noise per state; length must equal n_state.
            R_diag=(1e-3,),  # Measurement noise per observed channel; length equals len(meas_idx).
            P0_diag=(1e-2, 1e-2),  # Initial covariance per state; length must equal n_state.
        ),
        nn_cfg=NNConfig(
            in_dim=2,  # NN input dimension. Usually n_state unless you add a drive or mask.
            out_dim=1,  # Number of learned derivatives. Usually len(pseudo_idx).
        ),
        train_cfg=TrainConfig(),  # Epochs, rollout horizons, stride, pseudo loss weight.
        plot_cfg=PlotConfig(),  # Plot cadence, plot stride, and number of rollouts to draw.
        data_cfg=DataGenConfig(seed=1234),  # Synthetic data seed and solve_ivp time-grid options.
        state_labels=("x", "z"),  # Display names; order must match the state vector.
        title_prefix="NewModel",  # Human-readable label in plots and saved figures.
        meas_idx=(0,),  # Observed state indices. Here x is measured and z is hidden.
        pseudo_idx=(1,),  # Hidden/learned state indices supervised by smoother estimates.
        uses_pin=False,  # Set True only if the model has an external drive sequence.
    )

# augmented_system.py
# Augmented system for CKF-based joint state + parameter estimation.
#
# The augmented state is x_aug = [w, s] where:
#   w: flattened NN weights (p,)
#   s: ODE state (n,)
#
# State transition:
#   w_{k+1} = w_k                             (identity / random walk)
#   s_{k+1} = s_k + dt * f_hybrid(s_k, w_k)  (RK4 integration)
#
# Measurement:
#   y_k = s_k[meas_idx]                       (observe selected states)

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class FunctionalMLP:
    """
    A functional (stateless) MLP that takes explicit weight vectors.

    Instead of storing parameters internally like nn.Module, this MLP
    takes a flat weight vector and uses it to compute the forward pass.
    This is essential for the augmented CKF where NN weights are part
    of the state vector.
    """

    def __init__(self, layer_sizes, activation="tanh",
                 input_scale=None, input_shift=None):
        """
        Parameters
        ----------
        layer_sizes : list of int
            [in_dim, hidden1, hidden2, ..., out_dim]
        activation : str
            "tanh" or "relu"
        input_scale : (in_dim,) tensor or None
        input_shift : (in_dim,) tensor or None
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.input_scale = input_scale
        self.input_shift = input_shift

        # Compute total number of parameters
        self.n_params = 0
        self.param_slices = []  # (start, end, shape) for each W and b
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Weight matrix
            w_size = fan_in * fan_out
            self.param_slices.append((self.n_params, self.n_params + w_size, (fan_out, fan_in)))
            self.n_params += w_size
            # Bias vector
            self.param_slices.append((self.n_params, self.n_params + fan_out, (fan_out,)))
            self.n_params += fan_out

    def forward(self, x, weights):
        """
        Forward pass with explicit weights.

        x : (..., in_dim)
        weights : (n_params,)

        Returns: (..., out_dim)
        """
        if self.input_shift is not None:
            x = x - self.input_shift
        if self.input_scale is not None:
            x = x * self.input_scale

        n_layers = len(self.layer_sizes) - 1
        for i in range(n_layers):
            w_start, w_end, w_shape = self.param_slices[2 * i]
            b_start, b_end, b_shape = self.param_slices[2 * i + 1]

            W = weights[w_start:w_end].reshape(w_shape)  # (fan_out, fan_in)
            b = weights[b_start:b_end]  # (fan_out,)

            x = x @ W.T + b

            # Activation for all layers except the last
            if i < n_layers - 1:
                if self.activation == "tanh":
                    x = torch.tanh(x)
                elif self.activation == "relu":
                    x = torch.relu(x)

        return x

    def init_weights(self, device=None, dtype=torch.float32):
        """
        Xavier uniform initialization for weights, zeros for biases.
        Returns flat (n_params,) tensor.
        """
        w = torch.zeros(self.n_params, dtype=dtype, device=device)
        n_layers = len(self.layer_sizes) - 1
        for i in range(n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w_start, w_end, _ = self.param_slices[2 * i]
            b_start, b_end, _ = self.param_slices[2 * i + 1]

            # Xavier uniform
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            w[w_start:w_end] = torch.randn(w_end - w_start, dtype=dtype, device=device) * std
            w[b_start:b_end] = 0.0
        return w


class AugmentedSystem:
    """
    Augmented dynamical system for CKF-based training.

    Combines known physics with a functional NN to create a hybrid ODE
    where the NN weights are part of the state vector.

    The NN replaces the derivative of one state component (the "learned" state).
    All other state derivatives come from the known physics.
    """

    def __init__(
        self,
        physics,
        functional_mlp: FunctionalMLP,
        meas_idx: List[int],
        replaced_idx: int,
        n_states: int,
        max_dt: float = 0.01,
        input_mask: Optional[torch.Tensor] = None,
        has_drive: bool = False,
        residual_on_physics: bool = False,
        residual_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        physics : torch.jit.ScriptModule
            Physics model with rhs_true(y) and clip_states(y).
        functional_mlp : FunctionalMLP
            Stateless MLP for the unknown derivative.
        meas_idx : list of int
            Indices of measured states.
        replaced_idx : int
            Index of the state whose derivative is replaced by the NN.
        n_states : int
            Total number of ODE states.
        max_dt : float
            Maximum integration sub-step size.
        input_mask : (n_states,) tensor or None
            Optional mask for NN input dependency filtering.
        has_drive : bool
            Whether the system has an exogenous drive signal (e.g., retina).
        """
        self.physics = physics
        self.mlp = functional_mlp
        self.meas_idx = meas_idx
        self.replaced_idx = replaced_idx
        self.n_states = n_states
        self.n_weights = functional_mlp.n_params
        self.max_dt = max_dt
        self.input_mask = input_mask
        self.has_drive = has_drive
        self.residual_on_physics = residual_on_physics
        self.residual_scale = residual_scale

        self._meas_idx_t = None  # lazy init for device

    def _get_meas_idx_tensor(self, device):
        # Return meas idx tensor.
        if self._meas_idx_t is None or self._meas_idx_t.device != device:
            self._meas_idx_t = torch.tensor(self.meas_idx, dtype=torch.long, device=device)
        return self._meas_idx_t

    def _physics_rhs_batch(self, s_batch, drive=None):
        """Compute physics RHS for a batch of states."""
        if self._looks_like_hh_state(s_batch):
            return self._hh_rhs_safe_batch(s_batch)

        if self.has_drive and drive is not None:
            if hasattr(self.physics, 'rhs_true_with_drive'):
                if drive.dim() == 1:
                    drive_batch = drive.unsqueeze(0).expand(s_batch.shape[0], -1)
                else:
                    drive_batch = drive
                return self.physics.rhs_true_with_drive(s_batch, drive_batch)
        rhs = self.physics.rhs_true(s_batch)
        if torch.all(torch.isfinite(rhs)):
            return rhs
        return torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    def _looks_like_hh_state(self, s_batch):
        """Detect HH state layout [V, m, h, n] based on attached physics buffers."""
        return (
            self.n_states == 4
            and self.replaced_idx == 1
            and hasattr(self.physics, "C_m")
            and hasattr(self.physics, "g_Na")
            and hasattr(self.physics, "g_K")
            and hasattr(self.physics, "g_L")
            and hasattr(self.physics, "E_Na")
            and hasattr(self.physics, "E_K")
            and hasattr(self.physics, "E_L")
            and hasattr(self.physics, "I_ext")
        )

    def _hh_rhs_safe_batch(self, s_batch):
        """Numerically safe HH RHS to avoid singularities near V=-40 and V=-55."""
        V = s_batch[:, 0:1]
        m = s_batch[:, 1:2]
        h = s_batch[:, 2:3]
        n = s_batch[:, 3:4]

        x_m = (V + 40.0) / 10.0
        x_m_clip = torch.clamp(x_m, -50.0, 50.0)
        denom_m = -torch.expm1(-x_m_clip)
        denom_m_safe = torch.where(
            torch.abs(denom_m) > 1e-12,
            denom_m,
            torch.where(denom_m >= 0.0, torch.full_like(denom_m, 1e-12), torch.full_like(denom_m, -1e-12)),
        )
        alpha_m = torch.where(
            torch.abs(x_m) > 1e-7,
            x_m_clip / denom_m_safe,
            torch.ones_like(x_m),
        )
        beta_m = 4.0 * torch.exp(torch.clamp(-(V + 65.0) / 18.0, -50.0, 50.0))

        alpha_h = 0.07 * torch.exp(torch.clamp(-(V + 65.0) / 20.0, -50.0, 50.0))
        beta_h = 1.0 / (1.0 + torch.exp(torch.clamp(-(V + 35.0) / 10.0, -50.0, 50.0)))

        x_n = (V + 55.0) / 10.0
        x_n_clip = torch.clamp(x_n, -50.0, 50.0)
        denom_n = -torch.expm1(-x_n_clip)
        denom_n_safe = torch.where(
            torch.abs(denom_n) > 1e-12,
            denom_n,
            torch.where(denom_n >= 0.0, torch.full_like(denom_n, 1e-12), torch.full_like(denom_n, -1e-12)),
        )
        alpha_n = torch.where(
            torch.abs(x_n) > 1e-7,
            0.1 * x_n_clip / denom_n_safe,
            torch.full_like(x_n, 0.1),
        )
        beta_n = 0.125 * torch.exp(torch.clamp(-(V + 65.0) / 80.0, -50.0, 50.0))

        dm = alpha_m * (1.0 - m) - beta_m * m
        dh = alpha_h * (1.0 - h) - beta_h * h
        dn = alpha_n * (1.0 - n) - beta_n * n

        I_Na = self.physics.g_Na * (m ** 3) * h * (V - self.physics.E_Na)
        I_K = self.physics.g_K * (n ** 4) * (V - self.physics.E_K)
        I_L = self.physics.g_L * (V - self.physics.E_L)
        dV = (self.physics.I_ext - I_Na - I_K - I_L) / self.physics.C_m

        rhs = torch.cat([dV, dm, dh, dn], dim=-1)
        return torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    def _nn_input(self, s_batch, drive=None):
        """Prepare NN input from state batch."""
        if self.input_mask is not None:
            nn_in = s_batch * self.input_mask.unsqueeze(0)
        else:
            nn_in = s_batch
        if self.has_drive and drive is not None:
            if drive.dim() == 1:
                drive_exp = drive.unsqueeze(0).expand(s_batch.shape[0], -1)
            else:
                drive_exp = drive
            nn_in = torch.cat([nn_in, drive_exp], dim=-1)
        return nn_in

    def _merge_replaced_rhs(self, dsdt, nn_out):
        """Inject the learned term into the replaced state derivative."""
        nn_term = nn_out.reshape(dsdt.shape[0], 1) if dsdt.dim() == 2 else nn_out.reshape(1)
        if self.residual_on_physics:
            dsdt[:, self.replaced_idx:self.replaced_idx+1] = (
                dsdt[:, self.replaced_idx:self.replaced_idx+1] + self.residual_scale * nn_term
            )
        else:
            dsdt[:, self.replaced_idx:self.replaced_idx+1] = self.residual_scale * nn_term
        return dsdt

    def f_hybrid(self, s, w, drive=None):
        """
        Compute hybrid ODE RHS: known physics + NN for unknown component.

        s : (..., n_states)   ODE states
        w : (n_weights,)      NN weights
        drive : (..., d) or None

        Returns: (..., n_states) derivatives
        """
        squeeze = s.dim() == 1
        if squeeze:
            s = s.unsqueeze(0)

        # Build the known part of the ODE first; the NN only fills the missing derivative.
        dsdt = self._physics_rhs_batch(s, drive).clone()
        # The NN input is the current ODE state, optionally masked and augmented with drive.
        nn_input = self._nn_input(s, drive)
        # In RBSE the network weights live in the augmented state, so w changes per sigma point.
        nn_out = self.mlp.forward(nn_input, w)
        # Merge the learned derivative into the physics RHS to form f_hybrid(s, w).
        dsdt = self._merge_replaced_rhs(dsdt, nn_out)

        return dsdt.squeeze(0) if squeeze else dsdt

    def _rk4_step(self, s, w, dt_step, drive=None):
        """Single RK4 step for the hybrid ODE."""
        k1 = self.f_hybrid(s, w, drive)
        k2 = self.f_hybrid(s + 0.5 * dt_step * k1, w, drive)
        k3 = self.f_hybrid(s + 0.5 * dt_step * k2, w, drive)
        k4 = self.f_hybrid(s + dt_step * k3, w, drive)
        return s + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate_batch(self, X_aug, dt, drive=None):
        """
        Propagate a batch of augmented sigma points.

        Each sigma point has its own weight vector (since weights are part
        of the augmented state). Physics RHS is batched; NN is evaluated
        per-sigma-point since each has different weights.

        X_aug : (M, n_aug)
        dt : float
        drive : (d,) or None

        Returns: (M, n_aug)
        """
        M = X_aug.shape[0]
        n_w = self.n_weights

        W_batch = X_aug[:, :n_w]  # (M, n_w)
        S_batch = X_aug[:, n_w:]  # (M, n_s)

        n_sub = max(1, int(abs(dt) / self.max_dt + 0.999))
        dt_sub = dt / n_sub

        s = S_batch
        for _ in range(n_sub):
            # Physics RHS is batched because all sigma points share the same mechanistic model.
            dsdt_phys = self._physics_rhs_batch(s, drive)  # (M, n_s)

            # NN: loop because each sigma point carries a different flattened weight vector.
            nn_input = self._nn_input(s, drive)  # (M, in_dim)
            nn_outs = torch.stack([
                self.mlp.forward(nn_input[i:i+1], W_batch[i]).squeeze(0)
                for i in range(M)
            ], dim=0)  # (M, 1)

            k1 = dsdt_phys.clone()
            # k1 is physics plus the learned derivative evaluated at the current state.
            k1 = self._merge_replaced_rhs(k1, nn_outs)

            # k2
            s2 = s + 0.5 * dt_sub * k1
            dsdt2 = self._physics_rhs_batch(s2, drive)
            nn_input2 = self._nn_input(s2, drive)
            nn_outs2 = torch.stack([
                self.mlp.forward(nn_input2[i:i+1], W_batch[i]).squeeze(0)
                for i in range(M)
            ], dim=0)
            k2 = dsdt2.clone()
            # k2 repeats the same hybrid splice at the first midpoint state.
            k2 = self._merge_replaced_rhs(k2, nn_outs2)

            # k3
            s3 = s + 0.5 * dt_sub * k2
            dsdt3 = self._physics_rhs_batch(s3, drive)
            nn_input3 = self._nn_input(s3, drive)
            nn_outs3 = torch.stack([
                self.mlp.forward(nn_input3[i:i+1], W_batch[i]).squeeze(0)
                for i in range(M)
            ], dim=0)
            k3 = dsdt3.clone()
            # k3 uses the NN again because the midpoint state has changed.
            k3 = self._merge_replaced_rhs(k3, nn_outs3)

            # k4
            s4 = s + dt_sub * k3
            dsdt4 = self._physics_rhs_batch(s4, drive)
            nn_input4 = self._nn_input(s4, drive)
            nn_outs4 = torch.stack([
                self.mlp.forward(nn_input4[i:i+1], W_batch[i]).squeeze(0)
                for i in range(M)
            ], dim=0)
            k4 = dsdt4.clone()
            # k4 completes RK4 with the learned derivative at the endpoint state.
            k4 = self._merge_replaced_rhs(k4, nn_outs4)

            s = s + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Clip states
        s = self.clip_states_batch(s)

        # Weights unchanged (random walk)
        return torch.cat([W_batch, s], dim=1)

    def propagate_single(self, x_aug, dt, drive=None):
        """
        Propagate a single augmented state vector forward by dt.

        x_aug : (n_aug,) = [w, s]
        Returns: (n_aug,)
        """
        return self.propagate_batch(x_aug.unsqueeze(0), dt, drive).squeeze(0)

    def measure(self, x_aug):
        """
        Extract measurements from augmented state.

        x_aug : (n_aug,)
        Returns: (m_dim,)
        """
        s = x_aug[self.n_weights:]
        idx = self._get_meas_idx_tensor(s.device)
        return s[idx]

    def measure_batch(self, X_aug):
        """
        Extract measurements from batch of augmented states.

        X_aug : (M, n_aug)
        Returns: (M, m_dim)
        """
        S = X_aug[:, self.n_weights:]
        idx = self._get_meas_idx_tensor(S.device)
        return S[:, idx]

    def clip_states(self, s):
        """Clip ODE states to physical bounds (single)."""
        if s.dim() == 1:
            s_b = s.unsqueeze(0)
            return self.physics.clip_states(s_b).squeeze(0)
        return self.physics.clip_states(s)

    def clip_states_batch(self, s):
        """Clip ODE states to physical bounds (batch)."""
        return self.physics.clip_states(s)

    def extract_weights(self, x_aug):
        """Extract NN weights from augmented state."""
        return x_aug[:self.n_weights]

    def extract_states(self, x_aug):
        """Extract ODE states from augmented state."""
        return x_aug[self.n_weights:]

    def get_initial_augmented_state(self, w0, s0):
        """Combine initial weights and states into augmented vector."""
        return torch.cat([w0, s0], dim=0)


def build_augmented_system(
    physics,
    nn_config,
    meas_idx,
    replaced_idx,
    n_states,
    max_dt=0.01,
    input_mask=None,
    has_drive=False,
    residual_on_physics=False,
    residual_scale=1.0,
    device=None,
    dtype=torch.float32,
):
    """
    Build an AugmentedSystem from physics and NN configuration.

    Parameters
    ----------
    physics : Physics module
    nn_config : object with in_dim, out_dim, hidden_dims, activation,
                input_scale, input_shift
    meas_idx : list of int
    replaced_idx : int
    n_states : int
    max_dt : float
    input_mask : tensor or None
    has_drive : bool
    device : torch.device
    dtype : torch.dtype

    Returns
    -------
    aug_sys : AugmentedSystem
    w0 : (n_weights,) initial weights
    """
    hidden = nn_config.hidden_dims
    if isinstance(hidden, int):
        hidden = (hidden,)

    layer_sizes = [nn_config.in_dim] + list(hidden) + [nn_config.out_dim]

    input_scale = None
    if nn_config.input_scale is not None:
        input_scale = torch.tensor(nn_config.input_scale, dtype=dtype, device=device)

    input_shift = None
    if nn_config.input_shift is not None:
        input_shift = torch.tensor(nn_config.input_shift, dtype=dtype, device=device)

    mlp = FunctionalMLP(
        layer_sizes=layer_sizes,
        activation=nn_config.activation,
        input_scale=input_scale,
        input_shift=input_shift,
    )

    if input_mask is not None and not isinstance(input_mask, torch.Tensor):
        input_mask = torch.tensor(input_mask, dtype=dtype, device=device)

    aug_sys = AugmentedSystem(
        physics=physics,
        functional_mlp=mlp,
        meas_idx=meas_idx,
        replaced_idx=replaced_idx,
        n_states=n_states,
        max_dt=max_dt,
        input_mask=input_mask,
        has_drive=has_drive,
        residual_on_physics=residual_on_physics,
        residual_scale=residual_scale,
    )

    w0 = mlp.init_weights(device=device, dtype=dtype)

    return aug_sys, w0

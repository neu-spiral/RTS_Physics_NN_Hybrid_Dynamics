"""Retina blood-flow model.

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
class RetinaSystemConfig:
    """Retina pressure-network constants, initial states, and measurement choices."""
    # Cardiovascular waveform parameters (SBP/DBP adjusted to get Pin(0)???65)
    sbp: float = 100.0  # Systolic blood pressure
    dbp: float = 60.0   # Diastolic blood pressure
    iop: float = 10.0   # Intraocular pressure
    hr: float = 60.0    # Heart rate

    # Initial states
    p1_true: float = 33.0
    p2_true: float = 24.0
    p4_true: float = 17.0
    p5_true: float = 15.5

    p1_prior: float = 33.0
    p2_prior: float = 24.0
    p4_prior: float = 17.0
    p5_prior: float = 15.5

    # System parameters
    p_out: float = 10.0
    replaced_index: int = 2  # P4 dynamics replaced by NN (index 2 in [P1, P2, P4, P5])
    meas_state_indices: tuple = (0, 1, 3)  # Measure P1, P2, P5 (P4 is learned)
    init_csv_path: str = "Initial_wIOP.csv"
    # Scaling to O(1) magnitudes (also used by SNR-based R override in training)
    state_scales: tuple = (0.03, 0.04, 0.06, 0.06)  # [P1, P2, P4, P5]

    # Physics constraints
    pin_const: float = 0.0
    p_min: float = 0.0
    p_max: float = 120.0

class RetinaPhysics(torch.jit.ScriptModule):
    """
    Retina blood flow model based on Guidoboni et al.

    This class owns the pressure-flow circuit and measurement mapping; the
    learned wrapper only replaces selected pressure derivatives.

    State: y = [P1, P2, P4, P5] (pressures in mmHg)
    - P1: Central retinal artery pressure
    - P2: Arteriole pressure
    - P4: Venule pressure
    - P5: Central retinal vein pressure

    Measurement: h(y) = [P1, P2, P5] (all except P4)

    The model includes:
    - Fixed resistances (R_in, R_1a, R_1b, R_2a, R_2b, R_3a, R_3b, R_5c, R_5d, R_out)
    - Variable resistances (R_1c, R_1d, R_4a, R_4b, R_5a, R_5b) that depend on transmural pressure
    - Capacitances (C1, C2, C4, C5)
    - Time-varying inlet pressure Pin(t) from cardiac cycle
    """

    __constants__ = ["learns_p2"]

    def __init__(
        self,
        sys_cfg: RetinaSystemConfig,
        integrator_cfg: IntegratorConfig,
        ckf_cfg: CKFConfig,
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()

        # Measurement channels are all states not learned by the NN unless the
        # config provides an explicit measurement set.
        meas_idx = list(getattr(sys_cfg, "meas_state_indices", (0, 1, 3)))

        # Noise covariances
        Q_diag = list(ckf_cfg.Q_diag)
        R_full = list(ckf_cfg.R_diag)
        if len(R_full) == 4:
            R_diag = [R_full[i] for i in meas_idx]
        elif len(R_full) == len(meas_idx):
            R_diag = R_full
        elif len(R_full) == 3:
            old_meas_idx = [0, 1, 3]
            if all(i in old_meas_idx for i in meas_idx):
                R_diag = [R_full[old_meas_idx.index(i)] for i in meas_idx]
            else:
                raise ValueError(
                    f"Retina R_diag old measured-order indices {old_meas_idx} cannot cover meas_idx {meas_idx}"
                )
        else:
            raise ValueError(f"Retina R_diag length {len(R_full)} does not match meas_idx {len(meas_idx)}")
        self.register_buffer("Q", torch.diag(torch.tensor(Q_diag, dtype=torch.float32)))
        self.register_buffer("R", torch.diag(torch.tensor(R_diag, dtype=torch.float32)))

        # Integrator parameters
        self.register_buffer("default_dt", torch.tensor(integrator_cfg.default_dt))
        self.register_buffer("max_dt", torch.tensor(integrator_cfg.max_dt))
        self.register_buffer("min_dt", torch.tensor(integrator_cfg.min_dt))
        self.n_substeps: int = int(integrator_cfg.n_substeps)

        # Pressure bounds
        self.learns_p2: bool = 1 not in meas_idx
        self.register_buffer("meas_idx", torch.tensor(meas_idx, dtype=torch.long))
        self.register_buffer("P_MIN", torch.tensor(sys_cfg.p_min))
        self.register_buffer("P_MAX", torch.tensor(sys_cfg.p_max))

        # System parameters
        self.register_buffer("P_out", torch.tensor(sys_cfg.p_out))
        self.register_buffer("PIN_CONST", torch.tensor(sys_cfg.pin_const))

        # ---------------------------------------------------------------------
        # Fixed Resistances [mmHg·s/mL]
        # CRITICAL: These values MUST match integrate_retina_adaptive exactly!
        # ---------------------------------------------------------------------
        self.register_buffer("R_in", torch.tensor(2.25e4))
        self.register_buffer("R_1a", torch.tensor(4.30e3))
        self.register_buffer("R_1b", torch.tensor(4.30e3))
        self.register_buffer("R_2a", torch.tensor(6.00e3))
        self.register_buffer("R_2b", torch.tensor(6.00e3))
        self.register_buffer("R_3a", torch.tensor(5.68e3))
        self.register_buffer("R_3b", torch.tensor(5.68e3))
        self.register_buffer("R_5c", torch.tensor(1.35e3))
        self.register_buffer("R_5d", torch.tensor(1.35e3))
        self.register_buffer("R_out", torch.tensor(5.74e3))

        # ---------------------------------------------------------------------
        # Capacitances [mL/mmHg]
        # ---------------------------------------------------------------------
        self.register_buffer("C1", torch.tensor(7.22e-7))
        self.register_buffer("C2", torch.tensor(7.53e-7))
        self.register_buffer("C4", torch.tensor(1.67e-5))
        self.register_buffer("C5", torch.tensor(1.07e-5))

        # ---------------------------------------------------------------------
        # Variable Resistance Parameters
        # These compute pressure-dependent resistances R(ΔP) = (1/k0) * (1 + ΔP/(Kp*Kl))^(-4)
        # ---------------------------------------------------------------------

        # Common parameters
        v = 0.49  # Poisson's ratio

        # CRA (Central Retinal Artery)
        mu_CRA = 3.0 * 1e-3 * 0.00750062
        E_CRA = 0.3 * 1e6 * 0.00750062
        d_CRA_ref = 175e-4
        r_CRA_ref = d_CRA_ref / 2.0
        A_ref_CRA = math.pi * (r_CRA_ref ** 2)
        h_CRA = 39.7239e-4
        L_1c, L_1d = 0.2e-1, 1e-1

        K_l_CRA = 12.0 * A_ref_CRA / (math.pi * (h_CRA ** 2))
        K_p_CRA = (1.0 / 12.0) * (E_CRA * (h_CRA ** 3) / (1.0 - v ** 2)) * (math.pi / A_ref_CRA) ** 1.5
        k0_CRA_1c = A_ref_CRA ** 2 / (8.0 * math.pi * mu_CRA * L_1c)
        k0_CRA_1d = A_ref_CRA ** 2 / (8.0 * math.pi * mu_CRA * L_1d)

        self.register_buffer("K_l_CRA", torch.tensor(K_l_CRA))
        self.register_buffer("K_p_CRA", torch.tensor(K_p_CRA))
        self.register_buffer("k0_CRA_1c", torch.tensor(k0_CRA_1c))
        self.register_buffer("k0_CRA_1d", torch.tensor(k0_CRA_1d))

        # CRV (Central Retinal Vein)
        mu_CRV = 3.24 * 1e-3 * 0.00750062
        E_CRV = 0.6 * 1e6 * 0.00750062
        d_CRV_ref = 238e-4
        r_CRV_ref = d_CRV_ref / 2.0
        A_ref_CRV = math.pi * (r_CRV_ref ** 2)
        h_CRV = 10.7e-4  # FIXED to match data generation
        L_5a, L_5b = 1e-1, 0.2e-1  # FIXED order to match data generation

        K_l_CRV = 12.0 * A_ref_CRV / (math.pi * (h_CRV ** 2))
        K_p_CRV = (1.0 / 12.0) * (E_CRV * (h_CRV ** 3) / (1.0 - v ** 2)) * (math.pi / A_ref_CRV) ** 1.5
        k0_CRV_5a = A_ref_CRV ** 2 / (8.0 * math.pi * mu_CRV * L_5a)
        k0_CRV_5b = A_ref_CRV ** 2 / (8.0 * math.pi * mu_CRV * L_5b)

        self.register_buffer("K_l_CRV", torch.tensor(K_l_CRV))
        self.register_buffer("K_p_CRV", torch.tensor(K_p_CRV))
        self.register_buffer("k0_CRV_5a", torch.tensor(k0_CRV_5a))
        self.register_buffer("k0_CRV_5b", torch.tensor(k0_CRV_5b))

        # VEN (Venules)
        mu_VEN = 3.24 * 1e-3 * 0.00750062
        E_VEN = 0.066 * 1e6 * 0.00750062  # FIXED to match data generation
        d_VEN_ref = 0.015476685724901  # FIXED to match data generation
        r_VEN_ref = d_VEN_ref / 2.0
        A_ref_VEN = math.pi * (r_VEN_ref ** 2)
        h_VEN = d_VEN_ref / 20.0  # FIXED to match data generation
        L_VEN = 0.413520
        L_4a, L_4b = L_VEN / 2.0, L_VEN / 2.0  # FIXED to match data generation

        K_l_VEN = 12.0 * A_ref_VEN / (math.pi * (h_VEN ** 2))
        K_p_VEN = (1.0 / 12.0) * (E_VEN * (h_VEN ** 3) / (1.0 - v ** 2)) * (math.pi / A_ref_VEN) ** 1.5
        k0_VEN_4a = A_ref_VEN ** 2 / (8.0 * math.pi * mu_VEN * L_4a)
        k0_VEN_4b = A_ref_VEN ** 2 / (8.0 * math.pi * mu_VEN * L_4b)

        self.register_buffer("K_l_VEN", torch.tensor(K_l_VEN))
        self.register_buffer("K_p_VEN", torch.tensor(K_p_VEN))
        self.register_buffer("k0_VEN_4a", torch.tensor(k0_VEN_4a))
        self.register_buffer("k0_VEN_4b", torch.tensor(k0_VEN_4b))

        # External pressure on vessels
        self.register_buffer("C2TEN", torch.tensor(10.0))

    @torch.jit.script_method
    def clip_states(self, y: torch.Tensor) -> torch.Tensor:
        """Clamp pressures to physically valid range with P4-specific bounds.
        State: [P1, P2, P4, P5]
        P4 (index 2) has relaxed bounds [5, 50] - capillary pressure (wider to allow smoother flexibility)
        """
        if y.dim() == 1:
            # Single state vector
            y_clipped = y.clone()
            y_clipped[0] = torch.clamp(y[0], self.P_MIN, self.P_MAX)  # P1
            y_clipped[1] = torch.clamp(y[1], self.P_MIN, self.P_MAX)  # P2
            y_clipped[2] = torch.clamp(y[2], 5.0, 50.0)  # P4: [5, 50] - wider bounds for learned state
            y_clipped[3] = torch.clamp(y[3], self.P_MIN, self.P_MAX)  # P5
            return y_clipped
        else:
            # Batch of state vectors (N, 4)
            y_clipped = torch.clamp(y, self.P_MIN, self.P_MAX)
            if y.shape[-1] >= 3:
                y_clipped[..., 2] = torch.clamp(y[..., 2], 5.0, 50.0)  # P4: [5, 50] - wider bounds for learned state
            return y_clipped

    @torch.jit.script_method
    def _compute_resistances(
        self,
        deltaP1c: torch.Tensor,
        deltaP1d: torch.Tensor,
        deltaP4a: torch.Tensor,
        deltaP4b: torch.Tensor,
        deltaP5a: torch.Tensor,
        deltaP5b: torch.Tensor,
    ):
        """
        Compute pressure-dependent variable resistances.

        Formula: R(ΔP) = (1/k0) * [1 + ΔP/(Kp*Kl)]^(-4)

        where ΔP is transmural pressure (pressure inside - pressure outside)
        """
        term_1c = 1.0 + deltaP1c / (self.K_p_CRA * self.K_l_CRA)
        term_1d = 1.0 + deltaP1d / (self.K_p_CRA * self.K_l_CRA)
        R_1c = (1.0 / self.k0_CRA_1c) * term_1c ** (-4.0)
        R_1d = (1.0 / self.k0_CRA_1d) * term_1d ** (-4.0)

        # VEN resistances
        term_4a = 1.0 + deltaP4a / (self.K_p_VEN * self.K_l_VEN)
        term_4b = 1.0 + deltaP4b / (self.K_p_VEN * self.K_l_VEN)
        R_4a = (1.0 / self.k0_VEN_4a) * term_4a ** (-4.0)
        R_4b = (1.0 / self.k0_VEN_4b) * term_4b ** (-4.0)

        # CRV resistances
        term_5a = 1.0 + deltaP5a / (self.K_p_CRV * self.K_l_CRV)
        term_5b = 1.0 + deltaP5b / (self.K_p_CRV * self.K_l_CRV)
        R_5a = (1.0 / self.k0_CRV_5a) * term_5a ** (-4.0)
        R_5b = (1.0 / self.k0_CRV_5b) * term_5b ** (-4.0)

        return R_1c, R_1d, R_4a, R_4b, R_5a, R_5b

    @torch.jit.script_method
    def rhs_true_with_drive(self, y: Tensor, pin: Tensor) -> Tensor:
        """
        System dynamics with time-varying inlet pressure Pin(t).

        Args:
            y: State [P1, P2, P4, P5] - shape (..., 4)
            pin: Inlet pressure Pin(t) - scalar or tensor

        Returns:
            Time derivatives [dP1/dt, dP2/dt, dP4/dt, dP5/dt]

        Dynamics (Kirchhoff's current law at each node):
            dP1/dt = (1/C1) * [F_in - F_1]
            dP2/dt = (1/C2) * [F_1 - F_2]
            dP4/dt = (1/C4) * [F_2 - F_4]
            dP5/dt = (1/C5) * [F_4 - F_out]

        where flows F are driven by pressure differences across resistances.
        """
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        # Broadcast Pin to correct shape
        if pin.dim() == 0:
            Pin = torch.ones_like(y[..., :1]) * pin
        elif pin.dim() == 1:
            Pin = pin.unsqueeze(-1)
        else:
            Pin = pin

        # Extract state components
        P1 = y[..., 0:1]
        P2 = y[..., 1:2]
        P4 = y[..., 2:3]
        P5 = y[..., 3:4]

        # Transmural pressures (internal - external)
        c_2 = self.C2TEN
        deltaP1c = P1 - c_2
        deltaP1d = P1 - c_2
        deltaP4a = P4 - c_2
        deltaP4b = P4 - c_2
        deltaP5a = P5 - c_2
        deltaP5b = P5 - c_2

        # Compute variable resistances
        R_1c, R_1d, R_4a, R_4b, R_5a, R_5b = self._compute_resistances(
            deltaP1c, deltaP1d, deltaP4a, deltaP4b, deltaP5a, deltaP5b
        )

        # Fixed resistances
        R_in = self.R_in
        R_1a = self.R_1a
        R_1b = self.R_1b
        R_2a = self.R_2a
        R_2b = self.R_2b
        R_3a = self.R_3a
        R_3b = self.R_3b
        R_5c = self.R_5c
        R_5d = self.R_5d
        R_out = self.R_out

        # Capacitances
        C1 = self.C1
        C2 = self.C2
        C4 = self.C4
        C5 = self.C5
        P_out = self.P_out

        # Math form: dPi/dt = (inflow_i - outflow_i) / C_i for states [P1, P2, P4, P5].
        # Pressure derivatives (conservation of flow at each node)
        dP1 = ((Pin - P1) / (R_in + R_1a) - (P1 - P2) / (R_1b + R_1c + R_1d + R_2a)) / C1
        dP2 = ((P1 - P2) / (R_1b + R_1c + R_1d + R_2a) - (P2 - P4) / (R_2b + R_3a + R_3b + R_4a)) / C2
        dP4 = ((P2 - P4) / (R_2b + R_3a + R_3b + R_4a) - (P4 - P5) / (R_4b + R_5a + R_5b + R_5c)) / C4
        dP5 = ((P4 - P5) / (R_4b + R_5a + R_5b + R_5c) - (P5 - P_out) / (R_5d + R_out)) / C5

        out = torch.cat([dP1, dP2, dP4, dP5], dim=-1)
        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def rhs_true(self, y: Tensor) -> Tensor:
        """
        System dynamics with constant inlet pressure.

        This is a fallback for when Pin(t) is not provided.
        Uses PIN_CONST from configuration.
        """
        return self.rhs_true_with_drive(y, self.PIN_CONST)

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
        """
        RK4 integration with adaptive sub-stepping.

        Args:
            y: State [P1, P2, P4, P5]
            dt: Time step

        Returns:
            Updated state
        """
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
        Measurement model: observe the configured measured pressure states.

        Default is [P1, P2, P5]. Retina2 uses [P1, P5] while learning P2 and P4.

        Args:
            X: States [..., 4] = [P1, P2, P4, P5]

        Returns:
            Measurements [..., n_meas]
        """
        # Use meas_idx so retina and retina2 can share physics while observing different nodes.
        return X.index_select(-1, self.meas_idx)

def integrate_retina_adaptive(
    y0_np,
    t0,
    T,
    Pinput=None,
    max_step_scipy=0.0,
    uniform_dt=0.0,
    return_torch=True,
    device=None,
):
    """
    Retina lumped model integration.

    y0_np   : [P1_0, P2_0, P4_0, P5_0]
    Pinput  : [SBP, DBP, IOP, HR], if None uses [100, 60, 10, 60].

    Returns
    -------
    if return_torch:
        t_torch      : (T,)
        P_state      : (T, 1, 4)  [P1,P2,P4,P5]
        Pin_seq_torch: (T,)
    else:
        t_np         : (T,)
        P_state_np   : (T, 4)
        Pin_seq_np   : (T,)

    Generates true pressure trajectories with the same equations used by RetinaPhysics.
    """
    y0_np = np.asarray(y0_np, dtype=np.float64)

    if Pinput is None:
        Pinput = np.array([100.0, 60.0, 10.0, 60.0], dtype=np.float64)  # SBP, DBP, IOP, HR
    else:
        Pinput = np.asarray(Pinput, dtype=np.float64)
    SP, DP, IOP, HR = Pinput

    # output pressure
    P_out = 14.0

    # ---- Fixed resistances [mmHg s/mL] --------------------------------------
    R_in = 2.25e4
    R_1a = 4.30e3
    R_1b_base = 4.30e3       # will be scaled by 5 before t<=5 s
    R_2a = 6.00e3
    R_2b = 6.00e3
    R_3a = 5.68e3
    R_3b = 5.68e3
    R_5c = 1.35e3
    R_5d = 1.35e3
    R_out = 5.74e3

    # ---- Capacitances -------------------------------------------------------
    C1 = 7.22e-7
    C2 = 7.53e-7
    C4 = 1.67e-5
    C5 = 1.07e-5

    # ---- variable resistance parameters  (CRA, CRV, venules) ---------------
    v_wall = 0.49  # Poisson's ratio

    # CRA
    mu_CRA_cP = 3.0
    mu_CRA = mu_CRA_cP * 1e-3 * 0.00750062
    E_CRA_MPa = 0.3
    E_CRA = E_CRA_MPa * 1e6 * 0.00750062
    d_CRA_ref = 175e-4
    r_CRA_ref = d_CRA_ref / 2.0
    A_ref_CRA = math.pi * r_CRA_ref ** 2
    h_CRA = 39.7239e-4
    L_1c = 0.2e-1
    L_1d = 1e-1
    K_l_CRA = 12.0 * A_ref_CRA / (math.pi * h_CRA ** 2)
    K_p_CRA = (1.0 / 12.0) * (E_CRA * h_CRA ** 3 / (1.0 - v_wall ** 2)) * (math.pi / A_ref_CRA) ** (3.0 / 2.0)
    k0_CRA_1c = (A_ref_CRA ** 2) / (8.0 * math.pi * mu_CRA * L_1c)
    k0_CRA_1d = (A_ref_CRA ** 2) / (8.0 * math.pi * mu_CRA * L_1d)

    # CRV
    mu_CRV_cP = 3.24
    mu_CRV = mu_CRV_cP * 1e-3 * 0.00750062
    E_CRV_MPa = 0.6
    E_CRV = E_CRV_MPa * 1e6 * 0.00750062
    d_CRV_ref = 238e-4
    r_CRV_ref = d_CRV_ref / 2.0
    A_ref_CRV = math.pi * r_CRV_ref ** 2
    h_CRV = 10.7e-4
    L_5a = 1e-1
    L_5b = 0.2e-1
    K_l_CRV = 12.0 * A_ref_CRV / (math.pi * h_CRV ** 2)
    K_p_CRV = (1.0 / 12.0) * (E_CRV * h_CRV ** 3 / (1.0 - v_wall ** 2)) * (math.pi / A_ref_CRV) ** (3.0 / 2.0)
    k0_CRV_5a = A_ref_CRV ** 2 / (8.0 * math.pi * mu_CRV * L_5a)
    k0_CRV_5b = A_ref_CRV ** 2 / (8.0 * math.pi * mu_CRV * L_5b)

    # venules
    mu_VEN_cP = 3.24
    mu_VEN = mu_VEN_cP * 1e-3 * 0.00750062
    E_VEN_MPa = 0.066
    E_VEN = E_VEN_MPa * 1e6 * 0.00750062
    d_VEN_ref = 0.015476685724901
    r_VEN_ref = d_VEN_ref / 2.0
    A_ref_VEN = math.pi * r_VEN_ref ** 2
    h_VEN = d_VEN_ref / 20.0
    L_VEN = 0.413520
    L_4a = L_VEN / 2.0
    L_4b = L_VEN / 2.0
    K_l_VEN = 12.0 * A_ref_VEN / (math.pi * h_VEN ** 2)
    K_p_VEN = (1.0 / 12.0) * (E_VEN * h_VEN ** 3 / (1.0 - v_wall ** 2)) * (math.pi / A_ref_VEN) ** (3.0 / 2.0)
    k0_VEN_4a = A_ref_VEN ** 2 / (8.0 * math.pi * mu_VEN * L_4a)
    k0_VEN_4b = A_ref_VEN ** 2 / (8.0 * math.pi * mu_VEN * L_4b)

    # ---------- helper: Pin waveform -----------------------------------------

    def pin_waveform(time):
        # Support the pin waveform step in this module.
        SP_loc, DP_loc, _, HR_loc = SP, DP, IOP, HR
        Thr = 60.0 / HR_loc
        Tm = time % Thr

        if Tm <= 0.082 * Thr:
            v_val = 0.65 * SP_loc - 0.475 * DP_loc * math.sin(
                (2.0 * math.pi * Tm / (4.0 * 0.082 * Thr))
                + (2.0 * math.pi * 0.082 * Thr / (0.328 * Thr))
            )
        elif (Tm > 0.082 * Thr) and (Tm <= 0.112 * Thr):
            v_val = 0.65 * SP_loc + 0.9 * math.sin(
                (2.0 * math.pi * Tm / (0.03 * Thr))
                - (2.0 * math.pi * 0.082 * Thr / (0.03 * Thr))
            )
        elif (Tm > 0.112 * Thr) and (Tm <= 0.398 * Thr):
            v_val = 0.65 * SP_loc + 0.118 * SP_loc * math.sin(
                (2.0 * math.pi * Tm / (0.572 * Thr))
                - (2.0 * math.pi * 0.112 * Thr / (0.572 * Thr))
            )
        elif (Tm > 0.398 * Thr) and (Tm <= 0.432 * Thr):
            v_val = (
                -(0.13 * SP_loc * Tm / (0.034 * Thr))
                + 0.65 * SP_loc
                + (0.13 * SP_loc * 0.398 * Thr / (0.034 * Thr))
            )
        elif (Tm > 0.432 * Thr) and (Tm <= 0.482 * Thr):
            v_val = (
                0.52 * SP_loc
                - 0.8 * math.sin(
                    (2.0 * math.pi * Tm / (0.05 * Thr))
                    + (2.0 * math.pi * 0.332 * Thr / (0.05 * Thr))
                )
            )
        else:
            v_val = 0.52 * SP_loc + (0.52 * SP_loc - 0.5 * DP_loc) * math.sin(
                (2.0 * math.pi * Tm / (2.072 * Thr))
                + (2.0 * math.pi * 0.554 * Thr / (2.072 * Thr))
            )
        return v_val

    # ---------- helper: variable resistances ---------------------------------

    def compute_resistances(
        deltaP1c=None,
        deltaP1d=None,
        deltaP4a=None,
        deltaP4b=None,
        deltaP5a=None,
        deltaP5b=None,
    ):
        # Compute resistances.
        R_1c = R_1d = R_4a = R_4b = R_5a = R_5b = 0.0

        if deltaP1c is not None:
            R_1c = (1.0 / k0_CRA_1c) * (1.0 + deltaP1c / (K_p_CRA * K_l_CRA)) ** (-4.0)
        if deltaP1d is not None:
            R_1d = (1.0 / k0_CRA_1d) * (1.0 + deltaP1d / (K_p_CRA * K_l_CRA)) ** (-4.0)

        if deltaP4a is not None:
            R_4a = (1.0 / k0_VEN_4a) * (1.0 + deltaP4a / (K_p_VEN * K_l_VEN)) ** (-4.0)
        if deltaP4b is not None:
            R_4b = (1.0 / k0_VEN_4b) * (1.0 + deltaP4b / (K_p_VEN * K_l_VEN)) ** (-4.0)

        if deltaP5a is not None:
            R_5a = (1.0 / k0_CRV_5a) * (1.0 + deltaP5a / (K_p_CRV * K_l_CRV)) ** (-4.0)
        if deltaP5b is not None:
            R_5b = (1.0 / k0_CRV_5b) * (1.0 + deltaP5b / (K_p_CRV * K_l_CRV)) ** (-4.0)

        return R_1c, R_1d, R_4a, R_4b, R_5a, R_5b

    # ---------- dp equations (scalar, numpy) ---------------------------------

    def dp1(Pin, P1, P2, R_1b_local):
        # Support the dp1 step in this module.
        c_2 = 10.0
        deltaP1c = P1 - c_2
        deltaP1d = P1 - c_2

        R_1c, R_1d, _, _, _, _ = compute_resistances(
            deltaP1c=deltaP1c,
            deltaP1d=deltaP1d,
        )

        num_in = (Pin - P1) / (R_in + R_1a)
        num_mid = (P1 - P2) / (R_1b_local + R_1c + R_1d + R_2a)
        F1 = (num_in - num_mid) / C1
        return F1

    def dp2(P1, P2, P4, R_1b_local):
        # Support the dp2 step in this module.
        c_2 = 10.0
        deltaP1c = P1 - c_2
        deltaP1d = P1 - c_2
        deltaP4a = P4 - c_2
        deltaP4b = P4 - c_2

        R_1c, R_1d, R_4a, _, _, _ = compute_resistances(
            deltaP1c=deltaP1c,
            deltaP1d=deltaP1d,
            deltaP4a=deltaP4a,
            deltaP4b=deltaP4b,
        )

        num_in = (P1 - P2) / (R_1b_local + R_1c + R_1d + R_2a)
        num_mid = (P2 - P4) / (R_2b + R_3a + R_3b + R_4a)
        F2 = (num_in - num_mid) / C2
        return F2

    def dp4(P2, P4, P5):
        # Support the dp4 step in this module.
        c_2 = 10.0
        deltaP4a = P4 - c_2
        deltaP4b = P4 - c_2
        deltaP5a = P5 - c_2
        deltaP5b = P5 - c_2

        _, _, R_4a, R_4b, R_5a, R_5b = compute_resistances(
            deltaP4a=deltaP4a,
            deltaP4b=deltaP4b,
            deltaP5a=deltaP5a,
            deltaP5b=deltaP5b,
        )

        num_in = (P2 - P4) / (R_2b + R_3a + R_3b + R_4a)
        num_mid = (P4 - P5) / (R_4b + R_5a + R_5b + R_5c)
        F3 = (num_in - num_mid) / C4
        return F3

    def dp5(P4, P5):
        # Support the dp5 step in this module.
        c_2 = 10.0
        deltaP4a = P4 - c_2
        deltaP4b = P4 - c_2
        deltaP5a = P5 - c_2
        deltaP5b = P5 - c_2

        _, _, _, R_4b, R_5a, R_5b = compute_resistances(
            deltaP4a=deltaP4a,
            deltaP4b=deltaP4b,
            deltaP5a=deltaP5a,
            deltaP5b=deltaP5b,
        )

        num_in = (P4 - P5) / (R_4b + R_5a + R_5b + R_5c)
        num_out = (P5 - P_out) / (R_5d + R_out)
        F4 = (num_in - num_out) / C5
        return F4

    # ---------- main RHS for solve_ivp ---------------------------------------

    def rhs_retina(t, P_vec):
        # states:
        # Support the rhs retina step in this module.
        P1, P2, P4, P5 = P_vec[0], P_vec[1], P_vec[2], P_vec[3]

        Pin = pin_waveform(t)

        # time-dependent R_1b:
        if t > 5.0:
            R_1b = R_1b_base
        else:
            R_1b = 5.0 * R_1b_base

        F1 = dp1(Pin, P1, P2, R_1b)
        F2 = dp2(P1, P2, P4, R_1b)
        F3 = dp4(P2, P4, P5)
        F4 = dp5(P4, P5)

        # Same state order as rhs_true_with_drive: derivatives are for [P1, P2, P4, P5].
        return np.array([F1, F2, F3, F4], dtype=np.float64)

    # ---------- integrate -----------------------------------------------------

    t_span = (float(t0), float(T))
    kwargs = dict(method="RK45", rtol=1e-6, atol=1e-9)
    if max_step_scipy > 0.0:
        kwargs["max_step"] = max_step_scipy
    if uniform_dt > 0.0:
        N = int((T - t0) / uniform_dt + 0.5) + 1
        kwargs["t_eval"] = np.linspace(t0, T, N, endpoint=True)

    sol = solve_ivp(rhs_retina, t_span, y0_np, **kwargs)
    t_np = sol.t
    P_np = sol.y.T  # (T, 4) [P1,P2,P4,P5]

    # compute Pin(t) on the same grid
    Pin_seq_np = np.array([pin_waveform(tt) for tt in t_np], dtype=np.float32)

    if not return_torch:
        return t_np.astype(np.float32), P_np.astype(np.float32), Pin_seq_np

    if device is None:
        device = torch.device("cpu")

    t_t = torch.tensor(t_np.astype(np.float32), device=device)
    P_t = torch.tensor(P_np.astype(np.float32), device=device).unsqueeze(1)  # (T,1,4)
    Pin_t = torch.tensor(Pin_seq_np, device=device)  # (T,)

    return t_t, P_t, Pin_t

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    # Accept (T,4) or (T,1,4); return (T,4)
    # Support the ensure 2d step in this module.
    if x.dim() == 3 and x.size(1) == 1:
        return x[:, 0, :]
    return x

def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    # Accept (T,) or (T,1); return (T,1)
    # Support the ensure col step in this module.
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x

def _try_integrate_retina_adaptive(integrate_retina_adaptive, y0_np, T, max_step_scipy, uniform_dt, device):
    """
    Call integrate_retina_adaptive with either of the two common signatures:
      A) integrate_retina_adaptive(y0_np, t0, T, max_step_scipy, uniform_dt, return_torch=True, device=device)
      B) integrate_retina_adaptive(y0_np=y0_np, t0=0.0, T=T, max_step_scipy=..., uniform_dt=..., return_torch=True, device=device)
    """
    try:
        return integrate_retina_adaptive(
            y0_np=y0_np,
            t0=0.0,
            T=T,
            max_step_scipy=max_step_scipy,
            uniform_dt=uniform_dt,
            return_torch=True,
            device=device,
        )
    except TypeError:
        # Fallback to positional signature
        return integrate_retina_adaptive(
            y0_np,            # y0_np
            0.0,              # t0
            T,                # T
            max_step_scipy,   # max_step_scipy
            uniform_dt,       # uniform_dt
            True,             # return_torch
            device            # device
        )

def make_data_retina(exp, device):
    """
    Returns: t, P_meas, P_true, Pin_seq
      t:       (T,)
      P_true:  (T, 4) with columns [P1, P2, P4, P5]
      Pin_seq: (T, 1)
      P_meas:  (T, 3) using exp.meas_idx (typically [0,1,3] = P1,P2,P5)

    Packages the retinal pressure trajectory, drive signal, and noisy measurements.
    """
    sys_cfg = exp.system_cfg
    tc = exp.train_cfg
    dc = exp.data_gen if hasattr(exp, "data_gen") else exp.data_cfg  # tolerate either field name

    # Initial state: use p*_true for data generation to ensure consistency
    y0_np = np.array(
        [sys_cfg.p1_true, sys_cfg.p2_true, sys_cfg.p4_true, sys_cfg.p5_true],
        dtype=np.float64
    )

    # Integrate retina + Pin sequence (exactly 3 outputs)
    t, P_true, Pin_seq = _try_integrate_retina_adaptive(
        integrate_retina_adaptive,
        y0_np=y0_np,
        T=tc.T_train,
        max_step_scipy=dc.max_step_scipy,
        uniform_dt=dc.uniform_dt,
        device=device,
    )

    # Normalize shapes
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
    if not torch.is_tensor(P_true):
        P_true = torch.as_tensor(P_true, dtype=torch.float32, device=device)
    if not torch.is_tensor(Pin_seq):
        Pin_seq = torch.as_tensor(Pin_seq, dtype=torch.float32, device=device)

    t = t.to(device=device, dtype=torch.float32).contiguous()               # (T,)
    P_true = _ensure_2d(P_true.to(device=device, dtype=torch.float32))      # (T,4)
    Pin_seq = _ensure_col(Pin_seq.to(device=device, dtype=torch.float32))   # (T,1)

    # Build measurements for measured indices (e.g., [0,1,3] = P1,P2,P5)
    meas_idx = getattr(exp, "meas_idx", [0, 1, 3])
    P_meas = P_true[:, meas_idx].clone()                                    # (T,3)

    # Per-channel proportional noise: noise_std = meas_noise_std * |channel_rms|
    torch.manual_seed(dc.seed)
    noise_std = tc.meas_noise_std
    if noise_std and noise_std > 0.0:
        ch_scales = P_meas.abs().mean(dim=0).clamp(min=1e-6)  # (n_meas,)
        P_meas = P_meas + noise_std * ch_scales.unsqueeze(0) * torch.randn_like(P_meas)

    # Return exactly four items in the order expected by the training pipeline.
    return t, P_meas, P_true, Pin_seq

class LearnedODERetina4WithPin(torch.jit.ScriptModule):
    """
    Retina learned ODE:
      y = [P1, P2, P4, P5]   (ordering must match RetinaPhysics)
      teacher: physics.rhs_true_with_drive(y, pin)  → (…,4)
      Derivatives at `replaced_indices` are replaced with NN([y, pin]).

    APIs:
      - rk4_step_with_pin(y, dt, pin): main integrator (Pin required)
      - rk4_step_with_drive(...): alias for older code
      - rk4_step(y, dt): Pin=0 fallback (for plotting utilities without Pin)

    NOTE: replaced_index=1 learns P2 dynamics (arteriole pressure)
          replaced_index=2 learns P4 dynamics (capillary/venule pressure) - DEFAULT

    This wrapper exists because retina rollouts need the external Pin drive
    concatenated with the state before the NN predicts learned derivatives.
    """
    __constants__ = ['replaced_index', 'replaced_indices']

    def __init__(self, physics: torch.jit.ScriptModule, latent_model: nn.Module,
                 max_dt: float = 0.02, min_dt: float = 0.0, replaced_index: int = 2,
                 replaced_indices=None):
        # Initialize this object and store the configuration it needs later.
        super().__init__()
        self.physics = physics
        self.latent_model = latent_model
        if replaced_indices is None:
            replaced_indices = (int(replaced_index),)
        self.replaced_indices = tuple(int(i) for i in replaced_indices)
        self.replaced_index: int = int(self.replaced_indices[0])  # backward-compatible scalar
        self.register_buffer("max_dt", torch.tensor(max_dt))
        self.register_buffer("min_dt", torch.tensor(min_dt))

    # ---- helpers -------------------------------------------------------------

    @torch.jit.script_method
    def _expand_pin(self, pin: Tensor, y: Tensor) -> Tensor:
        """
        Ensure pin has shape matching y's batch dimension.
        Accept pin shapes: (), (1,), (B,), (B,1), (...,1).
        Always returns shape (B, 1) where B matches y.shape[0].
        """
        batch_size = y.shape[0]

        if pin.dim() == 0:
            # scalar: broadcast to (B, 1)
            return torch.ones((batch_size, 1), dtype=pin.dtype, device=pin.device) * pin

        if pin.dim() == 1:
            # (1,) or (B,) → (B, 1)
            if pin.shape[0] == 1:
                # Single scalar value: broadcast to all batch elements
                return torch.ones((batch_size, 1), dtype=pin.dtype, device=pin.device) * pin[0]
            else:
                # Already has batch dimension
                return pin.unsqueeze(-1)

        # Already (B, 1) or (..., 1)
        return pin

    # ---- dynamics ------------------------------------------------------------

    @torch.jit.script_method
    def f_full_with_pin(self, state: Tensor, pin_drive: Tensor) -> Tensor:
        # Support the f full with pin step in this module.
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        pin_b = self._expand_pin(pin_drive, state)                 # (...,1)

        # The retinal circuit physics still computes every pressure derivative first.
        rhs_phys = self.physics.rhs_true_with_drive(state, pin_b)  # (...,4)
        out = rhs_phys.clone()

        # The NN gets both pressure states and the current Pin drive, then predicts learned derivatives.
        state_and_drive_features = torch.cat([state, pin_b], dim=-1)  # (...,5)
        ds_rep = self.latent_model(state_and_drive_features)          # (...,n_replaced)

        for col in range(len(self.replaced_indices)):
            idx = self.replaced_indices[col]
            # Replace only the configured pressure derivative; all other nodes remain analytical.
            out[..., idx:idx+1] = ds_rep[..., col:col+1]
        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def rhs_true_with_drive(self, y: Tensor, pin: Tensor) -> Tensor:
        """
        CKF interface: hybrid dynamics with drive signal.
        This allows CKF to propagate with the correct Pin value.

        Args:
            y: States (..., 4) = [P1, P2, P4, P5]
            pin: Drive signal (..., 1) or scalar

        Returns:
            Derivatives (..., 4) using hybrid physics + NN
        """
        return self.f_full_with_pin(y, pin)

    @torch.jit.script_method
    def _rk4_single_with_pin(self, y: Tensor, h: float, pin: Tensor) -> Tensor:
        # Support the rk4 single with pin step in this module.
        hh = torch.tensor(h, dtype=y.dtype, device=y.device)
        k1 = self.f_full_with_pin(y,                pin)
        k2 = self.f_full_with_pin(y + 0.5 * hh * k1, pin)
        k3 = self.f_full_with_pin(y + 0.5 * hh * k2, pin)
        k4 = self.f_full_with_pin(y + hh * k3,       pin)
        y_next = y + (hh / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        y_next = self.physics.clip_states(y_next)
        return y_next

    # ---- public steppers -----------------------------------------------------

    @torch.jit.script_method
    def rk4_step_with_pin(self, y: Tensor, dt: float, pin: Tensor) -> Tensor:
        # Support the rk4 step with pin step in this module.
        single = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single = True

        total_dt_f = float(dt)
        if total_dt_f == 0.0:
            return y.squeeze(0) if single else y

        sign = 1.0
        if total_dt_f < 0.0:
            sign = -1.0
            total_dt_f = -total_dt_f

        max_dt_f = float(self.max_dt.item())
        min_dt_f = float(self.min_dt.item())

        if min_dt_f > 0.0 and total_dt_f < min_dt_f:
            total_dt_f = min_dt_f

        n_sub = 1
        if max_dt_f > 0.0 and total_dt_f > max_dt_f:
            n_sub = int(total_dt_f / max_dt_f + 0.999999)
        if n_sub < 1:
            n_sub = 1

        h = sign * total_dt_f / float(n_sub)

        out = y
        for _ in range(n_sub):
            out = self._rk4_single_with_pin(out, h, pin)
        return out.squeeze(0) if single else out

    @torch.jit.script_method
    def rk4_step_with_drive(self, y: Tensor, dt: float, pin: Tensor) -> Tensor:
        # alias for older loss code
        # Support the rk4 step with drive step in this module.
        return self.rk4_step_with_pin(y, dt, pin)

    @torch.jit.script_method
    def rk4_step(self, y: Tensor, dt: float) -> Tensor:
        # Pin=0 default (for utilities that don't pass Pin)
        # Support the rk4 step step in this module.
        if y.dim() == 1:
            pin0 = torch.tensor(0.0, dtype=y.dtype, device=y.device)
        else:
            pin0 = torch.zeros_like(y[..., :1])
        return self.rk4_step_with_pin(y, dt, pin0)

    @torch.jit.script_method
    def propagate_batch(self, X: Tensor, dt: float) -> Tensor:
        """CKF interface: propagate batch of sigma points (Pin=0 fallback)"""
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

def make_experiment() -> ExperimentConfig:
    # Central preset for the default retina model with P4 learned and Pin as a drive.
    return ExperimentConfig(system_name='retina', system_cfg=RetinaSystemConfig(), integrator_cfg=IntegratorConfig(default_dt=0.0025, max_dt=0.005, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(0.05, 0.05, 0.6, 0.05), R_diag=(0.0006, 0.0006, 0.0006), P0_diag=(0.5, 0.5, 8.0, 0.5)), nn_cfg=NNConfig(in_dim=5, out_dim=1, hidden_dims=(96, 96), activation='tanh', input_scale=(0.03, 0.04, 0.06, 0.06, 0.015), lr=0.0005, weight_decay=1e-06, grad_clip=1.0), train_cfg=TrainConfig(T_train=4.0, meas_noise_std=0.0025, meas_snr=400.0, n_epochs=120, min_rollout_horizon=1, max_rollout_horizon=50, stride=1, use_pseudo_latent=True, pseudo_weight=4.0, full_state_weight=0.02), plot_cfg=PlotConfig(plot_stride=30, show_every=5, max_rollouts=15), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.0, seed=1234), state_labels=('P1', 'P2', 'P4', 'P5'), title_prefix='Retina', meas_idx=(0, 1, 3), pseudo_idx=(2,), uses_pin=True)

def make_retina2_experiment() -> ExperimentConfig:
    # Variant preset that learns P2 and P4 while measuring only P1 and P5.
    return ExperimentConfig(system_name='retina', system_cfg=RetinaSystemConfig(replaced_index=1, meas_state_indices=(0, 3)), integrator_cfg=IntegratorConfig(default_dt=0.0025, max_dt=0.005, min_dt=0.0, n_substeps=1), ckf_cfg=CKFConfig(Q_diag=(0.05, 0.6, 0.6, 0.05), R_diag=(0.0006, 0.0006), P0_diag=(0.5, 8.0, 8.0, 0.5)), nn_cfg=NNConfig(in_dim=5, out_dim=2, hidden_dims=(96, 96), activation='tanh', input_scale=(0.03, 0.04, 0.06, 0.06, 0.015), lr=0.0005, weight_decay=1e-06, grad_clip=1.0), train_cfg=TrainConfig(T_train=4.0, meas_noise_std=0.0025, meas_snr=400.0, n_epochs=120, min_rollout_horizon=1, max_rollout_horizon=50, stride=1, use_pseudo_latent=True, pseudo_weight=4.0, full_state_weight=0.02), plot_cfg=PlotConfig(plot_stride=30, show_every=5, max_rollouts=15), data_cfg=DataGenConfig(max_step_scipy=0.0, uniform_dt=0.0, seed=1234), state_labels=('P1', 'P2', 'P4', 'P5'), title_prefix='Retina2', meas_idx=(0, 3), pseudo_idx=(1, 2), uses_pin=True)

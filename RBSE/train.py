# train.py
# RBSE training: joint state + parameter estimation.
#
# This implements the algorithm from:
#   "Cubature Kalman Filter Based Training of Hybrid Differential Equation
#    Recurrent Neural Network Physiological Dynamic Models"
#
# Key difference from the main codebase:
#   - No RTS smoother
#   - No rollout loss
#   - No gradient-based optimization (no backprop)
#   - NN weights are part of the CKF augmented state
#   - CKF simultaneously estimates states AND NN parameters
#
# Usage:
#   python -m RBSE --models ho
#   python -m RBSE --models ho retina --epochs 1

import os
import sys
import time
import argparse
import datetime as _dt
import random as _random
import string as _string

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ExperimentConfig,
    get_experiment,
    make_data_ho, make_data_pendulum, make_data_hh,
    make_data_brusselator, make_data_yeast, make_data_retina,
    HOPhysics, PendulumPhysics, HodgkinHuxleyPhysics,
    BrusselatorPhysics, YeastPhysics, RetinaPhysics,
)

from RBSE.augmented_system import AugmentedSystem, FunctionalMLP, build_augmented_system
from RBSE.ckf_augmented import ckf_augmented
from RBSE.configuration import VALID_MODELS, get_model_config
from RBSE.visualize import plot_all


# Default: 2 hidden layers with 8 units each.
# This keeps the augmented state manageable (NN weights are CKF states, not backprop params).
# Rough parameter counts (out_dim=1):
#   HO (in=2):     2*8+8 + 8*8+8 + 8*1+1  = 105 weights
#   HH (in=4):     4*8+8 + 8*8+8 + 8*1+1  = 121 weights
#   Retina (in=5): 5*8+8 + 8*8+8 + 8*1+1  = 129 weights
# RBSEModelConfig in RBSE/configuration.py owns these model defaults.

device = torch.device("cpu")


def _make_run_out_dir(results_root, model_name, run_id=None):
    """
    Create a unique output directory for a single run.

    Layout:
      <results_root>/<model_name>/<YYYYMMDD_HHMMSS>_<abc>[_run<id>]/
    """
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(_random.choices(_string.ascii_lowercase, k=3))
    run_tag = f"_run{run_id}" if run_id is not None else ""
    out_dir = os.path.join(results_root, model_name, f"{ts}_{rand}{run_tag}")
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


class _TeeTextIO:
    """Write-through wrapper that duplicates writes to multiple streams."""

    def __init__(self, *streams):
        # Initialize this object and store the configuration it needs later.
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        # Support the write step in this module.
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        # Support the flush step in this module.
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Preserve behavior if any code checks whether stdout is a TTY.
        # Support the isatty step in this module.
        for s in self._streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                pass
        return False


def _hh_gate_steady_state(V: float):
    """Approximate HH gate steady states from voltage for better CKF startup."""
    x_m = (V + 40.0) / 10.0
    if abs(x_m) > 1e-7:
        x_m_clip = np.clip(x_m, -50.0, 50.0)
        denom_m = -np.expm1(-x_m_clip)
        alpha_m = x_m_clip / denom_m
    else:
        alpha_m = 1.0
    beta_m = 4.0 * np.exp(np.clip(-(V + 65.0) / 18.0, -50.0, 50.0))

    alpha_h = 0.07 * np.exp(np.clip(-(V + 65.0) / 20.0, -50.0, 50.0))
    beta_h = 1.0 / (1.0 + np.exp(np.clip(-(V + 35.0) / 10.0, -50.0, 50.0)))

    x_n = (V + 55.0) / 10.0
    if abs(x_n) > 1e-7:
        x_n_clip = np.clip(x_n, -50.0, 50.0)
        denom_n = -np.expm1(-x_n_clip)
        alpha_n = 0.1 * x_n_clip / denom_n
    else:
        alpha_n = 0.1
    beta_n = 0.125 * np.exp(np.clip(-(V + 65.0) / 80.0, -50.0, 50.0))

    m_inf = alpha_m / max(alpha_m + beta_m, 1e-8)
    h_inf = alpha_h / max(alpha_h + beta_h, 1e-8)
    n_inf = alpha_n / max(alpha_n + beta_n, 1e-8)
    return float(m_inf), float(h_inf), float(n_inf)


def setup_system(exp, device):
    """
    Set up the augmented system for RBSE training.

    Returns
    -------
    aug_sys : AugmentedSystem
    w0 : (n_weights,) initial NN weights
    t : (T,) timestamps
    y_meas : (T, m_dim) measurements
    y_true : (T, n_states) true states (for evaluation)
    state_labels : list of str
    meas_idx : list of int
    prior_vec : list of float
    drive_seq : tensor or None
    replaced_idx : int
    """
    ic = exp.integrator_cfg
    ck = exp.ckf_cfg
    name = exp.system_name

    if name == "ho":
        t, x_meas, x_true, v_true = make_data_ho(exp, device)
        physics = HOPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["x", "v"]
        meas_idx = [0]
        replaced_idx = 1
        prior_vec = [exp.system_cfg.x0_prior, exp.system_cfg.v0_prior]
        y_true = torch.stack([x_true, v_true], dim=-1)
        y_meas = x_meas.unsqueeze(-1)  # (T, 1)
        drive_seq = None

    elif name == "pendulum":
        t, th_meas, th_true, om_true = make_data_pendulum(exp, device)
        physics = PendulumPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["theta", "omega"]
        meas_idx = [0]
        replaced_idx = 1
        prior_vec = [exp.system_cfg.theta0_prior, exp.system_cfg.omega0_prior]
        y_true = torch.stack([th_true, om_true], dim=-1)
        y_meas = th_meas.unsqueeze(-1)
        drive_seq = None

    elif name == "hh":
        t, meas, V_true, m_true, h_true, n_true = make_data_hh(exp, device)
        physics = HodgkinHuxleyPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["V", "m", "h", "n"]
        meas_idx = [0, 2, 3]  # V, h, n (not m)
        replaced_idx = 1  # m is learned
        prior_vec = [
            exp.system_cfg.V0_prior, exp.system_cfg.m0_prior,
            exp.system_cfg.h0_prior, exp.system_cfg.n0_prior,
        ]
        y_true = torch.stack([V_true, m_true, h_true, n_true], dim=-1)
        y_meas = meas
        drive_seq = None

    elif name == "brusselator":
        t, meas, x_true, y_true_b, z_true = make_data_brusselator(exp, device)
        physics = BrusselatorPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["x", "y", "z"]
        meas_idx = [1, 2]  # y, z (not x)
        replaced_idx = 0  # x is learned
        prior_vec = [
            exp.system_cfg.x0_prior, exp.system_cfg.y0_prior,
            exp.system_cfg.z0_prior,
        ]
        y_true = torch.stack([x_true, y_true_b, z_true], dim=-1)
        y_meas = meas
        drive_seq = None

    elif name == "yeast":
        result = make_data_yeast(exp, device)
        t = result[0]
        meas = result[1]
        x_trues = result[2:9]
        physics = YeastPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
        meas_idx = [0, 1, 2, 4, 5, 6]  # x4 unmeasured
        replaced_idx = 3  # x4 is learned
        prior_vec = list(exp.system_cfg.x0_prior)
        y_true = torch.stack(list(x_trues), dim=-1)
        y_meas = meas
        drive_seq = None

    elif name == "retina":
        t, P_meas, P_true, Pin_seq = make_data_retina(exp, device)
        physics = RetinaPhysics(exp.system_cfg, ic, ck).to(device)
        state_labels = ["P1", "P2", "P4", "P5"]
        meas_idx = [0, 1, 3]  # P1, P2, P5 (not P4)
        replaced_idx = 2  # P4 is learned
        prior_vec = [
            exp.system_cfg.p1_prior, exp.system_cfg.p2_prior,
            exp.system_cfg.p4_prior, exp.system_cfg.p5_prior,
        ]
        y_true = P_true
        y_meas = P_meas
        drive_seq = Pin_seq

    else:
        raise ValueError(f"Unknown system: {name}")

    n_states = len(state_labels)
    has_drive = drive_seq is not None

    input_mask = None

    # Use smaller hidden dims for RBSE (augmented state must be manageable)
    # The paper uses small NNs since weights are part of the state vector
    nn_cfg = exp.nn_cfg

    # HH is stiff around spikes; use configured max_dt (0.0025 for HH) instead
    # of the older coarse 0.01 local override.
    local_max_dt = ic.max_dt
    residual_on_physics = (name in {"hh", "retina"})

    # Build augmented system
    aug_sys, w0 = build_augmented_system(
        physics=physics,
        nn_config=nn_cfg,
        meas_idx=meas_idx,
        replaced_idx=replaced_idx,
        n_states=n_states,
        max_dt=local_max_dt,
        input_mask=input_mask,
        has_drive=has_drive,
        residual_on_physics=residual_on_physics,
        device=device,
        dtype=torch.float32,
    )

    return (aug_sys, w0, t, y_meas, y_true, state_labels,
            meas_idx, prior_vec, drive_seq, replaced_idx, physics)


def build_noise_matrices(exp, aug_sys, q_weight_scale=1e-6,
                         q_state_scale=1.0, r_scale=1.0):
    """
    Build augmented process noise Q_aug and measurement noise R.

    Q_aug = diag(q_w * I_p, Q_states)
    where q_w controls how fast the CKF can update NN weights.

    Parameters
    ----------
    exp : ExperimentConfig
    aug_sys : AugmentedSystem
    q_weight_scale : float
        Process noise for NN weights. Larger = faster adaptation.

    Returns
    -------
    Q_aug : (n_aug, n_aug)
    R : (m_dim, m_dim)
    """
    n_w = aug_sys.n_weights
    n_s = aug_sys.n_states
    n_aug = n_w + n_s

    ck = exp.ckf_cfg

    # Weight process noise: small but nonzero to allow CKF updates
    Q_w = q_weight_scale * torch.eye(n_w, device=device)

    # State process noise: from config
    Q_s = torch.diag(torch.tensor(list(ck.Q_diag), dtype=torch.float32,
                                  device=device)) * q_state_scale

    # Assemble block-diagonal
    Q_aug = torch.zeros(n_aug, n_aug, device=device)
    Q_aug[:n_w, :n_w] = Q_w
    Q_aug[n_w:, n_w:] = Q_s

    # Measurement noise
    R = torch.diag(torch.tensor(list(ck.R_diag), dtype=torch.float32,
                                device=device)) * r_scale

    return Q_aug, R


def build_initial_covariance(exp, aug_sys, p_weight_scale=1e-2):
    """
    Build initial augmented state covariance P0_aug.

    P0_aug = diag(p_w * I_p, P0_states)

    Parameters
    ----------
    exp : ExperimentConfig
    aug_sys : AugmentedSystem
    p_weight_scale : float
        Initial uncertainty for NN weights. Larger = more initial freedom.

    Returns
    -------
    P0_aug : (n_aug, n_aug)
    """
    n_w = aug_sys.n_weights
    n_s = aug_sys.n_states
    n_aug = n_w + n_s

    ck = exp.ckf_cfg

    P_w = p_weight_scale * torch.eye(n_w, device=device)
    P_s = torch.diag(torch.tensor(list(ck.P0_diag), dtype=torch.float32, device=device))

    P0 = torch.zeros(n_aug, n_aug, device=device)
    P0[:n_w, :n_w] = P_w
    P0[n_w:, n_w:] = P_s

    return P0


def rollout_from_weights(aug_sys, w_final, y_true, t, drive_seq=None):
    """
    Integrate the hybrid ODE forward from the true initial condition
    using the final CKF-estimated NN weights.

    This is the standalone test: no measurement corrections, just the
    learned model producing a trajectory on its own.

    Parameters
    ----------
    aug_sys : AugmentedSystem
    w_final : (n_weights,) tensor
    y_true : (T, n_states) tensor â€” ground truth for initial condition
    t : (T,) tensor â€” time array
    drive_seq : (T,) or (T,d) tensor or None

    Returns
    -------
    y_rollout : (T, n_states) ndarray
    """
    n_states = aug_sys.n_states
    T = t.shape[0]
    dt_list = (t[1:] - t[:-1]).tolist()

    y_rollout = torch.zeros(T, n_states, dtype=torch.float32, device=w_final.device)
    s = y_true[0].clone()  # start from true initial condition
    y_rollout[0] = s

    with torch.no_grad():
        for k in range(1, T):
            dt = dt_list[k - 1]
            drive = None
            if drive_seq is not None:
                drive = drive_seq[k - 1]
                if drive.dim() == 0:
                    drive = drive.unsqueeze(0)

            # Sub-step if needed
            n_sub = max(1, int(abs(dt) / aug_sys.max_dt + 0.999))
            dt_sub = dt / n_sub
            for _ in range(n_sub):
                s = aug_sys._rk4_step(s, w_final, dt_sub, drive)
            s = aug_sys.clip_states(s)
            y_rollout[k] = s

    return y_rollout.detach().numpy()


def run_rbse(
    model_name,
    n_epochs=1,
    q_weight_scale=1e-6,
    p_weight_scale=1e-2,
    q_state_scale=1.0,
    r_scale=1.0,
    hidden_dims=None,
    run_id=None,
    save_dir=None,
):
    """
    Run RBSE training for a given system.

    In addition to saving NPZ/PNG artifacts, this function also writes a copy of
    the console output into the run directory.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "results")

    out_dir = _make_run_out_dir(save_dir, model_name, run_id=run_id)

    # Save console output in the run folder.
    # Preserve the historical misspelled filename and add safe aliases.
    console_path = os.path.join(out_dir, "consolse_output.txt")
    console_path_alt = os.path.join(out_dir, "console_output.txt")
    console_path_space = os.path.join(out_dir, "consolse output.txt")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f_console = open(console_path, "w", encoding="utf-8")
    sys.stdout = _TeeTextIO(orig_stdout, f_console)
    sys.stderr = _TeeTextIO(orig_stderr, f_console)

    # Model-owned defaults apply only when this function is not being used by
    # an explicit hyperparameter sweep (for example RBSE/submit_grid.sh).
    model_config = get_model_config(model_name)
    if model_name == "hh" and q_weight_scale == 1e-6 and p_weight_scale == 1e-2 \
            and q_state_scale == 1.0 and r_scale == 1.0:
        q_weight_scale = model_config.q_weight_scale
        p_weight_scale = model_config.p_weight_scale
        q_state_scale = model_config.q_state_scale
        r_scale = model_config.r_scale
        n_epochs = max(n_epochs, model_config.minimum_epochs)
    elif model_name == "retina" and q_weight_scale == 1e-6 and p_weight_scale == 1e-2 \
            and q_state_scale == 1.0 and r_scale == 1.0:
        q_weight_scale = model_config.q_weight_scale
        p_weight_scale = model_config.p_weight_scale
        q_state_scale = model_config.q_state_scale
        r_scale = model_config.r_scale
        n_epochs = max(n_epochs, model_config.minimum_epochs)

    try:
        print(f"\n{'='*60}")
        print(f"  RBSE Training: {model_name}")
        print(f"  Epochs: {n_epochs}")
        print(f"  q_weight: {q_weight_scale}, p_weight: {p_weight_scale}")
        print(f"  q_state_scale: {q_state_scale}, r_scale: {r_scale}")
        print(f"  hidden_dims: {hidden_dims or model_config.hidden_dims}")
        print(f"  Output dir: {out_dir}")
        print(f"{'='*60}\n")

        # Get experiment config
        exp = get_experiment(model_name)
        # Force shallow NN — weights live inside the CKF state vector,
        # so fewer parameters = smaller augmented state = fewer sigma points.
        if hidden_dims is None:
            hidden_dims = model_config.hidden_dims
        exp.nn_cfg.hidden_dims = hidden_dims

        # Setup system
        (aug_sys, w0, t, y_meas, y_true, state_labels,
         meas_idx, prior_vec, drive_seq, replaced_idx, physics) = setup_system(exp, device)
        if model_name == "hh":
            # HH measurements are direct observations of V, h, n; anchor them strongly.
            aug_sys.observed_state_blend = 1.0
            # Random functional weights make HH startup brittle; begin from a neutral mean
            # and let the CKF move the weights through its covariance.
            w0 = torch.zeros_like(w0)
        elif model_name == "retina":
            # P1, P2, and P5 are directly measured; keep them tightly aligned to data.
            aug_sys.observed_state_blend = 1.0
            # Start from the mechanistic retina model and let the learned term act only
            # as a small residual correction on the hidden capillary pressure dynamics.
            w0 = torch.zeros_like(w0)

        n_w = aug_sys.n_weights
        n_s = aug_sys.n_states
        n_aug = n_w + n_s
        T = t.shape[0]

        print(f"  States: {n_s} ({state_labels})")
        print(f"  Measured: {len(meas_idx)} (indices {meas_idx})")
        print(f"  Replaced: {replaced_idx} ({state_labels[replaced_idx]})")
        print(f"  NN weights: {n_w}")
        print(f"  Augmented dim: {n_aug}")
        print(f"  Sigma points: {2 * n_aug}")
        print(f"  Time steps: {T}")
        print(f"  Data shape: meas={y_meas.shape}, true={y_true.shape}")
        print()

        # Build noise matrices
        Q_aug, R = build_noise_matrices(
            exp,
            aug_sys,
            q_weight_scale=q_weight_scale,
            q_state_scale=q_state_scale,
            r_scale=r_scale,
        )
        P0_aug = build_initial_covariance(exp, aug_sys, p_weight_scale)

        # Initial augmented state
        s0 = torch.tensor(prior_vec, dtype=torch.float32, device=device)
        # Anchor measured states to the first observation to reduce start-up bias.
        if y_meas.numel() > 0 and len(meas_idx) > 0:
            for col, state_i in enumerate(meas_idx):
                s0[state_i] = y_meas[0, col]
        if model_name == "hh":
            V0_meas = float(s0[0].item())
            m_inf, h_inf, n_inf = _hh_gate_steady_state(V0_meas)
            s0[1] = m_inf
            # If the first h/n observations are noisy or missing, fall back to HH steady state.
            if len(meas_idx) < 2:
                s0[2] = h_inf
            if len(meas_idx) < 3:
                s0[3] = n_inf
        m0_aug = aug_sys.get_initial_augmented_state(w0, s0)

        # Run CKF
        t_start = time.perf_counter()
        all_ms, all_Ps = ckf_augmented(
            augmented_system=aug_sys,
            m0_aug=m0_aug,
            P0_aug=P0_aug,
            y_seq=y_meas,
            t_seq=t,
            Q_aug=Q_aug,
            R=R,
            drive_seq=drive_seq,
            max_epochs=n_epochs,
            verbose=True,
        )
        total_time = time.perf_counter() - t_start
        print(f"\n  Total training time: {total_time:.1f}s")

        # Extract final results
        ms_final = all_ms[-1]  # (T, n_aug)
        Ps_final = all_Ps[-1]  # (T, n_aug, n_aug)

        # Extract estimated states
        states_est = ms_final[:, n_w:]  # (T, n_s)
        weights_est = ms_final[:, :n_w]  # (T, n_w)

        # Extract state covariances
        state_covs = Ps_final[:, n_w:, n_w:]  # (T, n_s, n_s)

        # Compute RMSE for each state
        print(f"\n  RMSE per state:")
        rmse_dict = {}
        for i, label in enumerate(state_labels):
            err = (states_est[:, i] - y_true[:, i]).detach().numpy()
            rmse = np.sqrt(np.mean(err**2))
            rmse_dict[label] = rmse
            tag = " (learned)" if i == replaced_idx else ""
            print(f"    {label:>10s}: {rmse:.6f}{tag}")

        total_rmse = np.sqrt(sum(v**2 for v in rmse_dict.values()) / len(rmse_dict))
        print(f"    {'TOTAL':>10s}: {total_rmse:.6f}")

        # Save data
        np.savez(
            os.path.join(out_dir, "results.npz"),
            t=t.numpy(),
            y_meas=y_meas.numpy(),
            y_true=y_true.numpy(),
            states_est=states_est.detach().numpy(),
            weights_est=weights_est.detach().numpy(),
            state_labels=np.array(state_labels),
            meas_idx=np.array(meas_idx),
            replaced_idx=replaced_idx,
            rmse=np.array([rmse_dict[l] for l in state_labels]),
            total_rmse=total_rmse,
            q_weight_scale=q_weight_scale,
            p_weight_scale=p_weight_scale,
            n_epochs=n_epochs,
        )

        # Save state covariance diagonals for confidence intervals
        state_cov_diag = torch.diagonal(state_covs, dim1=1, dim2=2).detach().numpy()
        np.save(os.path.join(out_dir, "state_cov_diag.npy"), state_cov_diag)

        # Save weight evolution for all epochs
        all_weights = []
        all_states_est = []
        for ms_epoch in all_ms:
            all_weights.append(ms_epoch[:, :n_w].detach().numpy())
            all_states_est.append(ms_epoch[:, n_w:].detach().numpy())
        np.savez(
            os.path.join(out_dir, "epoch_history.npz"),
            weights=[w for w in all_weights],
            states=[s for s in all_states_est],
        )

        print(f"\n  Results saved to: {out_dir}")

        # --- Standalone rollout (no measurement corrections) ---
        print(f"\n  Running standalone rollout from true IC with learned weights...")
        w_final = ms_final[-1, :n_w].detach()
        y_rollout = rollout_from_weights(aug_sys, w_final, y_true, t, drive_seq)

        rollout_rmse = {}
        print(f"  Rollout RMSE:")
        for i, label in enumerate(state_labels):
            err = y_rollout[:, i] - y_true[:, i].numpy()
            r = np.sqrt(np.mean(err**2))
            rollout_rmse[label] = r
            tag = " (learned)" if i == replaced_idx else ""
            print(f"    {label:>10s}: {r:.6f}{tag}")

        # Save rollout data
        np.savez(
            os.path.join(out_dir, "rollout.npz"),
            t=t.numpy(),
            y_true=y_true.numpy(),
            y_rollout=y_rollout,
            rollout_rmse=np.array([rollout_rmse[l] for l in state_labels]),
        )

        rollout_data = {
            "t": t.numpy(),
            "y_true": y_true.numpy(),
            "y_rollout": y_rollout,
            "rmse": rollout_rmse,
        }

        # Plot
        plot_all(
            t=t.numpy(),
            y_true=y_true.numpy(),
            y_meas=y_meas.numpy(),
            states_est=states_est.detach().numpy(),
            state_cov_diag=state_cov_diag,
            weights_est=weights_est.detach().numpy(),
            state_labels=state_labels,
            meas_idx=meas_idx,
            replaced_idx=replaced_idx,
            rmse_dict=rmse_dict,
            out_dir=out_dir,
            model_name=model_name,
            all_epoch_states=[s for s in all_states_est] if n_epochs > 1 else None,
            rollout_data=rollout_data,
        )

        return rmse_dict, out_dir
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            f_console.flush()
            f_console.close()
        except Exception:
            pass
        # Convenience: also write alternate filenames.
        try:
            if os.path.isfile(console_path):
                if not os.path.isfile(console_path_alt):
                    with open(console_path, "rb") as src, open(console_path_alt, "wb") as dst:
                        dst.write(src.read())
                if not os.path.isfile(console_path_space):
                    with open(console_path, "rb") as src, open(console_path_space, "wb") as dst:
                        dst.write(src.read())
        except Exception:
            pass

def main():
    """Run one or more configured RBSE profiles."""
    parser = argparse.ArgumentParser(
        description="RBSE training: CKF-based joint state and NN-parameter estimation"
    )
    parser.add_argument("--models", nargs="+", choices=(*VALID_MODELS, "all"), default=("ho",),
                        help="One or more configured RBSE models, or 'all'.")
    parser.add_argument("--list-models", action="store_true", help="Print available ODE systems and exit")
    parser.add_argument("--epochs", type=int, default=1, help="Number of CKF passes.")
    parser.add_argument("--run-id", type=int, default=None, help="Optional run identifier.")
    parser.add_argument("--output-dir", "--save-dir", dest="save_dir", type=str, default=None,
                        help="Output root (default: RBSE/results).")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(VALID_MODELS))
        return
    if "all" in args.models and len(args.models) != 1:
        raise SystemExit("Use --models all by itself, or list explicit model names.")
    systems = VALID_MODELS if args.models == ("all",) else args.models
    for sys_name in systems:
        run_rbse(model_name=sys_name, n_epochs=args.epochs, run_id=args.run_id, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

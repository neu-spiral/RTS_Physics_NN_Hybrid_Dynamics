# Generic Cubature Kalman Filter (CKF) + RTS smoother
# Works for arbitrary state dimension n and measurement dimension m.
#
# Interface expected by the training pipeline:
#   ckf_filter(physics, m0, P0, y_seq, t_seq) -> (ms, Ps)
#   rts_smoother(physics, ms, Ps, t_seq) -> (ms_s, Ps_s)
#
# Requirements on `physics`:
#   - physics.propagate_batch(X, dt) : (M, n) -> (M, n)
#   - physics.h(X)                  : (M, n) -> (M, m)
#   - physics.Q                     : (n, n) process noise
#   - physics.R                     : (m, m) measurement noise

import math
import os
import time
import torch


def _sigma_points(m: torch.Tensor, P: torch.Tensor, eps: float = 1e-6, eye: torch.Tensor = None):
    """
    2n cubature sigma points around mean m with covariance P.
    OPTIMIZED: Skip expensive eigenvalue checks, use fast Cholesky with fallback.

    m : (n,)
    P : (n, n)
    eps : small regularization for numerical stability

    Returns:
        X : (2n, n)
    """
    n = m.shape[0]
    device = m.device
    dtype = m.dtype

    # Ensure symmetry and add small regularization (faster than eigenvalue check)
    if eye is None:
        eye = torch.eye(n, dtype=dtype, device=device)
    P_reg = 0.5 * (P + P.transpose(0, 1)) + eps * eye

    # Cholesky factor - use cholesky_ex to avoid exception overhead
    S, info = torch.linalg.cholesky_ex(P_reg)
    if info.item() != 0:
        # Fallback: use diagonal approximation (fast, loses correlations but stable)
        S = torch.diag(torch.sqrt(torch.diag(P_reg).clamp(min=eps)))

    # Generate sigma points efficiently
    gamma = math.sqrt(float(n))
    # Use pre-allocated identity for speed
    S_scaled = gamma * S  # (n, n)

    # Stack positive and negative sigma points
    X = torch.cat([
        m.unsqueeze(0) + S_scaled.t(),  # (n, n) -> add rows
        m.unsqueeze(0) - S_scaled.t()   # (n, n) -> subtract rows
    ], dim=0)  # (2n, n)

    return X


def _clip_sigma_points(physics, X):
    """
    Clip sigma points to enforce physical constraints (e.g., non-negativity).
    This prevents numerical issues when sigma points go into unphysical regions.
    """
    if hasattr(physics, 'clip_states'):
        return physics.clip_states(X)
    elif hasattr(physics, 'physics') and hasattr(physics.physics, 'clip_states'):
        return physics.physics.clip_states(X)
    return X


def _replaced_indices_from_attr(value):
    # Support the replaced indices from attr step in this module.
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _get_replaced_attr(physics):
    # Return replaced attr.
    base = getattr(physics, "physics", None)
    value = getattr(physics, "replaced_indices", None)
    if value is None and base is not None:
        value = getattr(base, "replaced_indices", None)
    if value is None:
        value = getattr(physics, "replaced_index", None)
    if value is None and base is not None:
        value = getattr(base, "replaced_index", None)
    return value


def _enforce_min_covariance(P, m, nonneg_indices=None, min_cv_ratio=0.003, replaced_index=None):
    """
    Ensure minimum covariance for non-negative states based on their current values.

    For states that must be non-negative and can grow from near-zero to large values,
    the covariance should scale with the state magnitude to prevent sigma points
    from going negative.

    Args:
        P: (n, n) covariance matrix
        m: (n,) state mean
        nonneg_indices: list of indices for non-negative states
        min_cv_ratio: minimum coefficient of variation (std/mean) to maintain

    Returns:
        P: (n, n) adjusted covariance matrix
    """
    if nonneg_indices is None:
        return P

    n = P.shape[0]
    idxs = [idx for idx in nonneg_indices if idx < n]
    if len(idxs) == 0:
        return P

    idxs_t = torch.tensor(idxs, device=m.device, dtype=torch.long)
    state_vals = m[idxs_t].abs()
    min_var = (min_cv_ratio * state_vals) ** 2
    max_var = (state_vals / (math.sqrt(n) + 0.1)) ** 2  # Ensure sigma points stay positive

    if replaced_index is None:
        replaced_set = set()
    elif isinstance(replaced_index, (list, tuple, set)):
        replaced_set = {int(v) for v in replaced_index}
    else:
        replaced_set = {int(replaced_index)}

    # The replaced (unobserved) state needs much more variance freedom
    # so sigma points can explore the dynamics-only trajectory.
    if replaced_set:
        repl_mask = torch.zeros_like(idxs_t, dtype=torch.bool)
        for ridx in replaced_set:
            repl_mask = repl_mask | (idxs_t == ridx)
        max_var = torch.where(repl_mask, torch.full_like(max_var, float('inf')), max_var)

    valid = state_vals > 1e-10
    min_var = torch.where(valid, min_var, torch.zeros_like(min_var))
    max_var = torch.where(valid, max_var, torch.full_like(max_var, float('inf')))

    P_diag = torch.diagonal(P)
    current_var = P_diag[idxs_t]
    new_var = torch.minimum(torch.maximum(current_var, min_var), max_var)

    P_new = P.clone()
    P_new[idxs_t, idxs_t] = new_var
    return P_new


def _step_change_thresholds(n: int, dtype, device, *, smoother: bool):
    """Absolute per-step limits in the state space passed to CKF/RTS."""
    value = 1.0 if smoother else 1.5
    return torch.full((n,), value, dtype=dtype, device=device)


def _ckf_predict(physics, m_prev, P_prev, dt: float, drive_seq=None, drive_idx=None, eye=None):
    """
    CKF time update step.

    physics   : has propagate_batch(X, dt) and Q
    m_prev    : (n,)
    P_prev    : (n, n)
    dt        : float
    drive_seq : (T,) or (T, d) optional exogenous drive
    drive_idx : int, index into drive_seq for this step

    Returns:
        m_pred : (n,)
        P_pred : (n, n)
        X_pred : (2n, n) sigma points after propagation
    """
    n = m_prev.shape[0]
    # Generate sigma points at previous mean/cov
    X = _sigma_points(m_prev, P_prev, eye=eye)        # (2n, n)

    # IMPORTANT: Clip sigma points to enforce physical constraints
    # This prevents issues with states like TA, TUA that must be non-negative
    X = _clip_sigma_points(physics, X)

    # Propagate through dynamics (with optional drive)
    if drive_seq is not None and drive_idx is not None:
        # Use drive-aware propagation for retina
        drive_val = drive_seq[drive_idx]
        if drive_val.dim() == 0:
            drive_val = drive_val.unsqueeze(0)  # scalar -> (1,)
        # Check if physics has rhs_true_with_drive for custom propagation
        if hasattr(physics, 'rhs_true_with_drive'):
            # Custom propagation with Pin for retina
            X_pred = _propagate_batch_with_drive(physics, X, float(dt), drive_val)
        else:
            X_pred = physics.propagate_batch(X, float(dt))
    else:
        X_pred = physics.propagate_batch(X, float(dt))  # (2n, n)

    # Equal weights
    W = 1.0 / float(2 * n)

    # Predicted mean
    m_pred = X_pred.mean(dim=0)  # (n,)

    # Predicted covariance
    dX = X_pred - m_pred.unsqueeze(0)        # (2n, n)
    P_pred = dX.transpose(0, 1) @ dX         # (n, n)
    # Treat physics.Q as a continuous-time process noise intensity and scale by dt.
    # This makes uncertainty growth time-consistent when dt varies.
    dt_abs = float(dt) if float(dt) >= 0.0 else -float(dt)
    P_pred = P_pred * W + physics.Q * dt_abs

    return m_pred, P_pred, X_pred


def _propagate_batch_with_drive(physics, X, dt, drive):
    """
    Propagate batch of sigma points using RK4 with drive signal.
    For retina integrator with Pin(t).

    Args:
        physics: Actually the integrator (LearnedODERetina4WithPin)
        X: Batch of sigma points (2n, n_states)
        dt: Time step
        drive: Drive signal (Pin)
    """
    if hasattr(physics, "rk4_step_with_pin"):
        X_next = physics.rk4_step_with_pin(X, dt, drive)
    elif hasattr(physics, "rk4_step_with_drive"):
        X_next = physics.rk4_step_with_drive(X, dt, drive)
    else:
        total_dt = float(dt)
        if total_dt == 0.0:
            X_next = X
        else:
            sign = 1.0
            if total_dt < 0.0:
                sign = -1.0
                total_dt = -total_dt

            max_dt_attr = getattr(physics, "max_dt", None)
            try:
                max_dt = float(max_dt_attr.item()) if max_dt_attr is not None else 0.0
            except Exception:
                try:
                    max_dt = float(max_dt_attr)
                except Exception:
                    max_dt = 0.0

            n_sub = 1
            if max_dt > 0.0 and total_dt > max_dt:
                n_sub = int(total_dt / max_dt + 0.999999)
                if n_sub < 1:
                    n_sub = 1
            h = sign * (total_dt / float(n_sub))

            X_next = X
            for _ in range(n_sub):
                k1 = physics.rhs_true_with_drive(X_next, drive)
                k2 = physics.rhs_true_with_drive(X_next + 0.5 * h * k1, drive)
                k3 = physics.rhs_true_with_drive(X_next + 0.5 * h * k2, drive)
                k4 = physics.rhs_true_with_drive(X_next + h * k3, drive)
                X_next = X_next + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Access wrapper-aware clipping first so scaled wrappers stay scaled.
    if hasattr(physics, 'clip_states'):
        return physics.clip_states(X_next)
    elif hasattr(physics, 'physics') and hasattr(physics.physics, 'clip_states'):
        return physics.physics.clip_states(X_next)
    else:
        return X_next


def ckf_filter(physics, m0, P0, y_seq, t_seq, drive_seq=None, nonneg_indices=None):
    """
    Forward CKF over a full time series.

    physics : object with propagate_batch, h, Q, R
    m0      : (n,) initial mean
    P0      : (n, n) initial covariance
    y_seq   : (T, m) measurements
    t_seq   : (T,) time stamps (torch tensor)
    drive_seq : (T,) or (T, d) optional exogenous drive (e.g., Pin for retina)
    nonneg_indices : list of state indices that must remain non-negative

    Returns:
        ms : (T, n) filtered means
        Ps : (T, n, n) filtered covariances
    """
    device = m0.device
    dtype = m0.dtype

    T = y_seq.shape[0]
    n = m0.shape[0]

    ms = torch.zeros(T, n, dtype=dtype, device=device)
    Ps = torch.zeros(T, n, n, dtype=dtype, device=device)

    ms[0] = m0
    Ps[0] = P0

    m = m0
    P = P0

    # Pre-compute dt values (avoid repeated tensor operations in loop)
    dt_list = (t_seq[1:] - t_seq[:-1]).tolist()
    W = 1.0 / float(2 * n)
    eye_n = torch.eye(n, dtype=dtype, device=device)
    m_dim = y_seq.shape[1]
    eye_m = torch.eye(m_dim, dtype=dtype, device=device)
    # Step-change constraints help systems with strict physical bounds, but can
    # badly distort fast systems (e.g. HH voltage spikes).
    use_step_change_limits = nonneg_indices is not None
    abs_threshold_filter = _step_change_thresholds(n, dtype, device, smoother=False)
    const_1e8 = torch.tensor(1e-8, dtype=dtype, device=device)
    const_1e6 = torch.tensor(1e-6, dtype=dtype, device=device)
    const_1e4 = torch.tensor(1e-4, dtype=dtype, device=device)




    progress_every = int(os.getenv("ODE_CKF_PROGRESS_EVERY", "0"))
    show_progress = progress_every > 0
    show_total = os.getenv("ODE_CKF_SHOW_TIME", "0") == "1"
    t_start = time.perf_counter() if (show_progress or show_total) else None

    # Enforce adaptive covariance bounds for non-negative states at initialization
    if nonneg_indices is not None:
        P = _enforce_min_covariance(P, m, nonneg_indices, replaced_index=_get_replaced_attr(physics))

    for k in range(1, T):
        dt = dt_list[k - 1]

        # Time update (with optional drive)
        m_pred, P_pred, X_pred = _ckf_predict(
            physics,
            m,
            P,
            dt,
            drive_seq,
            k - 1 if drive_seq is not None else None,
            eye=eye_n,
        )

        # Check for NaN/Inf in prediction and use previous state if failed
        if torch.any(~torch.isfinite(m_pred)) or torch.any(~torch.isfinite(P_pred)):
            m_pred = m.clone()
            P_pred = P.clone()
            # Generate new sigma points from stable state
            X_pred = _sigma_points(m_pred, P_pred, eye=eye_n)
            X_pred = _clip_sigma_points(physics, X_pred)

        # Measurement update
        Y = physics.h(X_pred)                     # (2n, m)
        m_meas = Y.mean(dim=0)                    # (m,)
        dY = Y - m_meas.unsqueeze(0)              # (2n, m)

        S = dY.transpose(0, 1) @ dY               # (m, m)
        S = S * W + physics.R                     # measurement covariance

        dX = X_pred - m_pred.unsqueeze(0)         # (2n, n)
        C = dX.transpose(0, 1) @ dY               # (n, m)
        C = C * W

        # Kalman gain using Cholesky for stability
        # Add regularization to S for numerical stability
        # Ensure S is symmetric
        S = 0.5 * (S + S.transpose(0, 1))

        # Use adaptive regularization based on S diagonal magnitude
        # More aggressive regularization for numerical stability
        S_diag = S.diagonal()
        S_diag_max = S_diag.abs().max()
        S_diag_min = S_diag.min()

        # If any diagonal is negative or too small, add regularization
        reg_scale = torch.where(
            S_diag_min < const_1e8,
            torch.maximum(const_1e4, S_diag_max * 1e-4),
            torch.maximum(const_1e6, S_diag_max * 1e-6),
        )

        S_reg = S + reg_scale * eye_m

        # Compute Kalman gain using robust method
        # Try Cholesky (fast) then fall back to solve / lstsq
        S_chol, info = torch.linalg.cholesky_ex(S_reg)
        if info.item() == 0:
            K_t = torch.cholesky_solve(C.transpose(0, 1), S_chol)
            K = K_t.transpose(0, 1)
        else:
            # Fall back to solve using LU decomposition
            try:
                K_t = torch.linalg.solve(S_reg, C.transpose(0, 1))
                K = K_t.transpose(0, 1)
            except RuntimeError:
                # Last resort: use least squares which is most robust
                K_t, _ = torch.linalg.lstsq(S_reg, C.transpose(0, 1))
                K = K_t.transpose(0, 1)

        innov = y_seq[k] - m_meas                 # (m,)
        m_new = m_pred + K @ innov                # (n,)

        # Check for NaN/Inf and fall back to prediction if update failed
        if torch.any(~torch.isfinite(m_new)):
            m = m_pred.clone()  # Fall back to prediction
        else:
            # Clip the mean state to enforce physical constraints
            m_clipped = _clip_sigma_points(physics, m_new.unsqueeze(0)).squeeze(0)

            if use_step_change_limits:
                # ADDITIONAL CONSTRAINT: Limit maximum change per time step
                # Prevent huge spikes in filter by limiting state changes.
                max_relative_change = 0.4  # Max 40% change per time step (more lenient than smoother)
                delta = m_clipped - ms[k-1]  # Change from previous filtered state

                # For each state, compute maximum allowed absolute change
                abs_threshold = abs_threshold_filter
                max_abs_change = torch.maximum(
                    max_relative_change * torch.abs(ms[k-1]),
                    abs_threshold
                )

                # Let learned states move freely under their dynamics.
                _ri = _get_replaced_attr(physics)
                for _ridx in _replaced_indices_from_attr(_ri):
                    if 0 <= _ridx < n:
                        max_abs_change[_ridx] = float('inf')

                # Clip the delta to be within allowed range
                delta_clipped = torch.clamp(delta, -max_abs_change, max_abs_change)

                # Apply clipped change
                m = ms[k-1] + delta_clipped
            else:
                # Fast systems: accept the clipped update as-is.
                m = m_clipped

        # Joseph form covariance update (more numerically stable)
        # P = (Identity - K*H) * P_pred * (Identity - K*H)^T + K * R * K^T
        # CKF uses an implicit H, so the standard update is applied with care.
        # Approximate the update
        P_new = P_pred - K @ S @ K.transpose(0, 1)

        # Check for NaN/Inf in covariance and fall back to prediction if failed
        if torch.any(~torch.isfinite(P_new)):
            P = P_pred.clone()
        else:
            P = P_new

        # Ensure symmetry
        P = 0.5 * (P + P.transpose(0, 1))

        # Ensure positive definiteness using diagonal check (faster than eigenvalue)
        # If any diagonal is negative or too small, add regularization
        P_diag = torch.diag(P)
        P_diag_min = P_diag.min()
        need_reg = P_diag_min < const_1e8
        reg = torch.where(
            need_reg,
            torch.maximum(const_1e6, P_diag_min.abs() + const_1e6),
            torch.zeros_like(P_diag_min),
        )
        P = P + reg * eye_n

        # Enforce adaptive covariance bounds for non-negative states
        if nonneg_indices is not None:
            P = _enforce_min_covariance(P, m, nonneg_indices, replaced_index=_get_replaced_attr(physics))

        ms[k] = m
        Ps[k] = P

        if show_progress and (k % progress_every == 0 or k == T - 1):
            elapsed = time.perf_counter() - t_start
            rate = k / elapsed if elapsed > 0 else 0.0
            eta = (T - 1 - k) / rate if rate > 0 else float('inf')
            print(f"[ckf] step {k}/{T - 1}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    if show_progress or show_total:
        total = time.perf_counter() - t_start
        print(f"[ckf] done in {total:.2f}s")

    return ms, Ps


def rts_smoother(physics, ms, Ps, t_seq, drive_seq=None, nonneg_indices=None):
    """
    Rauch–Tung–Striebel smoother on top of CKF filter output.

    physics   : same object as CKF (has propagate_batch and Q)
    ms        : (T, n) filtered means   from ckf_filter
    Ps        : (T, n, n) filtered covs from ckf_filter
    t_seq     : (T,) time stamps
    drive_seq : (T,) or (T, d) optional exogenous drive

    Returns:
        ms_s : (T, n) smoothed means
        Ps_s : (T, n, n) smoothed covariances
    """
    device = ms.device
    dtype = ms.dtype

    T, n = ms.shape

    ms_s = torch.zeros_like(ms)
    Ps_s = torch.zeros_like(Ps)

    # start from last time step
    ms_s[-1] = ms[-1]
    Ps_s[-1] = Ps[-1]

    # Pre-compute dt values (avoid repeated tensor operations in loop)
    dt_list = (t_seq[1:] - t_seq[:-1]).tolist()
    W = 1.0 / float(2 * n)
    eye_n = torch.eye(n, dtype=dtype, device=device)
    use_step_change_limits = nonneg_indices is not None
    abs_threshold_smooth = _step_change_thresholds(n, dtype, device, smoother=True)
    const_1e6 = torch.tensor(1e-6, dtype=dtype, device=device)
    const_1 = torch.tensor(1.0, dtype=dtype, device=device)

    progress_every = int(os.getenv("ODE_RTS_PROGRESS_EVERY", "0"))
    show_progress = progress_every > 0
    show_total = os.getenv("ODE_RTS_SHOW_TIME", "0") == "1"
    t_start = time.perf_counter() if (show_progress or show_total) else None

    # backward pass
    for k in range(T - 2, -1, -1):
        dt = dt_list[k]

        m_k = ms[k]          # (n,)
        P_k = Ps[k]          # (n, n)

        # Skip if current filtered state has NaN/Inf
        if torch.any(~torch.isfinite(m_k)) or torch.any(~torch.isfinite(P_k)):
            ms_s[k] = ms_s[k + 1].clone()
            Ps_s[k] = Ps_s[k + 1].clone()
            continue

        # Predict from k to k+1 (with optional drive)
        m_pred, P_pred, X_pred = _ckf_predict(
            physics,
            m_k,
            P_k,
            dt,
            drive_seq,
            k if drive_seq is not None else None,
            eye=eye_n,
        )

        # Check for NaN/Inf in prediction
        if torch.any(~torch.isfinite(m_pred)) or torch.any(~torch.isfinite(P_pred)):
            ms_s[k] = m_k.clone()
            Ps_s[k] = P_k.clone()
            continue

        # Cross-covariance between x_k and x_{k+1}
        # Need original sigma points Xk for cross-covariance computation
        Xk = _sigma_points(m_k, P_k, eye=eye_n)             # (2n, n)
        Xk = _clip_sigma_points(physics, Xk)     # Clip sigma points
        dXk = Xk - m_k.unsqueeze(0)              # (2n, n)
        dX_pred = X_pred - m_pred.unsqueeze(0)   # (2n, n)

        Ck = dXk.transpose(0, 1) @ dX_pred       # (n, n)
        Ck = Ck * W

        # Smoothing gain Gk = Ck * P_pred^{-1}
        # Add adaptive regularization for numerical stability
        P_diag_max = torch.diagonal(P_pred).abs().max()
        finite = torch.isfinite(P_diag_max) & (P_diag_max > 0)
        safe_max = torch.where(finite, P_diag_max, const_1)
        reg_scale = torch.maximum(const_1e6, safe_max * 1e-6)
        P_pred_reg = P_pred + reg_scale * eye_n

        # Ensure P_pred_reg is symmetric
        P_pred_reg = 0.5 * (P_pred_reg + P_pred_reg.transpose(0, 1))

        # Try Cholesky, fall back to solve, then pseudo-inverse
        P_pred_chol, info = torch.linalg.cholesky_ex(P_pred_reg)
        if info.item() == 0:
            Gk_t = torch.cholesky_solve(Ck.transpose(0, 1), P_pred_chol)  # (n, n)
            Gk = Gk_t.transpose(0, 1)                                     # (n, n)
        else:
            try:
                # Try LU solve
                Gk_t = torch.linalg.solve(P_pred_reg, Ck.transpose(0, 1))
                Gk = Gk_t.transpose(0, 1)
            except RuntimeError:
                try:
                    # Fall back to pseudo-inverse if solve fails
                    Gk = Ck @ torch.linalg.pinv(P_pred_reg)                   # (n, n)
                except RuntimeError:
                    # If everything fails, use zero gain (no smoothing correction)
                    Gk = torch.zeros_like(Ck)

        # Apply damping only when enforcing strong physical constraints.
        # Use lighter damping for the replaced (unobserved) state so the
        # smoother can propagate cross-correlation information more freely.
        if use_step_change_limits:
            damp_factor = 0.7  # balanced damping for constrained systems
            _ri = _get_replaced_attr(physics)
            _ridxs = [idx for idx in _replaced_indices_from_attr(_ri) if 0 <= idx < n]
            if _ridxs:
                damp_vec = torch.full((n,), damp_factor, dtype=dtype, device=device)
                for _ridx in _ridxs:
                    damp_vec[_ridx] = 0.85 if n == 6 else 0.95
                Gk_damped = damp_vec.unsqueeze(1) * Gk
            else:
                Gk_damped = damp_factor * Gk
        else:
            Gk_damped = Gk

        # Compute smoothed state
        m_s_new = m_k + Gk_damped @ (ms_s[k + 1] - m_pred)
        P_s_new = P_k + Gk_damped @ (Ps_s[k + 1] - P_pred) @ Gk_damped.transpose(0, 1)

        # Check for NaN/Inf and fall back to filtered values
        if torch.any(~torch.isfinite(m_s_new)):
            ms_s[k] = m_k.clone()
        else:
            m_s_clipped = _clip_sigma_points(physics, m_s_new.unsqueeze(0)).squeeze(0)
            if use_step_change_limits:
                # ADDITIONAL CONSTRAINT: Limit maximum relative change per step
                max_relative_change = 0.3  # Max 30% change per time step
                delta = m_s_clipped - m_k

                abs_threshold = abs_threshold_smooth
                max_abs_change = torch.maximum(
                    max_relative_change * torch.abs(m_k),
                    abs_threshold
                )

                # Let learned states move freely under their dynamics.
                for _ridx in _replaced_indices_from_attr(_ri):
                    if 0 <= _ridx < n:
                        max_abs_change[_ridx] = float('inf')

                delta_clipped = torch.clamp(delta, -max_abs_change, max_abs_change)
                ms_s[k] = m_k + delta_clipped
            else:
                ms_s[k] = m_s_clipped

        if torch.any(~torch.isfinite(P_s_new)):
            Ps_s[k] = P_k.clone()
        else:
            # Ensure symmetry
            Ps_s[k] = 0.5 * (P_s_new + P_s_new.transpose(0, 1))

        if show_progress and (k % progress_every == 0 or k == 0):
            elapsed = time.perf_counter() - t_start
            done = (T - 1) - k
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = k / rate if rate > 0 else float('inf')
            print(f"[rts] step {done}/{T - 1}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    if show_progress or show_total:
        total = time.perf_counter() - t_start
        print(f"[rts] done in {total:.2f}s")

    return ms_s, Ps_s

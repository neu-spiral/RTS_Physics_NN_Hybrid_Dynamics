# ckf_augmented.py
# Cubature Kalman Filter for augmented state [NN_weights, ODE_states].
#
# The augmented state vector is:
#   x_aug = [w_1, ..., w_p, s_1, ..., s_n]
#
# State transition:
#   w_{k+1} = w_k                              (random walk for NN weights)
#   s_{k+1} = s_k + dt * f_hybrid(s_k, w_k)   (ODE integration)
#
# Measurement model:
#   y_k = H * s_k   (extract measured state components)

import math
import time
import torch


def _blend_observed_states(augmented_system, x_aug, y_obs):
    """Optionally re-anchor directly observed states to the measurement."""
    blend = float(getattr(augmented_system, "observed_state_blend", 0.0))
    if blend <= 0.0:
        return x_aug

    n_w = augmented_system.n_weights
    for meas_col, state_i in enumerate(augmented_system.meas_idx):
        state_idx = n_w + state_i
        x_aug[state_idx] = (1.0 - blend) * x_aug[state_idx] + blend * y_obs[meas_col]
    return x_aug


def sigma_points(m, P, eps=1e-6):
    """
    Generate 2n cubature sigma points.

    m : (n,)
    P : (n, n)

    Returns: (2n, n)
    """
    n = m.shape[0]
    device = m.device
    dtype = m.dtype

    eye = torch.eye(n, dtype=dtype, device=device)
    P_reg = 0.5 * (P + P.T) + eps * eye

    S, info = torch.linalg.cholesky_ex(P_reg)
    if info.item() != 0:
        # Fallback: diagonal approximation
        S = torch.diag(torch.sqrt(torch.diag(P_reg).clamp(min=eps)))

    gamma = math.sqrt(float(n))
    S_scaled = gamma * S

    X = torch.cat([
        m.unsqueeze(0) + S_scaled.T,
        m.unsqueeze(0) - S_scaled.T,
    ], dim=0)
    return X


def ckf_augmented(
    augmented_system,
    m0_aug,
    P0_aug,
    y_seq,
    t_seq,
    Q_aug,
    R,
    drive_seq=None,
    max_epochs=1,
    verbose=True,
):
    """
    Run the CKF over the time series with augmented state.

    This is the core algorithm from the paper: the CKF simultaneously
    estimates ODE states and NN parameters at each timestep.

    Parameters
    ----------
    augmented_system : AugmentedSystem
        Has propagate(x_aug, dt, drive=None) and measure(x_aug) methods.
    m0_aug : (n_aug,)
        Initial augmented state mean [w_init, s_init].
    P0_aug : (n_aug, n_aug)
        Initial augmented state covariance.
    y_seq : (T, m)
        Measurement sequence.
    t_seq : (T,)
        Timestamps.
    Q_aug : (n_aug, n_aug)
        Process noise for augmented state.
    R : (m, m)
        Measurement noise covariance.
    drive_seq : (T,) or (T, d) or None
        Exogenous drive signal (e.g., Pin for retina).
    max_epochs : int
        Number of passes over the data (re-running CKF with updated
        weight estimates from the previous pass).
    verbose : bool
        Print progress.

    Returns
    -------
    all_ms : list of (T, n_aug) tensors, one per epoch
    all_Ps : list of (T, n_aug, n_aug) tensors, one per epoch
    """
    device = m0_aug.device
    dtype = m0_aug.dtype
    T = y_seq.shape[0]
    n_aug = m0_aug.shape[0]
    m_dim = y_seq.shape[1]

    all_ms = []
    all_Ps = []

    base_m = m0_aug.clone()
    base_P = P0_aug.clone()
    n_w = augmented_system.n_weights
    n_s = augmented_system.n_states

    for epoch in range(max_epochs):
        t_start = time.perf_counter()

        # Re-start each pass from the sequence initial condition, but carry over
        # the latest weight estimate so later epochs refine parameters rather than
        # incorrectly starting from the previous epoch's final state at t=T.
        m = base_m.clone()
        P = base_P.clone()
        if epoch > 0 and all_ms:
            m[:n_w] = all_ms[-1][-1, :n_w]
            P[:n_w, :n_w] = all_Ps[-1][-1, :n_w, :n_w]

        ms = torch.zeros(T, n_aug, dtype=dtype, device=device)
        Ps = torch.zeros(T, n_aug, n_aug, dtype=dtype, device=device)

        dt_list = (t_seq[1:] - t_seq[:-1]).tolist()
        eye_aug = torch.eye(n_aug, dtype=dtype, device=device)
        eye_m = torch.eye(m_dim, dtype=dtype, device=device)

        # Assimilate the first observation so t0 is not just the prior.
        X0 = sigma_points(m, P)
        n_sigma0 = X0.shape[0]
        W0 = 1.0 / float(n_sigma0)
        Y0 = augmented_system.measure_batch(X0)
        m_meas0 = Y0.mean(dim=0)
        dY0 = Y0 - m_meas0.unsqueeze(0)
        S0 = (dY0.T @ dY0) * W0 + R
        dX0 = X0 - m.unsqueeze(0)
        C_xy0 = (dX0.T @ dY0) * W0

        S0 = 0.5 * (S0 + S0.T)
        S0_reg = S0 + 1e-6 * eye_m
        S0_chol, info0 = torch.linalg.cholesky_ex(S0_reg)
        if info0.item() == 0:
            K0 = torch.cholesky_solve(C_xy0.T, S0_chol).T
        else:
            try:
                K0 = torch.linalg.solve(S0_reg, C_xy0.T).T
            except RuntimeError:
                K0 = C_xy0 @ torch.linalg.pinv(S0_reg)

        innov0 = y_seq[0] - m_meas0
        m = m + K0 @ innov0
        m[n_w:] = augmented_system.clip_states(m[n_w:])
        m = _blend_observed_states(augmented_system, m, y_seq[0])
        P = 0.5 * ((P - K0 @ S0 @ K0.T) + (P - K0 @ S0 @ K0.T).T)
        if torch.diag(P).min() < 1e-10:
            P = P + 1e-8 * eye_aug

        ms[0] = m
        Ps[0] = P

        for k in range(1, T):
            dt = dt_list[k - 1]
            drive = None
            if drive_seq is not None:
                drive = drive_seq[k - 1]
                if drive.dim() == 0:
                    drive = drive.unsqueeze(0)

            # --- Time Update (Predict) ---
            X = sigma_points(m, P)  # (2*n_aug, n_aug)
            n_sigma = X.shape[0]

            # Propagate each sigma point through augmented dynamics
            X_pred = augmented_system.propagate_batch(X, dt, drive=drive)

            # Predicted mean and covariance
            W = 1.0 / float(n_sigma)
            m_pred = X_pred.mean(dim=0)

            dX = X_pred - m_pred.unsqueeze(0)
            P_pred = (dX.T @ dX) * W + Q_aug * abs(dt)

            # NaN/Inf check
            if not torch.all(torch.isfinite(m_pred)):
                m_pred = m.clone()
                P_pred = P.clone()

            # --- Measurement Update (Correct) ---
            # Re-generate sigma points from predicted distribution
            X_pred2 = sigma_points(m_pred, P_pred)

            # Predicted measurements
            Y = augmented_system.measure_batch(X_pred2)  # (2*n_aug, m_dim)
            m_meas = Y.mean(dim=0)

            dY = Y - m_meas.unsqueeze(0)
            S = (dY.T @ dY) * W + R

            dX2 = X_pred2 - m_pred.unsqueeze(0)
            C_xy = (dX2.T @ dY) * W  # Cross-covariance (n_aug, m_dim)

            # Kalman gain
            S = 0.5 * (S + S.T)
            S_reg = S + 1e-6 * eye_m
            S_chol, info = torch.linalg.cholesky_ex(S_reg)
            if info.item() == 0:
                K = torch.cholesky_solve(C_xy.T, S_chol).T
            else:
                try:
                    K = torch.linalg.solve(S_reg, C_xy.T).T
                except RuntimeError:
                    K = C_xy @ torch.linalg.pinv(S_reg)

            # Innovation
            innov = y_seq[k] - m_meas
            m_new = m_pred + K @ innov

            # Covariance update
            P_new = P_pred - K @ S @ K.T

            # NaN/Inf fallback
            if not torch.all(torch.isfinite(m_new)):
                m = m_pred.clone()
            else:
                # Clip ODE states to physical bounds
                m = m_new.clone()
                s_part = m[n_w:]
                s_clipped = augmented_system.clip_states(s_part)
                m[n_w:] = s_clipped
                m = _blend_observed_states(augmented_system, m, y_seq[k])

            if not torch.all(torch.isfinite(P_new)):
                P = P_pred.clone()
            else:
                P = 0.5 * (P_new + P_new.T)

            # Ensure positive semi-definiteness
            P_diag = torch.diag(P)
            min_diag = P_diag.min()
            if min_diag < 1e-10:
                P = P + max(1e-8, float(-min_diag) + 1e-8) * eye_aug

            ms[k] = m
            Ps[k] = P

            if verbose and k % max(1, T // 10) == 0:
                elapsed = time.perf_counter() - t_start
                # Show weight norm and state values for monitoring
                w_norm = m[:n_w].norm().item()
                s_vals = m[n_w:].detach().cpu().numpy()
                print(f"  [CKF-aug] epoch {epoch+1} step {k}/{T-1} "
                      f"elapsed={elapsed:.1f}s |w|={w_norm:.4f}")

        all_ms.append(ms)
        all_Ps.append(Ps)

        elapsed = time.perf_counter() - t_start
        if verbose:
            w_final = ms[-1, :n_w].norm().item()
            print(f"  [CKF-aug] epoch {epoch+1}/{max_epochs} done in {elapsed:.1f}s "
                  f"|w_final|={w_final:.4f}")

    return all_ms, all_Ps

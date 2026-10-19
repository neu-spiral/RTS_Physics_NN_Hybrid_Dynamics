# loss_rollout.py
import torch

from typing import Optional, Sequence, List

torch.set_default_dtype(torch.float32)

def _rollout_clip_value() -> float:
    """Return the shared numerical safety bound used by rollout losses."""
    # Global guardrail used when rollouts diverge before the optimizer can recover.
    return 1e6


def _sanitize_rollout_values(x: torch.Tensor, clip_value: float) -> torch.Tensor:
    """Keep residuals and intermediate tensors finite during unstable rollouts."""
    # Replace NaN/Inf and cap extreme values so one unstable rollout segment
    # cannot poison the accumulated loss or backward pass.
    x = torch.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return torch.clamp(x, min=-clip_value, max=clip_value)


def _sanitize_rollout_state(x: torch.Tensor, integrator, clip_value: float) -> torch.Tensor:
    """Clean a state tensor, then apply model-specific physical state bounds."""
    # Apply numerical cleanup first, then enforce model-specific physical bounds
    # such as nonnegative volumes, pressures, or neural activations.
    x = _sanitize_rollout_values(x, clip_value)
    if hasattr(integrator, "physics") and hasattr(integrator.physics, "clip_states"):
        x = integrator.physics.clip_states(x)
        x = _sanitize_rollout_values(x, clip_value)
    return x


def _near_uniform_rollout_dt(dt_tensor: torch.Tensor):
    """Classify whether timestamps are close enough to use one rounded dt."""
    # Treat tiny timestamp jitter as a uniform sample clock so rollout training
    # can use the batched fast path on converted experimental data.
    if dt_tensor.numel() == 0:
        return False, 0.0, 0.0, 0.0, 0.0, False
    dt_min = dt_tensor.min()
    dt_max = dt_tensor.max()
    dt_mean = dt_tensor.mean()
    span = dt_max - dt_min
    strict_tol = torch.maximum(
        torch.tensor(1e-8, device=dt_tensor.device, dtype=dt_tensor.dtype),
        dt_mean.abs() * 5e-4,
    )
    near_tol = torch.maximum(
        torch.tensor(1e-8, device=dt_tensor.device, dtype=dt_tensor.dtype),
        dt_mean.abs() * 1e-2,
    )
    strict_uniform = bool((span <= strict_tol).item())
    near_uniform = bool((span <= near_tol).item())
    if not near_uniform:
        return False, float(dt_mean.item()), float(dt_min.item()), float(dt_max.item()), float(dt_mean.item()), False

    dt_step = float(dt_mean.item())
    rounded_dt = round(dt_step, 6)
    if rounded_dt > 0.0:
        dt_step = rounded_dt
    return True, dt_step, float(dt_min.item()), float(dt_max.item()), float(dt_mean.item()), not strict_uniform


def _rk4_step_batched_dt(y: torch.Tensor, dt_vec: torch.Tensor, integrator, clip_value: float) -> torch.Tensor:
    """Vectorized RK4 for per-row dt values.

    This exists so nonuniform time grids can still train in one batched tensor
    pass instead of falling back to one Python loop per rollout start.
    """
    if not hasattr(integrator, "f_full"):
        raise AttributeError("integrator has no f_full")

    # Each row may have a different time step, so dt is shaped as (batch, 1)
    # to broadcast cleanly against state vectors and RK derivatives.
    dt = dt_vec.to(device=y.device, dtype=y.dtype).view(-1, 1)
    if dt.numel() == 0:
        return y

    # Keep the sign separately so the sub-step count is based on interval length.
    # CKF/RTS can call propagators backward, so negative dt is allowed.
    sign = torch.where(dt < 0.0, -torch.ones_like(dt), torch.ones_like(dt))
    total_dt = torch.abs(dt)

    min_dt = 0.0
    max_dt = 0.0
    if hasattr(integrator, "min_dt"):
        try:
            min_dt = float(integrator.min_dt.item())
        except Exception:
            min_dt = float(integrator.min_dt)
    if hasattr(integrator, "max_dt"):
        try:
            max_dt = float(integrator.max_dt.item())
        except Exception:
            max_dt = float(integrator.max_dt)

    if min_dt > 0.0:
        total_dt = torch.where((total_dt > 0.0) & (total_dt < min_dt), torch.full_like(total_dt, min_dt), total_dt)

    if max_dt > 0.0:
        n_sub_vec = torch.ceil(total_dt / max_dt).to(dtype=torch.long).clamp_min(1)
    else:
        n_sub_vec = torch.ones_like(total_dt, dtype=torch.long)
    # h is per row. Rows with a larger dt may take more internal RK sub-steps.
    h = sign * total_dt / n_sub_vec.to(dtype=y.dtype)
    n_sub_max = int(n_sub_vec.max().detach().cpu().item())
    n_sub_flat = n_sub_vec.view(-1)

    out = y
    for sub_idx in range(n_sub_max):
        # Only rows that still need another internal sub-step participate.
        # This avoids forcing every row to take the largest number of sub-steps.
        active = torch.nonzero(n_sub_flat > sub_idx, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        out_a = out.index_select(0, active)
        h_a = h.index_select(0, active)
        # f_full is the hybrid RHS, so each RK stage also re-evaluates the NN derivative.
        k1 = integrator.f_full(out_a)
        k2 = integrator.f_full(out_a + 0.5 * h_a * k1)
        k3 = integrator.f_full(out_a + 0.5 * h_a * k2)
        k4 = integrator.f_full(out_a + h_a * k3)
        out_a = out_a + (h_a / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out_a = _sanitize_rollout_state(out_a, integrator, clip_value)
        out = out.index_copy(0, active, out_a)
    return out


def _rk4_step_batched_dt_with_drive(
    y: torch.Tensor,
    dt_vec: torch.Tensor,
    drive: torch.Tensor,
    integrator,
    clip_value: float,
) -> torch.Tensor:
    """Vectorized RK4 for per-row dt values when the ODE has an external drive.

    This is the driven-system version of _rk4_step_batched_dt, used for models
    such as retina where the RHS depends on both state and an input signal.
    """
    if hasattr(integrator, "f_full_with_pin"):
        rhs_fn = integrator.f_full_with_pin
    elif hasattr(integrator, "f_full_with_drive"):
        rhs_fn = integrator.f_full_with_drive
    else:
        raise AttributeError("integrator has no f_full_with_pin or f_full_with_drive")

    # Same per-row dt layout as _rk4_step_batched_dt, with drive carried through
    # every RK stage. This keeps nonuniform retina-style rollouts batched.
    dt = dt_vec.to(device=y.device, dtype=y.dtype).view(-1, 1)
    drive = drive.to(device=y.device, dtype=y.dtype)
    if dt.numel() == 0:
        return y

    sign = torch.where(dt < 0.0, -torch.ones_like(dt), torch.ones_like(dt))
    total_dt = torch.abs(dt)

    min_dt = 0.0
    max_dt = 0.0
    if hasattr(integrator, "min_dt"):
        try:
            min_dt = float(integrator.min_dt.item())
        except Exception:
            min_dt = float(integrator.min_dt)
    if hasattr(integrator, "max_dt"):
        try:
            max_dt = float(integrator.max_dt.item())
        except Exception:
            max_dt = float(integrator.max_dt)

    if min_dt > 0.0:
        total_dt = torch.where((total_dt > 0.0) & (total_dt < min_dt), torch.full_like(total_dt, min_dt), total_dt)

    if max_dt > 0.0:
        n_sub_vec = torch.ceil(total_dt / max_dt).to(dtype=torch.long).clamp_min(1)
    else:
        n_sub_vec = torch.ones_like(total_dt, dtype=torch.long)
    h = sign * total_dt / n_sub_vec.to(dtype=y.dtype)
    n_sub_max = int(n_sub_vec.max().detach().cpu().item())
    n_sub_flat = n_sub_vec.view(-1)

    out = y
    for sub_idx in range(n_sub_max):
        active = torch.nonzero(n_sub_flat > sub_idx, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        out_a = out.index_select(0, active)
        h_a = h.index_select(0, active)
        drive_a = drive.index_select(0, active)
        # The drive is held constant across this observation interval, matching
        # rk4_step_with_drive while allowing each row to use its own dt.
        k1 = rhs_fn(out_a, drive_a)
        k2 = rhs_fn(out_a + 0.5 * h_a * k1, drive_a)
        k3 = rhs_fn(out_a + 0.5 * h_a * k2, drive_a)
        k4 = rhs_fn(out_a + h_a * k3, drive_a)
        out_a = out_a + (h_a / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out_a = _sanitize_rollout_state(out_a, integrator, clip_value)
        out = out.index_copy(0, active, out_a)
    return out


def cov_weighted_multi_step_rollout_loss(
    t: torch.Tensor,
    meas: torch.Tensor,
    smoothed_states: torch.Tensor,
    smoother_covs: torch.Tensor,
    integrator,
    meas_idx,
    horizon_steps: int,
    stride: int = 1,
    pseudo_idx: Optional[Sequence[int]] = None,
    pseudo_weight: float = 0.1,
    meas_cov: Optional[torch.Tensor] = None,
    use_meas_weight: bool = False,
    latent_cov_exponent: float = 0.5,
    latent_cov_scale: float = 0.3,
    full_state_weight: float = 0.0,
    state_scales=None,
    eps: float = 1e-6,
    tbptt_steps: Optional[int] = None,
):
    """
    Covariance-weighted multi-step rollout loss.

    Use this when smoother covariance should control how strongly each
    measurement or latent residual is trusted during hybrid ODE training.

    Brief paper map: rollouts are Eq. 12, residuals are Eq. 13, and the
    multi-horizon sum is Eq. 14. Measurement weighting uses diag(R)^-1 and
    latent weighting uses a softened diag(P)^-1 from the smoother.
    """
    device = meas.device
    T = t.shape[0]
    clip_value = _rollout_clip_value()

    # dt_list is used only by the scalar slow path; the batched path keeps dt
    # as a tensor so it can stay on GPU.
    dt_list: List[float] = (t[1:] - t[:-1]).tolist()
    meas_list: List[int] = list(meas_idx)

    # Accumulators are separated by loss type so the final weighting is explicit.
    sum_meas_loss = torch.tensor(0.0, device=device)
    sum_latent_loss = torch.tensor(0.0, device=device)
    sum_full_state_loss = torch.tensor(0.0, device=device)
    n_meas_terms = 0
    n_latent_terms = 0
    n_full_state_terms = 0

    # Measurement precision from R. Keep only the diagonal for speed.
    if use_meas_weight and (meas_cov is not None):
        if meas_cov.shape[0] != len(meas_list):
            # Some configs store R in full state order. Select measured channels
            # so R lines up with columns of meas and y_pred.
            idx_t = torch.tensor(meas_list, device=meas_cov.device, dtype=torch.long)
            meas_cov = meas_cov.index_select(0, idx_t).index_select(1, idx_t)
        R_diag = torch.diag(meas_cov)
        # Larger measurement noise means smaller penalty for that residual.
        W_y_diag = 1.0 / (R_diag + eps)
    else:
        W_y_diag = None

    # pseudo_idx means "hidden/smoother-supervised channels" rather than measured channels.
    use_pseudo = pseudo_idx is not None and len(pseudo_idx) > 0
    pseudo_list: List[int] = list(pseudo_idx) if use_pseudo else []
    use_full_state = full_state_weight > 0.0
    full_scales = torch.tensor(state_scales, device=device, dtype=t.dtype) if (state_scales is not None and use_full_state) else None

    # Fast path for uniform grids: roll out all starts in one batch per horizon
    # step instead of running a Python loop per start index.
    dt_tensor = t[1:] - t[:-1]
    uniform_dt, dt0, _, _, _, _ = _near_uniform_rollout_dt(dt_tensor)
    if uniform_dt or hasattr(integrator, "f_full"):
        h_full = min(int(horizon_steps), T - 1)
        # Each start begins from the RTS smoother state, then the learned ODE
        # is rolled forward without seeing future measurements.
        starts = torch.arange(0, T - h_full, int(stride), device=device, dtype=torch.long)
        if h_full > 0 and starts.numel() > 0:
            meas_idx_t = torch.tensor(meas_list, device=device, dtype=torch.long)
            pseudo_idx_t = torch.tensor(pseudo_list, device=device, dtype=torch.long) if use_pseudo else None
            x = _sanitize_rollout_state(smoothed_states.index_select(0, starts), integrator, clip_value)

            for n in range(1, h_full + 1):
                # Roll all starts forward by one observation interval. Uniform
                # grids can call the integrator directly; adaptive grids use
                # one dt per row.
                if uniform_dt:
                    x = integrator.rk4_step(x, dt0)
                else:
                    step_dt = dt_tensor.index_select(0, starts + (n - 1))
                    x = _rk4_step_batched_dt(x, step_dt, integrator, clip_value)
                x = _sanitize_rollout_state(x, integrator, clip_value)
                target_idx = starts + n

                # Compare in measurement space first. If physics.h already
                # returns exactly the measured columns, use it; otherwise fall
                # back to selecting meas_idx from the state vector.
                y_pred_full = integrator.physics.h(x)
                y_pred_full = _sanitize_rollout_values(y_pred_full, clip_value)
                if y_pred_full.shape[-1] == len(meas_list):
                    y_pred = y_pred_full
                else:
                    y_pred = _sanitize_rollout_values(x.index_select(1, meas_idx_t), clip_value)
                e_y = _sanitize_rollout_values(y_pred - meas.index_select(0, target_idx), clip_value)

                if W_y_diag is not None:
                    meas_terms = (W_y_diag.view(1, -1) * (e_y * e_y)).mean(dim=1)
                else:
                    meas_terms = (e_y * e_y).mean(dim=1)
                sum_meas_loss = sum_meas_loss + meas_terms.sum()
                n_meas_terms += int(meas_terms.numel())

                if use_pseudo:
                    # Pseudo targets are smoother estimates for states that were
                    # not directly measured. They guide the hidden learned channels.
                    target_states = _sanitize_rollout_values(smoothed_states.index_select(0, target_idx), clip_value)
                    e_x = _sanitize_rollout_values(
                        x.index_select(1, pseudo_idx_t) - target_states.index_select(1, pseudo_idx_t),
                        clip_value,
                    )
                    P_diag = torch.diagonal(
                        smoother_covs.index_select(0, target_idx),
                        dim1=-2,
                        dim2=-1,
                    )
                    P_sub = P_diag.index_select(1, pseudo_idx_t)
                    # Softened smoother precision for hidden-state residuals.
                    # exponent < 1 keeps very small covariance from dominating training.
                    W_x = (1.0 / (P_sub + eps)) ** latent_cov_exponent
                    latent_terms = (W_x * (e_x * e_x)).mean(dim=1)
                    sum_latent_loss = sum_latent_loss + latent_terms.sum()
                    n_latent_terms += int(latent_terms.numel())

                if use_full_state:
                    # Optional stronger constraint: match every state to the
                    # smoother, not only measured and pseudo channels.
                    target_states = _sanitize_rollout_values(smoothed_states.index_select(0, target_idx), clip_value)
                    e_full = _sanitize_rollout_values(x - target_states, clip_value)
                    if full_scales is not None:
                        e_full = e_full * full_scales.unsqueeze(0)
                    full_terms = (e_full * e_full).mean(dim=1)
                    sum_full_state_loss = sum_full_state_loss + full_terms.sum()
                    n_full_state_terms += int(full_terms.numel())

            if n_meas_terms == 0:
                raise RuntimeError("No measurement terms accumulated in cov_weighted_multi_step_rollout_loss.")

            meas_loss = sum_meas_loss / float(n_meas_terms)
            if n_latent_terms > 0:
                latent_loss = sum_latent_loss / float(n_latent_terms)
            else:
                latent_loss = torch.tensor(0.0, device=device)
            if n_full_state_terms > 0:
                full_state_loss = sum_full_state_loss / float(n_full_state_terms)
            else:
                full_state_loss = torch.tensor(0.0, device=device)
            # Final cov-weighted objective: measured rollout fit plus optional
            # smoother-guided latent/full-state penalties.
            return meas_loss + pseudo_weight * latent_cov_scale * latent_loss + full_state_weight * full_state_loss

    # Scalar fallback: same loss as above, but loops over each start and each
    # horizon. This is slower but works for integrators without f_full batching.
    for k in range(0, T - 1, stride):
        max_h = min(horizon_steps, T - 1 - k)
        if max_h <= 0:
            continue

        # Initial condition for this rollout is the smoother estimate at time k.
        x = _sanitize_rollout_state(smoothed_states[k], integrator, clip_value)  # (D,)

        for n in range(1, max_h + 1):
            # Integrate forward n one-step intervals from the same rollout start.
            dt_step = dt_list[k + n - 1]
            x = integrator.rk4_step(x, dt_step)
            x = _sanitize_rollout_state(x, integrator, clip_value)
            k_n = k + n

            # ---- MEASUREMENT-SPACE ERROR ----
            y_pred_full = integrator.physics.h(x.unsqueeze(0)).squeeze(0)
            y_pred_full = _sanitize_rollout_values(y_pred_full, clip_value)
            if y_pred_full.shape[-1] == len(meas_list):
                y_pred = y_pred_full
            else:
                y_pred = _sanitize_rollout_values(x[meas_list], clip_value)
            e_y = _sanitize_rollout_values(y_pred - meas[k_n], clip_value)

            if W_y_diag is not None:
                meas_term = (W_y_diag * (e_y * e_y)).mean()
            else:
                meas_term = (e_y * e_y).mean()

            sum_meas_loss = sum_meas_loss + meas_term
            n_meas_terms += 1

            # ---- LATENT / PSEUDO-STATE ERROR ----
            if use_pseudo:
                x_true = _sanitize_rollout_values(smoothed_states[k_n], clip_value)
                e_x = _sanitize_rollout_values(x[pseudo_list] - x_true[pseudo_list], clip_value)

                P_diag = torch.diagonal(smoother_covs[k_n])
                P_sub = P_diag[pseudo_list]

                # Softened smoother precision for hidden-state residuals.
                # Low smoother variance increases the penalty, high variance relaxes it.
                W_x = (1.0 / (P_sub + eps)) ** latent_cov_exponent

                latent_term = (W_x * (e_x * e_x)).mean()

                sum_latent_loss = sum_latent_loss + latent_term
                n_latent_terms += 1

            # ---- FULL-STATE ROLLOUT ERROR ----
            if use_full_state:
                x_true = _sanitize_rollout_values(smoothed_states[k_n], clip_value)
                e_full = _sanitize_rollout_values(x - x_true, clip_value)
                if full_scales is not None:
                    e_full = e_full * full_scales
                sum_full_state_loss = sum_full_state_loss + (e_full * e_full).mean()
                n_full_state_terms += 1

            if tbptt_steps is not None and tbptt_steps > 0 and (n % tbptt_steps == 0):
                # Truncated BPTT caps graph length for long horizons while
                # still training on later rollout residuals.
                x = x.detach()

    if n_meas_terms == 0:
        raise RuntimeError("No measurement terms accumulated in cov_weighted_multi_step_rollout_loss.")

    meas_loss = sum_meas_loss / float(n_meas_terms)

    if n_latent_terms > 0:
        latent_loss = sum_latent_loss / float(n_latent_terms)
    else:
        latent_loss = torch.tensor(0.0, device=device)
    if n_full_state_terms > 0:
        full_state_loss = sum_full_state_loss / float(n_full_state_terms)
    else:
        full_state_loss = torch.tensor(0.0, device=device)

    # Same weighted sum as the batched path.
    total_loss = meas_loss + pseudo_weight * latent_cov_scale * latent_loss + full_state_weight * full_state_loss
    return total_loss

def _multi_step_rollout_loss_slow(
    t,
    meas,
    smoothed_states,
    integrator,
    meas_idx,
    horizon_steps,
    stride,
    pseudo_idx=None,
    pseudo_weight: float = 0.0,
    drive_seq=None,
    intermediate_weight: float = 1.0,
    full_state_weight: float = 0.0,
    state_scales=None,
    point_weights=None,
    tbptt_steps: Optional[int] = None,
):
    """
    Unweighted multi-step rollout loss with intermediate step losses.

    This compatibility path is kept for integrators that cannot expose a
    batched hybrid RHS, or for cases where exact per-start stepping is needed.

    Same rollout/residual structure as Eq. 12-14, but with plain MSE or
    state scaling instead of covariance weights.

    Args
    ----
    t : (T,) tensor
        Time stamps (monotone).
    meas : (T, m_y) tensor
        Measurements.
    smoothed_states : (T, n_x) tensor
        CKF+RTS smoothed state sequence.
    integrator : object
        Must implement either
          - rk4_step(y, dt)                      (no drive)
          - rk4_step_with_drive(y, dt, drive)    (with drive)
    meas_idx : list[int]
        Indices of states that are directly measured.
    horizon_steps : int
        Max integration depth in *steps*.
    stride : int
        Spacing between starting indices along the sequence.
    pseudo_idx : list[int] or None
        Optional indices of latent states for pseudo loss.
    pseudo_weight : float
        Weight of the pseudo-state penalty.
    drive_seq : (T, d_drive) tensor or None
        Exogenous drive sequence (e.g. Pin(t) for retina). If provided,
        the loss calls integrator.rk4_step_with_drive.
    intermediate_weight : float
        Weight for intermediate step losses (1.0 = equal to final, 0.0 = final only).
    full_state_weight : float
        Weight for full-state loss comparing ALL integrated states to smoother estimates.
        This creates coupling so that learned states (e.g., P4) affect the loss through
        their influence on other states (e.g., P1, P2, P5). Default 0.0 = disabled.
    """
    T = t.shape[0]
    clip_value = _rollout_clip_value()
    if T < 2:
        raise ValueError("t must have length >= 2")

    if meas.dim() == 1:
        meas = meas.unsqueeze(-1)

    n_meas = meas.shape[1]
    if len(meas_idx) != n_meas:
        raise ValueError(
            f"len(meas_idx) = {len(meas_idx)} but meas has {n_meas} columns"
        )

    use_drive = drive_seq is not None
    if use_drive:
        if drive_seq.shape[0] != T:
            raise ValueError("drive_seq must have shape (T, d_drive)")
        if not hasattr(integrator, "rk4_step_with_drive"):
            raise AttributeError(
                "drive_seq provided but integrator has no rk4_step_with_drive"
            )

    # The slow path preserves every original interval, so adaptive time grids
    # do not get rounded to a single dt.
    dt_list: List[float] = (t[1:] - t[:-1]).tolist()

    # meas_list indexes channels compared to real measurements. pseudo_list
    # indexes hidden channels compared to smoother estimates.
    # Example: HO uses meas=[0] and pseudo=[1].
    meas_list: List[int] = list(meas_idx)
    use_pseudo = pseudo_idx is not None and pseudo_weight > 0.0
    pseudo_list: List[int] = list(pseudo_idx) if use_pseudo else []
    use_full_state = full_state_weight > 0.0

    start_indices = list(range(0, T - 1, stride))
    pw = None
    if point_weights is not None:
        # point_weights usually come from smoother confidence. Dropping the
        # weakest starts avoids training on the least reliable local targets.
        pw = point_weights.to(device=meas.device, dtype=t.dtype)
        if pw.shape[0] != T:
            raise ValueError("point_weights must have shape (T,)")
        if len(start_indices) > 0:
            start_tensor = torch.tensor(start_indices, device=meas.device, dtype=torch.long)
            thresh = pw[start_tensor].quantile(0.25)
            start_indices = [k for k in start_indices if bool((pw[k] >= thresh).item())]
            if len(start_indices) == 0:
                raise RuntimeError("All rollout starts filtered by point_weights.")

    total_loss = torch.tensor(0.0, device=meas.device)
    n_terms = 0

    for k in start_indices:
        max_h = min(horizon_steps, T - 1 - k)
        if max_h <= 0:
            continue

        # Each segment is a local initial-value problem starting from the RTS
        # smoother state at index k.
        y = _sanitize_rollout_state(smoothed_states[k : k + 1, :], integrator, clip_value)  # (1, n_x)

        # Accumulate loss at every intermediate step, not just the endpoint.
        # This gives gradients to short-term dynamics even when long rollouts drift.
        for j in range(max_h):
            dt_step = dt_list[k + j]
            if use_drive:
                # Drive is sampled at the current source time for this RK step,
                # e.g. retina Pin(t) used to advance pressures.
                drive_j = drive_seq[k + j].unsqueeze(0)
                y = integrator.rk4_step_with_drive(y, dt_step, drive_j)
            else:
                y = integrator.rk4_step(y, dt_step)
            y = _sanitize_rollout_state(y, integrator, clip_value)

            target_t_idx = k + j + 1  # Target at this step
            is_final_step = (j == max_h - 1)

            # Final point is the named horizon; earlier points are auxiliary
            # constraints controlled by intermediate_weight.
            step_weight = 1.0 if is_final_step else intermediate_weight

            # Measurement loss uses only observed channels, not the whole state.
            y_meas_pred = _sanitize_rollout_values(y[0, meas_list], clip_value)
            y_meas_true = _sanitize_rollout_values(meas[target_t_idx, :], clip_value)
            diff = _sanitize_rollout_values(y_meas_pred - y_meas_true, clip_value)
            loss_step = (diff * diff).mean()

            # Pseudo loss uses smoother-inferred hidden states as soft targets.
            if use_pseudo:
                y_pseudo_pred = _sanitize_rollout_values(y[0, pseudo_list], clip_value)
                y_pseudo_true = _sanitize_rollout_values(smoothed_states[target_t_idx, pseudo_list], clip_value)
                diff_p = _sanitize_rollout_values(y_pseudo_pred - y_pseudo_true, clip_value)
                pseudo_loss = (diff_p * diff_p).mean()
                if pw is not None:
                    # Less confident smoother targets contribute less.
                    pseudo_loss = pseudo_loss * pw[target_t_idx]
                loss_step = loss_step + pseudo_weight * pseudo_loss

            # Full-state loss is optional and stricter than pseudo loss: it
            # asks the entire rollout state to follow the smoother trajectory.
            if use_full_state:
                y_full_pred = _sanitize_rollout_values(y[0, :], clip_value)  # All states
                y_full_true = _sanitize_rollout_values(smoothed_states[target_t_idx, :], clip_value)  # All smoother states
                diff_full = _sanitize_rollout_values(y_full_pred - y_full_true, clip_value)
                full_state_loss = (diff_full * diff_full).mean()
                loss_step = loss_step + full_state_weight * full_state_loss

            total_loss = total_loss + step_weight * loss_step
            n_terms += 1

            if tbptt_steps is not None and tbptt_steps > 0 and ((j + 1) % tbptt_steps == 0):
                # Stop gradients from spanning the entire horizon when long
                # sequences would be too memory-heavy.
                y = y.detach()

    if n_terms == 0:
        raise RuntimeError("No rollout segments were formed (check stride / horizon).")

    return total_loss / float(n_terms)


def multi_step_rollout_loss(
    t,
    meas,
    smoothed_states,
    integrator,
    meas_idx,
    horizon_steps,
    stride,
    pseudo_idx=None,
    pseudo_weight: float = 0.0,
    drive_seq=None,
    intermediate_weight: float = 1.0,
    full_state_weight: float = 0.0,
    state_scales=None,
    point_weights=None,
    tbptt_steps: Optional[int] = None,
):
    """
    Batched GPU rollout loss for uniform and nonuniform grids.

    This is the standard training loss path: it starts from smoother states,
    rolls the learned ODE forward, and compares the rollout to measurements
    plus optional smoother-derived hidden/full-state targets.

    Plain MSE rollout loss with optional state scaling. It follows the same
    multi-step residual structure as Eq. 12-14.

    Uniform grids use one shared dt. Nonuniform/adaptive grids stay batched
    when the integrator exposes f_full or f_full_with_pin/f_full_with_drive.

    point_weights : (T,) optional tensor of per-timestep confidence weights
        (e.g. 1/smoother_variance). High-uncertainty starts are filtered out
        (bottom 25% discarded) and pseudo loss is scaled by weight at each
        target timestep.
    """
    print(f"[rollout] start: T={t.shape[0]} horizon={horizon_steps} stride={stride}")
    T = t.shape[0]
    clip_value = _rollout_clip_value()
    if T < 2:
        raise ValueError("t must have length >= 2")

    if meas.dim() == 1:
        meas = meas.unsqueeze(-1)

    n_meas = meas.shape[1]
    if len(meas_idx) != n_meas:
        raise ValueError(
            f"len(meas_idx) = {len(meas_idx)} but meas has {n_meas} columns"
        )

    use_drive = drive_seq is not None
    if use_drive:
        if drive_seq.shape[0] != T:
            raise ValueError("drive_seq must have shape (T, d_drive)")
        if not hasattr(integrator, "rk4_step_with_drive"):
            raise AttributeError(
                "drive_seq provided but integrator has no rk4_step_with_drive"
            )

    # Uniform grids use one shared dt. Nonuniform grids use one dt per row but
    # still stay batched when the integrator exposes the hybrid RHS directly.
    dt_tensor = t[1:] - t[:-1]
    if dt_tensor.numel() == 0:
        raise ValueError("t must have length >= 2")

    dt_uniform, dt_step, dt_min_f, dt_max_f, dt_mean_f, rounded_uniform = _near_uniform_rollout_dt(dt_tensor)
    if not dt_uniform:
        can_batch_nonuniform = (
            (not use_drive and hasattr(integrator, "f_full"))
            or (use_drive and (hasattr(integrator, "f_full_with_pin") or hasattr(integrator, "f_full_with_drive")))
        )
        if can_batch_nonuniform:
            print(
                "[rollout] non-uniform dt detected; using batched per-row dt fast path "
                f"(dt_min={dt_min_f:.6g}, "
                f"dt_max={dt_max_f:.6g}, "
                f"dt_mean={dt_mean_f:.6g})"
            )
        else:
            print(
                "[rollout] non-uniform dt detected but integrator has no batched RHS; "
                "using compatibility fallback "
                f"(dt_min={dt_min_f:.6g}, "
                f"dt_max={dt_max_f:.6g}, "
                f"dt_mean={dt_mean_f:.6g})"
            )
            return _multi_step_rollout_loss_slow(
                t=t,
                meas=meas,
                smoothed_states=smoothed_states,
                integrator=integrator,
                meas_idx=meas_idx,
                horizon_steps=horizon_steps,
                stride=stride,
                pseudo_idx=pseudo_idx,
                pseudo_weight=pseudo_weight,
                drive_seq=drive_seq,
                intermediate_weight=intermediate_weight,
                full_state_weight=full_state_weight,
                state_scales=state_scales,
                point_weights=point_weights,
                tbptt_steps=tbptt_steps,
            )
    if rounded_uniform:
        print(
            "[rollout] near-uniform dt detected; using rounded uniform fast path "
            f"(dt_step={dt_step:.6g}, dt_min={dt_min_f:.6g}, "
            f"dt_max={dt_max_f:.6g}, dt_mean={dt_mean_f:.6g})"
        )

    if dt_uniform:
        print(f"[rollout] dt_step={dt_step:.6g} use_drive={int(use_drive)}")
    else:
        print(f"[rollout] per-row dt fast path use_drive={int(use_drive)}")
    print(f"[rollout] clip_value={clip_value:.6g}")

    # One row in the batch per rollout start.
    device = meas.device
    start_idxs = torch.arange(0, T - 1, stride, device=device)
    if start_idxs.numel() == 0:
        raise RuntimeError("No rollout segments were formed (check stride / horizon).")

    # Gate start indices to high-confidence timesteps (drop bottom 25% by weight).
    # This is only about where rollouts start; valid targets are still masked by horizon.
    pw = None
    if point_weights is not None:
        pw = point_weights.to(device=device, dtype=t.dtype)
        w_starts = pw[start_idxs]
        thresh = w_starts.quantile(0.25)
        start_idxs = start_idxs[w_starts >= thresh]
        if start_idxs.numel() == 0:
            raise RuntimeError("All rollout starts filtered by point_weights.")
        print(f"[rollout] confidence-gated batch size={start_idxs.numel()}")
    else:
        print(f"[rollout] batch size={start_idxs.numel()}")

    # Starts near the end of the sequence have shorter valid horizons.
    max_h = torch.minimum(
        torch.full_like(start_idxs, int(horizon_steps)),
        (T - 1) - start_idxs,
    )

    # Tensor versions of index lists avoid Python indexing inside the rollout loop.
    meas_list = torch.tensor(list(meas_idx), device=device, dtype=torch.long)
    use_pseudo = pseudo_idx is not None and pseudo_weight > 0.0
    pseudo_list = torch.tensor(list(pseudo_idx), device=device, dtype=torch.long) if use_pseudo else None
    use_full_state = full_state_weight > 0.0

    # Build normalization scales for measurement and pseudo losses. These are
    # multipliers that put large/small physical units on a comparable scale.
    if state_scales is not None:
        # Scale channels to comparable magnitude before squaring residuals.
        _ss = torch.tensor(state_scales, device=device, dtype=t.dtype)
        meas_scales = _ss[meas_list]  # (n_meas,)
        pseudo_scales = _ss[pseudo_list] if use_pseudo else None
        full_scales = _ss  # (n_x,)
        print(f'[rollout] using state_scales for normalization')
    else:
        meas_scales = None
        pseudo_scales = None
        full_scales = None

    # Start each rollout from the smoother state at that time index. From here
    # onward the learned ODE must predict forward without correction.
    y = _sanitize_rollout_state(smoothed_states[start_idxs], integrator, clip_value)  # (B, n_x)

    total_loss = torch.zeros((), device=device)
    n_terms = torch.zeros((), device=device)

    log_every = max(1, int(horizon_steps) // 5)
    for j in range(int(horizon_steps)):
        if j % log_every == 0:
            print(f"[rollout] step {j+1}/{int(horizon_steps)}")
        target_idx = start_idxs + j + 1  # (B,)
        valid = target_idx < T
        if not torch.any(valid):
            break

        # Integrate one step for all batch items. Uniform grids use the shared
        # dt_step; nonuniform grids gather the correct interval for each row.
        if use_drive:
            drive_j = drive_seq[torch.clamp(start_idxs + j, max=T - 1)]  # (B, d_drive)
            if dt_uniform:
                y = integrator.rk4_step_with_drive(y, dt_step, drive_j)
            else:
                step_dt = dt_tensor.index_select(0, torch.clamp(start_idxs + j, max=T - 2))
                y = _rk4_step_batched_dt_with_drive(y, step_dt, drive_j, integrator, clip_value)
        else:
            if dt_uniform:
                y = integrator.rk4_step(y, dt_step)
            else:
                step_dt = dt_tensor.index_select(0, torch.clamp(start_idxs + j, max=T - 2))
                y = _rk4_step_batched_dt(y, step_dt, integrator, clip_value)
        y = _sanitize_rollout_state(y, integrator, clip_value)

        # Clamp indices only to keep tensor gathers valid; the valid mask below
        # prevents those padded rows from affecting the loss.
        target_idx_safe = torch.clamp(target_idx, max=T - 1)

        # Measurement residual, with optional scale normalization.
        y_meas_pred = _sanitize_rollout_values(y[:, meas_list], clip_value)
        y_meas_true = _sanitize_rollout_values(meas[target_idx_safe], clip_value)
        diff = _sanitize_rollout_values(y_meas_pred - y_meas_true, clip_value)
        if meas_scales is not None:
            diff = diff * meas_scales.unsqueeze(0)  # normalize to O(1)
        loss_step = (diff * diff).mean(dim=1)

        # Pseudo-state residual on hidden/learned states. This is what teaches
        # a learned derivative when the corresponding state is not measured.
        if use_pseudo:
            y_pseudo_pred = _sanitize_rollout_values(y[:, pseudo_list], clip_value)
            y_pseudo_true = _sanitize_rollout_values(smoothed_states[target_idx_safe][:, pseudo_list], clip_value)
            diff_p = _sanitize_rollout_values(y_pseudo_pred - y_pseudo_true, clip_value)
            if pseudo_scales is not None:
                diff_p = diff_p * pseudo_scales.unsqueeze(0)
            pseudo_loss = (diff_p * diff_p).mean(dim=1)
            # Scale pseudo loss by per-timestep confidence if available.
            if pw is not None:
                pseudo_loss = pseudo_loss * pw[target_idx_safe]
            loss_step = loss_step + pseudo_weight * pseudo_loss

        # Full-state residual against the smoother trajectory. Useful when
        # measured channels alone do not constrain the hidden dynamics enough.
        if use_full_state:
            y_full_true = _sanitize_rollout_values(smoothed_states[target_idx_safe], clip_value)
            diff_full = _sanitize_rollout_values(y - y_full_true, clip_value)
            if full_scales is not None:
                diff_full = diff_full * full_scales.unsqueeze(0)
            full_state_loss = (diff_full * diff_full).mean(dim=1)
            loss_step = loss_step + full_state_weight * full_state_loss

        # Every valid horizon contributes; intermediate horizons can be downweighted.
        is_final = (j == (max_h - 1))
        step_weight = torch.where(is_final, torch.ones_like(loss_step), torch.full_like(loss_step, intermediate_weight))

        # Apply valid mask so short rollouts near the end do not read padded targets.
        valid_f = valid.to(loss_step.dtype)
        total_loss = total_loss + torch.sum(step_weight * loss_step * valid_f)
        n_terms = n_terms + torch.sum(valid_f)

        if tbptt_steps is not None and tbptt_steps > 0 and ((j + 1) % tbptt_steps == 0):
            # Truncate the computation graph but keep the numerical rollout state.
            y = y.detach()

    if n_terms.item() == 0:
        raise RuntimeError("No rollout segments were formed (check stride / horizon).")

    return total_loss / n_terms

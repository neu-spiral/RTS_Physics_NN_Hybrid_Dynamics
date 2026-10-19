"""Filter, dataset, and model-comparison plots."""

from .common import *


def plot_filter_smoother_uncertainty(t, ms, Ps, ms_s, Ps_s, true_states=None, observations_by_state=None, state_indices=None, state_labels=None, title='CKF+RTS', show=True, save_dir='results', save_prefix='experiment', drive_seq=None, epoch=None, epoch_timestamp=None):
    """
    t: (T,) tensor
    ms, ms_s: (T, n)
    Ps, Ps_s: (T, n, n)
    true_states: (T, n) or None
    observations_by_state: {state_index: (T,) Tensor} or None
    drive_seq: optional (T,) or (T, d) tensor (e.g., Pin)
    """
    print(f'[viz] plot_filter_smoother_uncertainty: save_dir={save_dir}, prefix={save_prefix}, epoch={epoch}')
    plt.close('all')
    reset_plot_params()
    if state_indices is None:
        n = ms.shape[1]
        state_indices = list(range(n))
    if state_labels is None:
        state_labels = [f's{i}' for i in state_indices]
    t_np = t.detach().cpu().numpy()
    T = t_np.shape[0]
    drive_np = None
    if drive_seq is not None:
        print(f'[viz] drive_seq provided: shape={drive_seq.shape}, type={type(drive_seq)}')
        d = drive_seq
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        drive_np = d.detach().cpu().numpy()
        print(f'[viz] drive_np: shape={drive_np.shape}, min={drive_np.min():.3f}, max={drive_np.max():.3f}')
    n_state_plots = len(state_indices)
    has_drive = drive_np is not None
    (n_rows_states, n_cols_states) = _state_grid_shape(n_state_plots)
    total_rows = n_rows_states + (1 if has_drive else 0)
    fig_width = 6.6 * n_cols_states
    fig_height = 5.2 * total_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    height_ratios = [1.0] * n_rows_states + ([0.75] if has_drive else [])
    gs = fig.add_gridspec(total_rows, n_cols_states, height_ratios=height_ratios)
    state_axes = []
    ax_drive = None
    idx_state = 0
    for r in range(n_rows_states):
        for c in range(n_cols_states):
            if idx_state >= n_state_plots:
                break
            share_ax = state_axes[0] if state_axes else None
            ax = fig.add_subplot(gs[r, c], sharex=share_ax)
            state_axes.append(ax)
            idx_state += 1
        if idx_state >= n_state_plots:
            continue
    if has_drive:
        share_ax = state_axes[0] if state_axes else None
        ax_drive = fig.add_subplot(gs[-1, :], sharex=share_ax)
    axes_all = state_axes + ([ax_drive] if ax_drive is not None else [])
    ms_np = ms.detach().cpu().numpy()
    ms_s_np = ms_s.detach().cpu().numpy()
    Ps_np = Ps.detach().cpu().numpy()
    Ps_s_np = Ps_s.detach().cpu().numpy()
    true_np = None
    if true_states is not None:
        true_np = true_states.detach().cpu().numpy()
    obs_dict_np = {}
    if observations_by_state is not None:
        for (k, v) in observations_by_state.items():
            obs_dict_np[int(k)] = v.detach().cpu().numpy()
    try:
        marker_target = max(1, int(os.getenv('ODE_FS_MARKER_TARGET', '8')))
    except Exception:
        marker_target = 8
    marker_every = max(1, T // marker_target)
    marker_every_meas = max(1, marker_every)
    try:
        marker_size_filter = float(os.getenv('ODE_FS_FILTER_MARKER_SIZE', '3.2'))
        marker_size_smoother = float(os.getenv('ODE_FS_SMOOTHER_MARKER_SIZE', '3.4'))
        marker_size_true = float(os.getenv('ODE_FS_TRUE_MARKER_SIZE', '3.2'))
        marker_size_drive = float(os.getenv('ODE_FS_DRIVE_MARKER_SIZE', '3.2'))
        marker_size_meas = float(os.getenv('ODE_FS_MEAS_MARKER_SIZE', '2.4'))
    except Exception:
        marker_size_filter = 3.2
        marker_size_smoother = 3.4
        marker_size_true = 3.2
        marker_size_drive = 3.2
        marker_size_meas = 2.4
    first_state_idx = state_indices[0] if state_indices else None
    for (ax, idx, label) in zip(state_axes, state_indices, state_labels):
        m_f = ms_np[:, idx]
        m_s = ms_s_np[:, idx]
        std_f = np.sqrt(np.clip(Ps_np[:, idx, idx], 0.0, None))
        std_s = np.sqrt(np.clip(Ps_s_np[:, idx, idx], 0.0, None))
        ax.plot(t_np, m_f, '-', lw=3.2, color=COLORS['filter'], label='filter' if idx == first_state_idx else None, alpha=0.7, marker='s', markevery=marker_every, markersize=marker_size_filter, markeredgecolor='black', markeredgewidth=0.4)
        ax.fill_between(t_np, m_f - 2 * std_f, m_f + 2 * std_f, color=COLORS['uncertainty'], alpha=0.15, label='filter ±2σ' if idx == first_state_idx else None)
        ax.fill_between(t_np, m_s - 2 * std_s, m_s + 2 * std_s, color=COLORS['smoother'], alpha=0.15, label='smoother ±2σ' if idx == first_state_idx else None)
        ax.plot(t_np, m_s, '-', lw=3.8, color=COLORS['smoother'], label='smoother' if idx == first_state_idx else None, marker='o', markevery=marker_every, markersize=marker_size_smoother, markeredgecolor='black', markeredgewidth=0.4)
        if true_np is not None:
            ax.plot(t_np, true_np[:, idx], '--', lw=3.2, color=COLORS['true'], label='true' if idx == first_state_idx else None, marker='^', markevery=marker_every, markersize=marker_size_true, markeredgecolor='black', markeredgewidth=0.4)
        if idx in obs_dict_np:
            ax.plot(t_np, obs_dict_np[idx], '-', lw=1.4, alpha=0.65, color=COLORS['meas'], marker='o', markevery=marker_every_meas, markersize=marker_size_meas, markeredgewidth=0, label='measurement' if idx == first_state_idx else None)
        ax.set_ylabel(label, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.9)
        if label.lower() == 'x5':
            ax.set_ylim(0.33, 0.39)
        if label.lower() == 'x7':
            ax.set_ylim(0.0, 0.18)
        if label.lower() == 'x4':
            ax.set_ylim(0.8, 0.9)
    if drive_np is not None and ax_drive is not None:
        print(f'[viz] Plotting Pin on bottom axis, drive_np shape: {drive_np.shape}')
        ax_drive.plot(t_np, drive_np[:, 0], '-', lw=3.2, color=COLORS['drive'], label='Pin', marker='d', markevery=marker_every, markersize=marker_size_drive, markeredgecolor='black', markeredgewidth=0.4)
        ax_drive.set_ylabel('Pin', fontweight='bold')
        ax_drive.grid(True, alpha=0.3, linewidth=0.9)
        print(f'[viz] Pin plotting complete')
    if axes_all:
        axes_all[-1].set_xlabel('Time', fontweight='bold')
    fig.suptitle(title, fontweight='bold', fontsize=22)
    print('[viz] rollouts: figure assembly complete, saving next')
    legend_handles = []
    legend_labels = []
    for ax in axes_all:
        if ax is None:
            continue
        (handles, labels) = ax.get_legend_handles_labels()
        for (handle, lbl) in zip(handles, labels):
            if lbl and lbl not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(lbl)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(6, len(legend_labels)), fontsize=16, framealpha=0.9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.12 if legend_handles else 0.045, 1, 0.97])
    arrays = {'t': t_np, 'ms': ms_np, 'ms_s': ms_s_np, 'smoothed_states': ms_s_np, 'Ps': Ps_np, 'Ps_s': Ps_s_np, 'state_labels': np.array(state_labels)}
    if true_np is not None:
        arrays['true_states'] = true_np
    for (k, v) in obs_dict_np.items():
        arrays[f'obs_state_{k}'] = v
    if epoch_timestamp is None and epoch is not None:
        epoch_timestamp = _timestamp_tag()
    _save_fig_and_arrays(base_dir=save_dir, prefix=save_prefix, tag=title.replace(' ', '_'), fig=fig, arrays_dict=arrays, epoch=epoch, epoch_timestamp=epoch_timestamp)
    _show_if_interactive(show)
    plt.close(fig)


def _continuous_rollout(integrator, y0, t_np):
    """Roll an integrator forward over the (possibly non-uniform) time grid t_np
    starting from a single initial condition y0, one rk4_step per interval.

    Returns an (T, n_state) numpy array; rows after a non-finite step are filled
    with NaN so the caller can see exactly where (and if) the model diverged.
    """
    T = int(t_np.shape[0])
    n = int(y0.shape[-1])
    out = np.full((T, n), np.nan, dtype=np.float64)
    y = y0.detach().clone().reshape(-1)
    out[0] = y.cpu().numpy()
    with torch.no_grad():
        for k in range(1, T):
            dt = float(t_np[k] - t_np[k - 1])
            try:
                y = integrator.rk4_step(y, dt)
            except Exception as exc:
                print(f'[viz] model-vs-true rollout step {k} failed: {exc}')
                break
            if not bool(torch.isfinite(y).all().item()):
                print(f'[viz] model-vs-true rollout went non-finite at step {k}/{T}')
                break
            out[k] = y.cpu().numpy()
    return out

def plot_model_vs_true_ode(t, integrator, physics_integrator, state_indices, state_labels, smoothed_states=None, true_states=None, init_from='smoother', title='Learned ODE vs true ODE', show=True, save_dir='results', save_prefix='model', epoch=None, epoch_timestamp=None):
    """Model-vs-model comparison: one continuous rollout of the LEARNED ODE and
    one of the TRUE physics ODE, both started from the *same* single initial
    condition and advanced across the entire time grid.

    Unlike ``plot_rollouts`` (many short re-initialized segments), this isolates
    pure model error: how far the learned vector field drifts from the true
    vector field when both are integrated freely from t=0.  Per-state RMSE of the
    learned rollout against the true-ODE rollout (and, if available, the true
    states) is annotated on each panel.
    """
    reset_plot_params()
    t_t = t if torch.is_tensor(t) else torch.as_tensor(t)
    t_np = t_t.detach().cpu().numpy().reshape(-1)
    T = int(t_np.shape[0])
    state_indices = list(state_indices)
    n_plot = len(state_indices)

    def _row0(x):
        xt = x if torch.is_tensor(x) else torch.as_tensor(x)
        return xt.detach().to(torch.float32).reshape(T, -1)[0]
    y0 = None
    if init_from == 'true' and true_states is not None:
        y0 = _row0(true_states)
    elif smoothed_states is not None:
        y0 = _row0(smoothed_states)
    elif true_states is not None:
        y0 = _row0(true_states)
    if y0 is None:
        print('[viz] plot_model_vs_true_ode: no initial condition available; skipping.')
        return
    learned_roll = _continuous_rollout(integrator, y0, t_np)
    true_roll = _continuous_rollout(physics_integrator, y0, t_np)
    sm_np = None
    if smoothed_states is not None:
        st = smoothed_states if torch.is_tensor(smoothed_states) else torch.as_tensor(smoothed_states)
        sm_np = st.detach().cpu().numpy().reshape(T, -1)
    tr_np = None
    if true_states is not None:
        tt = true_states if torch.is_tensor(true_states) else torch.as_tensor(true_states)
        tr_np = tt.detach().cpu().numpy().reshape(T, -1)

    def _rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            return float('nan')
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
    (n_rows, n_cols) = _state_grid_shape(n_plot)
    (fig, axes) = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    rmse_vs_true = {}
    rmse_vs_truth = {}
    for (ax_i, s) in enumerate(state_indices):
        ax = axes_flat[ax_i]
        lbl = state_labels[s] if s < len(state_labels) else f'x{s}'
        if tr_np is not None:
            ax.plot(t_np, tr_np[:, s], color=COLORS['true'], ls='--', lw=2.0, label='true states', zorder=2)
        if sm_np is not None:
            ax.plot(t_np, sm_np[:, s], color=COLORS['smoother'], lw=1.6, alpha=0.7, label='smoothed states', zorder=2)
        ax.plot(t_np, true_roll[:, s], color=COLORS['rollout'], lw=2.4, label='true ODE rollout', zorder=3)
        ax.plot(t_np, learned_roll[:, s], color='blue', lw=2.0, label='learned ODE rollout', zorder=4)
        r_t = _rmse(learned_roll[:, s], true_roll[:, s])
        rmse_vs_true[lbl] = r_t
        ann = f'RMSE vs true ODE = {r_t:.3g}'
        if tr_np is not None:
            r_g = _rmse(learned_roll[:, s], tr_np[:, s])
            rmse_vs_truth[lbl] = r_g
            ann += f'\nRMSE vs true states = {r_g:.3g}'
        ax.set_title(lbl)
        ax.set_xlabel('Time')
        ax.text(0.02, 0.97, ann, transform=ax.transAxes, va='top', ha='left', fontsize=12, bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.8))
    for ax_i in range(n_plot, len(axes_flat)):
        axes_flat[ax_i].axis('off')
    (handles, labels) = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.04))
    mean_rmse = np.nanmean([v for v in rmse_vs_true.values()]) if rmse_vs_true else float('nan')
    fig.suptitle(f'{title}  (mean state RMSE vs true ODE = {mean_rmse:.3g})', fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    print('[viz] model-vs-true-ODE per-state RMSE (learned vs true ODE rollout):')
    for (lbl, v) in rmse_vs_true.items():
        extra = f"  | vs true states {rmse_vs_truth.get(lbl, float('nan')):.3g}" if rmse_vs_truth else ''
        print(f'        {lbl:>6}: {v:.4g}{extra}')
    _save_fig_and_arrays(base_dir=save_dir, prefix=save_prefix, tag=title, fig=fig, arrays_dict={'t': t_np, 'learned_rollout': learned_roll, 'true_ode_rollout': true_roll, 'smoothed_states': sm_np if sm_np is not None else np.array([]), 'true_states': tr_np if tr_np is not None else np.array([]), 'state_indices': np.asarray(state_indices), 'rmse_vs_true_ode': np.asarray([rmse_vs_true[state_labels[s] if s < len(state_labels) else f'x{s}'] for s in state_indices])}, epoch=epoch, epoch_timestamp=epoch_timestamp)
    _show_if_interactive(show)
    plt.close(fig)
    return {'rmse_vs_true_ode': rmse_vs_true, 'rmse_vs_true_states': rmse_vs_truth}


def plot_dataset_ode(t, true_states, meas=None, meas_idx=None, state_labels=None, title='Dataset (full ODE)', show=True, save_dir='results', save_prefix='dataset', drive_seq=None, epoch_timestamp=None):
    """
    Visualize the raw dataset generated by the full ODE:

      - true_states: states from the ODE solver
      - meas: noisy measurements (if any)
      - drive_seq: exogenous input like Pin(t) (if any)

    One subplot per state, plus an extra subplot for drive_seq if provided.
    """
    plt.close('all')
    reset_plot_params()
    t_np = t.detach().cpu().numpy()
    states = true_states
    if states.dim() == 1:
        states = states.unsqueeze(-1)
    states_np = states.detach().cpu().numpy()
    (T, n_states) = states_np.shape
    if state_labels is None:
        state_labels = [f's{i}' for i in range(n_states)]
    meas_np = None
    if meas is not None:
        meas_t = meas
        if meas_t.dim() == 1:
            meas_t = meas_t.unsqueeze(-1)
        meas_np = meas_t.detach().cpu().numpy()
        m = meas_np.shape[1]
        if meas_idx is None:
            meas_idx = list(range(m))
    if meas_idx is None:
        meas_idx = []
    drive_np = None
    if drive_seq is not None:
        d = drive_seq
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        drive_np = d.detach().cpu().numpy()
    has_drive = drive_np is not None
    (n_rows_states, n_cols_states) = _state_grid_shape(n_states)
    total_rows = n_rows_states + (1 if has_drive else 0)
    fig_width = 5.8 * n_cols_states
    fig_height = 4.6 * total_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    height_ratios = [1.0] * n_rows_states + ([0.75] if has_drive else [])
    gs = fig.add_gridspec(total_rows, n_cols_states, height_ratios=height_ratios)
    state_axes = []
    ax_drive = None
    idx_state = 0
    for r in range(n_rows_states):
        for c in range(n_cols_states):
            if idx_state >= n_states:
                break
            share_ax = state_axes[0] if state_axes else None
            ax = fig.add_subplot(gs[r, c], sharex=share_ax)
            state_axes.append(ax)
            idx_state += 1
        if idx_state >= n_states:
            continue
    if has_drive:
        share_ax = state_axes[0] if state_axes else None
        ax_drive = fig.add_subplot(gs[-1, :], sharex=share_ax)
    axes_all = state_axes + ([ax_drive] if ax_drive is not None else [])
    obs_map = {}
    if meas_np is not None:
        for (j, idx) in enumerate(meas_idx):
            obs_map[int(idx)] = meas_np[:, j]
    marker_every = 100
    legend_handles = []
    legend_labels = []
    for (ax, idx, label) in zip(state_axes, range(n_states), state_labels):
        ax.plot(t_np, states_np[:, idx], '--', lw=3.2, color=COLORS['true'], label='true' if idx == 0 else None, marker='^', markevery=marker_every, markersize=9, markeredgecolor='black', markeredgewidth=0.6)
        if idx in obs_map:
            ax.plot(t_np, obs_map[idx], 'o', ms=7, alpha=0.6, color=COLORS['meas'], markeredgewidth=0, label='measurement' if idx == 0 else None)
        ax.set_ylabel(label, fontweight='bold')
        ax.grid(alpha=0.3, linewidth=0.9)
        (h, l) = ax.get_legend_handles_labels()
        for (handle, lbl) in zip(h, l):
            if lbl not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(lbl)
    if drive_np is not None and ax_drive is not None:
        ax_drive.plot(t_np, drive_np[:, 0], '-', lw=3.2, color=COLORS['drive'], label='drive (Pin)', marker='d', markevery=marker_every, markersize=9, markeredgecolor='black', markeredgewidth=0.6)
        ax_drive.set_ylabel('Pin', fontweight='bold')
        ax_drive.grid(alpha=0.3, linewidth=0.9)
        (h, l) = ax_drive.get_legend_handles_labels()
        for (handle, lbl) in zip(h, l):
            if lbl not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(lbl)
    if axes_all:
        axes_all[-1].set_xlabel('Time', fontweight='bold')
    fig.suptitle(title, fontweight='bold', fontsize=22)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=min(len(legend_labels), 4), framealpha=0.9, fontsize=14)
    fig.tight_layout(rect=[0, 0.08 if legend_handles else 0.0, 1, 0.98])
    arrays = {'t': t_np, 'true_states': states_np}
    if meas_np is not None:
        arrays['meas'] = meas_np
        arrays['meas_idx'] = np.asarray(meas_idx, dtype=np.int64)
    if drive_np is not None:
        arrays['drive_seq'] = drive_np
    _save_fig_and_arrays(base_dir=save_dir, prefix=save_prefix, tag=title.replace(' ', '_'), fig=fig, arrays_dict=arrays, epoch_timestamp=epoch_timestamp)
    _show_if_interactive(show)
    plt.close(fig)

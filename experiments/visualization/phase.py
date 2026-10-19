"""Phase portraits, vector fields, and phase-error diagnostics."""

from __future__ import annotations

from .common import *


def _compute_vector_field(integrator, true_states, state_indices, n_grid=20):
    """
    Compute the learned ODE vector field on a 2D grid in phase space.

    Uses the integrator's f_full to evaluate dx/dt at each grid point,
    keeping non-plotted states fixed at their mean values from the data.

    Returns (X_grid, Y_grid, dX, dY) suitable for plt.streamplot.
    """
    (idx_x, idx_y) = state_indices
    device = true_states.device
    n_states = true_states.shape[1]
    true_np = true_states.detach().cpu().numpy()
    x_vals = true_np[:, idx_x]
    y_vals = true_np[:, idx_y]
    x_pad = 0.1 * (x_vals.max() - x_vals.min()) or 0.5
    y_pad = 0.1 * (y_vals.max() - y_vals.min()) or 0.5
    (x_min, x_max) = (x_vals.min() - x_pad, x_vals.max() + x_pad)
    (y_min, y_max) = (y_vals.min() - y_pad, y_vals.max() + y_pad)
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    (X_grid, Y_grid) = np.meshgrid(xs, ys)
    mean_state = true_states.mean(dim=0)
    flat_x = X_grid.ravel()
    flat_y = Y_grid.ravel()
    n_pts = len(flat_x)
    batch = mean_state.unsqueeze(0).expand(n_pts, -1).clone()
    batch[:, idx_x] = torch.tensor(flat_x, dtype=batch.dtype, device=device)
    batch[:, idx_y] = torch.tensor(flat_y, dtype=batch.dtype, device=device)
    with torch.no_grad():
        try:
            dydt = integrator.f_full(batch)
        except Exception:
            return (None, None, None, None)
    dydt_np = dydt.detach().cpu().numpy()
    dX = dydt_np[:, idx_x].reshape(X_grid.shape)
    dY = dydt_np[:, idx_y].reshape(Y_grid.shape)
    return (X_grid, Y_grid, dX, dY)

def plot_phase_portrait(t, true_states, integrator, estimated_states=None, initial_conditions=None, state_indices=(0, 1), state_labels=('x', 'v'), n_trajectories=2, integration_time=None, title='Phase Portrait: Learned vs True ODE', show=True, save_dir='results', save_prefix='experiment', epoch=None, epoch_timestamp=None):
    """
    Plot phase portraits comparing learned ODE vs true ODE trajectories.

    Produces TWO separate figures:
      1. Phase portrait with vector field background (streamplot) + trajectories
      2. Time-series comparison (state_i and state_j vs time)

    Args:
        t: (T,) time tensor from data generation
        true_states: (T, n) true state trajectories
        estimated_states: optional (T, n) CKF/RTS or smoother state estimates for overlay in time-series
        integrator: learned ODE integrator with rk4_step / f_full methods
        initial_conditions: list of (n,) tensors for starting points, or None to auto-select
        state_indices: tuple of 2 indices for phase space dimensions
        state_labels: either full state label list/tuple, or a 2-label tuple for the selected phase dims
        n_trajectories: number of trajectories to plot if initial_conditions is None
        integration_time: time to integrate for each trajectory (default: use t range)
        title: plot title
        show: whether to display
        save_dir: directory to save
        save_prefix: filename prefix
        epoch: epoch number (for folder naming)
        epoch_timestamp: timestamp string (reused within same epoch)
    """
    print(f'[viz] plot_phase_portrait: save_dir={save_dir}, prefix={save_prefix}, epoch={epoch}')
    plt.close('all')
    reset_plot_params()
    t_np = t.detach().cpu().numpy()
    true_np = true_states.detach().cpu().numpy()
    estimated_np = None
    if estimated_states is not None:
        estimated_np = estimated_states.detach().cpu().numpy()
    device = true_states.device
    T = t_np.shape[0]
    n_states = true_states.shape[1]
    (idx_x, idx_y) = state_indices
    labels_seq = list(state_labels) if state_labels is not None else []
    if len(labels_seq) > max(idx_x, idx_y):
        phase_label_x = str(labels_seq[idx_x])
        phase_label_y = str(labels_seq[idx_y])
    elif len(labels_seq) >= 2:
        phase_label_x = str(labels_seq[0])
        phase_label_y = str(labels_seq[1])
    else:
        phase_label_x = f's{idx_x}'
        phase_label_y = f's{idx_y}'

    def _label_for_idx(idx: int) -> str:
        if len(labels_seq) > idx:
            return str(labels_seq[idx])
        if idx == idx_x:
            return phase_label_x
        if idx == idx_y:
            return phase_label_y
        return f's{idx}'
    if integration_time is None:
        integration_time = t_np[-1] - t_np[0]
    dt = (t_np[-1] - t_np[0]) / (T - 1) if T > 1 else 0.01
    n_steps = min(T - 1, 40)
    progress_every = _viz_progress_every(500)
    if initial_conditions is None:
        step = max(1, T // n_trajectories)
        ic_indices = list(range(0, T, step))[:n_trajectories]
        initial_conditions = [true_states[i] for i in ic_indices]
    else:
        ic_indices = [0] * len(initial_conditions)
    learned_trajectories = []
    all_errors = []
    print(f'[viz] Integrating learned ODE: {integration_time:.2f}s, {n_steps} steps')
    y0_full = true_states[0].clone().unsqueeze(0)
    full_learned_path = [y0_full.squeeze(0).detach().cpu().numpy()]
    y_curr = y0_full
    t_full_start = time.perf_counter()
    for step_idx in range(n_steps):
        if step_idx == 0 or (step_idx + 1) % progress_every == 0 or step_idx + 1 == n_steps:
            print(f'[viz] phase full trajectory progress: step {step_idx + 1}/{n_steps}')
        y_curr = integrator.rk4_step(y_curr, dt)
        full_learned_path.append(y_curr.squeeze(0).detach().cpu().numpy())
    full_learned_path = np.array(full_learned_path)
    print(f'[viz] phase full trajectory complete in {time.perf_counter() - t_full_start:.2f}s')
    learned_trajectories.append(full_learned_path)
    if len(full_learned_path) > 0:
        errors_full = compute_phase_portrait_error(true_np, full_learned_path, state_indices)
        errors_full['trajectory_idx'] = -1
        all_errors.append(errors_full)
    for (i, ic) in enumerate(initial_conditions):
        print(f'[viz] phase sampled trajectory {i + 1}/{len(initial_conditions)}: start_idx={ic_indices[i]}')
        y0 = ic.clone().unsqueeze(0)
        learned_path = [y0.squeeze(0).detach().cpu().numpy()]
        y_curr = y0
        t_ic_start = time.perf_counter()
        for step_idx in range(n_steps):
            if step_idx == 0 or (step_idx + 1) % progress_every == 0 or step_idx + 1 == n_steps:
                print(f'[viz] phase sampled progress: traj {i + 1}/{len(initial_conditions)} step {step_idx + 1}/{n_steps}')
            y_curr = integrator.rk4_step(y_curr, dt)
            learned_path.append(y_curr.squeeze(0).detach().cpu().numpy())
        learned_path = np.array(learned_path)
        print(f'[viz] phase sampled trajectory {i + 1}/{len(initial_conditions)} complete in {time.perf_counter() - t_ic_start:.2f}s')
        learned_trajectories.append(learned_path)
        ic_idx = ic_indices[i]
        true_segment = true_np[ic_idx:ic_idx + len(learned_path)]
        if len(true_segment) > 0:
            errors = compute_phase_portrait_error(true_segment, learned_path, state_indices)
            errors['trajectory_idx'] = i
            all_errors.append(errors)
    print('[viz] phase portrait: computing vector field')
    (X_grid, Y_grid, dX, dY) = _compute_vector_field(integrator, true_states, state_indices, n_grid=22)
    avg_errors = None
    if all_errors:
        avg_errors = {'mean_phase_error': np.mean([e['mean_phase_error'] for e in all_errors]), 'hausdorff_phase_error': np.mean([e['hausdorff_phase_error'] for e in all_errors])}
        log_dir = Path(save_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{save_prefix}_phase_errors.csv'
        import csv
        file_exists = log_file.exists()
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'timestamp'] + list(avg_errors.keys()))
            if not file_exists:
                writer.writeheader()
            row = {'epoch': epoch if epoch is not None else 0, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            row.update(avg_errors)
            writer.writerow(row)
    if epoch_timestamp is None and epoch is not None:
        epoch_timestamp = _timestamp_tag()
    traj_cmap = plt.cm.tab10
    marker_every = max(1, T // 20)
    (fig_phase, ax) = plt.subplots(figsize=(7, 7))
    if X_grid is not None:
        speed = np.sqrt(dX ** 2 + dY ** 2)
        lw_stream = 0.4 + 1.6 * speed / (speed.max() + 1e-12)
        sp = ax.streamplot(X_grid, Y_grid, dX, dY, density=1.3, arrowsize=1.0, linewidth=lw_stream, color=speed, cmap='Blues', zorder=0)
        sp.lines.set_alpha(0.35)
    ax.plot(true_np[:, idx_x], true_np[:, idx_y], ':', lw=2.5, color=COLORS['true'], alpha=0.85, label='True ODE', zorder=3)
    ax.plot(true_np[::marker_every, idx_x], true_np[::marker_every, idx_y], 'o', ms=6, color=COLORS['true'], alpha=0.8, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
    ax.plot(full_learned_path[:, idx_x], full_learned_path[:, idx_y], '-', lw=2.0, color=COLORS['rollout'], alpha=0.85, label='Learned ODE (from IC)', zorder=4)
    ax.plot(full_learned_path[::marker_every, idx_x], full_learned_path[::marker_every, idx_y], '*', ms=8, color=COLORS['rollout'], alpha=0.85, markeredgecolor='black', markeredgewidth=0.5, zorder=4)
    for (i, traj) in enumerate(learned_trajectories[1:]):
        ci = traj_cmap(i / max(1, len(learned_trajectories) - 1))
        ax.plot(traj[:, idx_x], traj[:, idx_y], '-', lw=1.2, color=ci, alpha=0.6, zorder=2)
        ax.plot(traj[0, idx_x], traj[0, idx_y], 'o', ms=7, color=ci, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
    ax.plot(full_learned_path[0, idx_x], full_learned_path[0, idx_y], 'o', ms=10, color=COLORS['meas'], markeredgecolor='black', markeredgewidth=1.5, zorder=6, label='Initial Conditions')
    ax.axhline(0, color='gray', linewidth=0.5, zorder=1)
    ax.axvline(0, color='gray', linewidth=0.5, zorder=1)
    ax.set_xlabel(phase_label_x, fontweight='bold')
    ax.set_ylabel(phase_label_y, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=10, framealpha=0.9, loc='best')
    if avg_errors and 'hausdorff_phase_error' in avg_errors:
        haus = avg_errors['hausdorff_phase_error']
        ax.text(0.02, 0.02, f'Hausdorff = {haus:.4f}', transform=ax.transAxes, fontsize=9, va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    fig_phase.suptitle(title, fontweight='bold', fontsize=14)
    fig_phase.tight_layout()
    phase_arrays = {'t': t_np, 'true_states': true_np, 'learned_trajectories': np.stack(learned_trajectories, axis=0), 'state_indices': np.array(state_indices), 'state_labels': np.array(state_labels), 'errors': np.array(all_errors, dtype=object) if all_errors else None}
    if estimated_np is not None:
        phase_arrays['estimated_states'] = estimated_np
    phase_tag = title.replace(' ', '_').replace(':', '')
    print('[viz] phase portrait: saving phase figure')
    _save_fig_and_arrays(base_dir=save_dir, prefix=save_prefix, tag=phase_tag, fig=fig_phase, arrays_dict=phase_arrays, epoch=epoch, epoch_timestamp=epoch_timestamp)
    _show_if_interactive(show)
    plt.close(fig_phase)
    ts_indices = list(range(true_np.shape[1])) if true_np.shape[1] <= 3 and len(labels_seq) > 2 else [idx_x, idx_y]
    n_ts = len(ts_indices)
    (fig_ts, ts_axes) = plt.subplots(n_ts, 1, figsize=(8, 2.2 * n_ts), sharex=True)
    if not isinstance(ts_axes, np.ndarray):
        ts_axes = np.array([ts_axes])
    t_learned_full = np.linspace(0, integration_time, len(full_learned_path))
    min_len = min(len(t_learned_full), len(t_np))
    for (ax_t, idx) in zip(ts_axes, ts_indices):
        lbl = _label_for_idx(idx)
        ax_t.plot(t_np[:min_len], true_np[:min_len, idx], ':', lw=2, color=COLORS['true'], label='True')
        ax_t.plot(t_np[:min_len:marker_every], true_np[:min_len:marker_every, idx], 'o', ms=5, color=COLORS['true'], markeredgecolor='black', markeredgewidth=0.5)
        ax_t.plot(t_learned_full[:min_len], full_learned_path[:min_len, idx], '-', lw=2, color=COLORS['rollout'], label='Learned')
        ax_t.plot(t_learned_full[:min_len:marker_every], full_learned_path[:min_len:marker_every, idx], '*', ms=7, color=COLORS['rollout'], markeredgecolor='black', markeredgewidth=0.5)
        if estimated_np is not None:
            est_len = min(min_len, estimated_np.shape[0])
            ax_t.plot(t_np[:est_len], estimated_np[:est_len, idx], '-.', lw=1.8, color=COLORS['smoother'], label='Estimated (CKF/RTS)')
            ax_t.plot(t_np[:est_len:marker_every], estimated_np[:est_len:marker_every, idx], 's', ms=4.5, color=COLORS['smoother'], markeredgecolor='black', markeredgewidth=0.4)
        ax_t.set_ylabel(lbl, fontweight='bold')
        ax_t.legend(fontsize=10, framealpha=0.9)
        ax_t.grid(True, alpha=0.3)
    ts_axes[-1].set_xlabel('Time', fontweight='bold')
    ts_title = title.replace('Phase Portrait', 'Time Series')
    fig_ts.suptitle(ts_title, fontweight='bold', fontsize=14)
    fig_ts.tight_layout()
    ts_tag = phase_tag.replace('Phase_Portrait', 'Time_Series')
    ts_arrays = {'t': t_np[:min_len]}
    for idx in ts_indices:
        lbl = _label_for_idx(idx)
        ts_arrays[f'true_{lbl}'] = true_np[:min_len, idx]
        ts_arrays[f'learned_{lbl}'] = full_learned_path[:min_len, idx]
        if estimated_np is not None:
            est_len = min(min_len, estimated_np.shape[0])
            ts_arrays[f'estimated_{lbl}'] = estimated_np[:est_len, idx]
    print('[viz] phase portrait: saving time-series figure')
    _save_fig_and_arrays(base_dir=save_dir, prefix=save_prefix, tag=ts_tag, fig=fig_ts, arrays_dict=ts_arrays, epoch=epoch, epoch_timestamp=epoch_timestamp)
    _show_if_interactive(show)
    plt.close(fig_ts)
    return avg_errors

__all__ = ['_compute_vector_field', 'plot_phase_portrait']

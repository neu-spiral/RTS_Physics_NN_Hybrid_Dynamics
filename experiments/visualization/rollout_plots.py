"""Public rollout plots with preparation and primary integration."""

from .common import *
from .rollout_rendering import integrate_comparison_rollouts, render_rollouts


def prepare_rollouts(self):
    print(f'[viz] plot_rollouts: save_dir={self.save_dir}, prefix={self.save_prefix}, epoch={self.epoch}')
    if self.init_states is None:
        self.init_states = self.smoothed_states
    self.show_env = os.getenv('ODE_PLOT_SHOW', '')
    if self.show_env:
        self.show = self.show_env == '1'
    elif os.name != 'nt' and (not os.getenv('DISPLAY')):
        self.show = False
    plt.close('all')
    reset_plot_params()
    self.t_np = self.t.detach().cpu().numpy()
    self.T = self.t_np.shape[0]
    self.states_np = self.smoothed_states.detach().cpu().numpy()
    self.true_np = None
    if self.true_states is not None:
        self.true_np = self.true_states.detach().cpu().numpy()
    self.draw_hybrid_rollout_lines = True
    (self.rollout_vis_lower, self.rollout_vis_upper, self.blowup_mult) = _rollout_visibility_bounds(self.states_np, self.true_np)
    print(f'[viz] rollouts: blowup guard multiplier={self.blowup_mult:g}')
    if self.drive_seq is not None:
        self.drive_seq = self.drive_seq.to(self.smoothed_states.device)
        if self.drive_seq.dim() == 1:
            self.drive_seq = self.drive_seq.unsqueeze(-1)
        assert self.drive_seq.shape[0] == self.T, 'drive_seq must align with t'
    self.start_idxs = list(range(0, self.T - 1, self.stride))
    if len(self.start_idxs) > self.max_rollouts:
        self.step = max(1, len(self.start_idxs) // self.max_rollouts)
        self.start_idxs = self.start_idxs[::self.step]
    self.point_stride = max(1, int(os.getenv('ODE_PLOT_POINT_STRIDE', '1')))
    self.plot_markers = os.getenv('ODE_PLOT_MARKERS', '1') == '1'
    try:
        self.rollout_marker_target = max(1, int(os.getenv('ODE_ROLLOUT_MARKER_TARGET', '3')))
    except Exception:
        self.rollout_marker_target = 3
    try:
        self.rollout_marker_size = float(os.getenv('ODE_ROLLOUT_MARKER_SIZE', '3.2'))
        self.rollout_start_marker_size = float(os.getenv('ODE_ROLLOUT_START_MARKER_SIZE', '4.5'))
    except Exception:
        self.rollout_marker_size = 3.2
        self.rollout_start_marker_size = 4.5
    self.marker_scale = 3.0
    self.line_width_bump = 0.25
    self.rollout_marker_size *= self.marker_scale
    self.rollout_start_marker_size *= self.marker_scale
    print(f'[viz] rollouts: count={len(self.start_idxs)}  horizon={self.horizon_steps}  stride={self.stride}  point_stride={self.point_stride}  markers={int(self.plot_markers)}')
    self.rollout_alpha = 0.5
    self.rollout_linewidth = (3.0 if len(self.start_idxs) <= 8 else 2.4) + self.line_width_bump
    self.n_state_plots = len(self.state_indices)
    self.has_drive = self.drive_seq is not None
    (self.n_rows_states, self.n_cols_states) = _state_grid_shape(self.n_state_plots)
    self.total_rows = self.n_rows_states + (1 if self.has_drive else 0)
    self.fig_width = 6.6 * self.n_cols_states
    self.fig_height = 5.2 * self.total_rows
    self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
    self.height_ratios = [1.0] * self.n_rows_states + ([0.75] if self.has_drive else [])
    self.gs = self.fig.add_gridspec(self.total_rows, self.n_cols_states, height_ratios=self.height_ratios)
    self.state_axes = []
    self.ax_drive = None
    self.idx_state = 0
    for self.r in range(self.n_rows_states):
        for self.c in range(self.n_cols_states):
            if self.idx_state >= self.n_state_plots:
                break
            self.share_ax = self.state_axes[0] if self.state_axes else None
            self.ax = self.fig.add_subplot(self.gs[self.r, self.c], sharex=self.share_ax)
            self.state_axes.append(self.ax)
            self.idx_state += 1
        if self.idx_state >= self.n_state_plots:
            continue
    if self.has_drive:
        self.share_ax = self.state_axes[0] if self.state_axes else None
        self.ax_drive = self.fig.add_subplot(self.gs[-1, :], sharex=self.share_ax)
    self.axes_all = self.state_axes + ([self.ax_drive] if self.ax_drive is not None else [])
    self.device = self.smoothed_states.device
    self.n_states = self.smoothed_states.shape[1]
    self.progress_every = _viz_progress_every(250)
    self.rollout_values_by_state = {self.idx: [] for self.idx in self.state_indices}
    self.plotted_rollouts = 0
    self.truncated_rollouts = 0
    self.skipped_rollouts = 0
    self.all_rollouts = []
    self.plot_fast_default = '1'
    self.plot_fast = os.getenv('ODE_PLOT_FAST', self.plot_fast_default) == '1'
    self.fast_method = os.getenv('ODE_PLOT_FAST_METHOD', 'rk4_single').lower()
    if self.fast_method not in ('rk4_single', 'rk2', 'euler'):
        self.fast_method = 'rk4_single'
    self.use_fast_step = self.plot_fast and self.drive_seq is None and hasattr(self.integrator, 'f_full')
    try:
        self.plot_dt_cap = float(self.max_plot_dt) if self.max_plot_dt is not None else 0.0
    except Exception:
        self.plot_dt_cap = 0.0
    if not np.isfinite(self.plot_dt_cap) or self.plot_dt_cap <= 0.0:
        self.plot_dt_cap = 0.0

    def _fast_step(y, dt):
        if self.fast_method == 'rk4_single' and hasattr(self.integrator, '_rk4_single'):
            return self.integrator._rk4_single(self.y, dt)
        if self.fast_method == 'rk2':
            k1 = self.integrator.f_full(self.y)
            k2 = self.integrator.f_full(self.y + 0.5 * dt * k1)
            y_next = self.y + dt * k2
        else:
            k1 = self.integrator.f_full(self.y)
            y_next = self.y + dt * k1
        if hasattr(self.integrator, 'physics') and hasattr(self.integrator.physics, 'clip_states'):
            return self.integrator.physics.clip_states(y_next)
        return y_next
    self._fast_step = _fast_step

    def _plot_substep_count(dt_val):
        dt_abs = abs(float(dt_val))
        if self.plot_dt_cap > 0.0 and dt_abs > self.plot_dt_cap:
            return max(1, int(dt_abs / self.plot_dt_cap + 0.999999))
        return 1
    self._plot_substep_count = _plot_substep_count

    def _dense_rollout_times(base_t_np):
        base_t_np = np.asarray(base_t_np, dtype=np.float64)
        if base_t_np.ndim != 1 or base_t_np.size <= 1 or self.plot_dt_cap <= 0.0:
            return base_t_np
        dense_times = [float(base_t_np[0])]
        for i in range(base_t_np.size - 1):
            t0_seg = float(base_t_np[i])
            t1_seg = float(base_t_np[i + 1])
            n_seg = self._plot_substep_count(t1_seg - t0_seg)
            if n_seg <= 1:
                dense_times.append(t1_seg)
                continue
            seg = np.linspace(t0_seg, t1_seg, n_seg + 1, endpoint=True, dtype=np.float64)
            dense_times.extend(seg[1:].tolist())
        return np.asarray(dense_times, dtype=np.float64)
    self._dense_rollout_times = _dense_rollout_times
    self.solver_method = str(self.reference_solver_method).strip().upper() if self.reference_solver_method is not None else 'BDF'
    if self.solver_method not in ('RK4', 'BDF', 'RADAU'):
        raise ValueError(f"Unsupported reference_solver_method '{self.reference_solver_method}'. Expected 'rk4', 'BDF', or 'Radau'.")

    def _reference_physics_rollout(y0_np, t_eval_np):
        return None
    self._reference_physics_rollout = _reference_physics_rollout

    def _reference_hybrid_rollout(y0_np, t_eval_np):
        if self.drive_seq is not None:
            return None
        try:
            from scipy.integrate import solve_ivp
        except Exception as e:
            print(f'[viz] scipy solve_ivp unavailable for hybrid rollout: {e}')
            return None
        y0_arr = np.asarray(y0_np, dtype=np.float64)
        t_eval_arr = np.asarray(t_eval_np, dtype=np.float64)
        if t_eval_arr.ndim != 1 or t_eval_arr.size < 2:
            return None
        atol_vec = np.full_like(y0_arr, 1e-06, dtype=np.float64)
        max_step_bdf = self.plot_dt_cap if self.plot_dt_cap > 0.0 else 0.05
        torch_dtype = self.smoothed_states.dtype

        def rhs(_t_scalar, y_np):
            with torch.inference_mode():
                y_t = torch.as_tensor(self.y_np, dtype=torch_dtype, device=self.device).unsqueeze(0)
                if hasattr(self.integrator, 'physics') and hasattr(self.integrator.physics, 'clip_states'):
                    y_t = self.integrator.physics.clip_states(y_t)
                dydt_t = self.integrator.f_full(y_t)
                if dydt_t.dim() > 1:
                    dydt_t = dydt_t[0]
                dydt_t = torch.where(torch.isfinite(dydt_t), dydt_t, torch.zeros_like(dydt_t))
                return dydt_t.detach().to('cpu', dtype=torch.float64).numpy()
        try:
            sol = solve_ivp(rhs, (float(t_eval_arr[0]), float(t_eval_arr[-1])), y0_arr, method='Radau' if self.solver_method == 'RADAU' else 'BDF', rtol=0.0001, atol=atol_vec, max_step=max_step_bdf, t_eval=t_eval_arr)
            if not sol.success:
                print(f'[viz] reference hybrid rollout failed; falling back to native integrator: {sol.message}')
                return None
            return sol.y.T
        except Exception as e:
            print(f'[viz] reference hybrid rollout failed; falling back to native integrator: {e}')
            return None
    self._reference_hybrid_rollout = _reference_hybrid_rollout


def integrate_primary_rollouts(self):
    self.dt_tensor = self.t[1:] - self.t[:-1]
    if self.dt_tensor.numel() > 0:
        self.dt_min = self.dt_tensor.min()
        self.dt_max = self.dt_tensor.max()
        self.dt_mean = self.dt_tensor.mean()
        self.assume_uniform = os.getenv('ODE_PLOT_ASSUME_UNIFORM_DT', '1') == '1'
        self.tol = max(1e-08, float(self.dt_mean.abs().item()) * (0.05 if self.assume_uniform else 0.01))
        self.dt_uniform = self.assume_uniform or float((self.dt_max - self.dt_min).abs().item()) <= self.tol
        print(f'[viz] rollouts: dt min={float(self.dt_min.item()):.8f} max={float(self.dt_max.item()):.8f} mean={float(self.dt_mean.item()):.8f} tol={self.tol:.8f} uniform={self.dt_uniform}')
        if self.dt_uniform and self.plot_dt_cap > 0.0 and (float(self.dt_mean.item()) > self.plot_dt_cap + self.tol):
            print(f'[viz] rollouts: forcing dense per-rollout plotting with max_plot_dt={self.plot_dt_cap:.6g}')
            self.dt_uniform = False
    else:
        self.dt_uniform = False
    self.t_integ_start = time.perf_counter()
    self.use_reference_hybrid = self.hybrid_use_reference_solver and (self.drive_seq is None)
    self.start_t = None
    self.max_h_each = None
    self.max_h_each_cpu = None
    self.max_h_all = 0
    self.dt_step = None
    if self.dt_uniform:
        self.dt_step = round(float(self.dt_mean.item()), 6)
        self.start_t = torch.tensor(self.start_idxs, device=self.device, dtype=torch.long)
        self.can_extrapolate_uniform = self.allow_extrapolation and self.drive_seq is None and np.isfinite(self.dt_step) and (abs(self.dt_step) > 0.0)
        if self.can_extrapolate_uniform:
            self.max_h_each = torch.full_like(self.start_t, int(self.horizon_steps))
            print(f'[viz] rollouts: allowing extrapolated uniform rollout horizon={int(self.horizon_steps)} with dt={self.dt_step:.6g}')
        else:
            self.max_h_each = torch.minimum(torch.full_like(self.start_t, int(self.horizon_steps)), self.T - 1 - self.start_t)
        self.max_h_all = int(self.max_h_each.max().item()) if self.max_h_each.numel() > 0 else 0
        self.max_h_each_cpu = self.max_h_each.detach().cpu().numpy()
    else:
        self.can_extrapolate_uniform = False

    def _rollout_base_times(start_idx, length):
        if length <= 0:
            return np.empty((0,), dtype=np.float64)
        if self.can_extrapolate_uniform:
            return float(self.t_np[self.start_idx]) + self.dt_step * np.arange(length, dtype=np.float64)
        idx_seq = self.start_idx + np.arange(length, dtype=np.int64)
        return self.t_np[idx_seq]
    self._rollout_base_times = _rollout_base_times
    if self.dt_uniform and (not self.use_reference_hybrid):
        self.y = self.init_states[self.start_t].detach().clone()
        print(f'[viz] rollouts: using batched integration path max_h={self.max_h_all}')
        with torch.no_grad():
            if self.drive_seq is None and hasattr(self.integrator, 'rk4_rollout') and (not self.use_fast_step):
                print('[viz] rollouts: using integrator.rk4_rollout kernel')
                self.ys_stack = self.integrator.rk4_rollout(self.y, self.dt_step, self.max_h_all)
            else:
                print(f'[viz] rollouts: using stepped batched path fast_step={int(self.use_fast_step)} drive={int(self.drive_seq is not None)}')
                self.ys = [self.y]
                for self.j in range(self.max_h_all):
                    if self.j == 0 or (self.j + 1) % self.progress_every == 0 or self.j + 1 == self.max_h_all:
                        print(f'[viz] rollouts batch progress: step {self.j + 1}/{self.max_h_all}')
                    if self.drive_seq is not None:
                        self.d = self.drive_seq[self.start_t + self.j]
                        if hasattr(self.integrator, 'rk4_step_with_drive'):
                            self.y = self.integrator.rk4_step_with_drive(self.y, self.dt_step, self.d)
                        elif hasattr(self.integrator, 'rk4_step_with_pin'):
                            self.y = self.integrator.rk4_step_with_pin(self.y, self.dt_step, self.d)
                        else:
                            try:
                                self.y = self.integrator.rk4_step(self.y, self.dt_step, self.d)
                            except TypeError:
                                self.y = self.integrator.rk4_step(self.y, self.dt_step)
                    elif self.use_fast_step:
                        self.y = self._fast_step(self.y, self.dt_step)
                    else:
                        self.y = self.integrator.rk4_step(self.y, self.dt_step)
                    self.ys.append(self.y)
                self.ys_stack = torch.stack(self.ys, dim=1)
        self.ys_stack_cpu = self.ys_stack.detach().cpu().numpy()
        for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
            if self.rollout_idx == 0 or self.rollout_idx + 1 == len(self.start_idxs):
                print(f'[viz] rollouts postprocess: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} start_idx={self.start_idx}')
            self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1
            self.ts_roll = self._rollout_base_times(self.start_idx, self.L_i)
            self.ys_roll = self.ys_stack_cpu[self.rollout_idx, :self.L_i]
            (self.ts_roll, self.ys_roll, self.trunc_reason, self.trunc_step) = _truncate_rollout_for_plot(self.ts_roll, self.ys_roll, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
            if self.trunc_reason is not None:
                self.truncated_rollouts += 1
                print(f'[viz] rollouts: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} truncated at local step {self.trunc_step} ({self.trunc_reason})')
            self.rollout_color = self.rollout_color_override if self.rollout_color_override is not None else ROLLOUT_COLORS[self.rollout_idx % len(ROLLOUT_COLORS)]
            if self.point_stride > 1:
                self.ts_roll = self.ts_roll[::self.point_stride]
                self.ys_roll = self.ys_roll[::self.point_stride]
            self.all_rollouts.append({'start_idx': self.start_idx, 't': self.ts_roll, 'y': self.ys_roll})
            if len(self.ts_roll) == 0:
                self.skipped_rollouts += 1
                print(f'[viz] rollouts: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} has no visible finite prefix; skipping draw')
                continue
            self.marker_every_rollout = max(1, len(self.ts_roll) // self.rollout_marker_target)
            if not self.draw_hybrid_rollout_lines:
                continue
            for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                self.rollout_values_by_state[self.idx].append(np.asarray(self.ys_roll[:, self.idx], dtype=np.float64))
                self.lbl_roll = self.rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                if self.plot_markers:
                    self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll, marker='*', markevery=self.marker_every_rollout, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                else:
                    self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll)
                self.ax.plot(self.ts_roll[0], self.ys_roll[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            self.plotted_rollouts += 1
    else:
        if self.use_reference_hybrid:
            print(f'[viz] rollouts: using {self.physics_system_name} adaptive {self.solver_method} solver for learned overlays')
            for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                print(f'[viz] rollouts BDF: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} start_idx={self.start_idx}')
                self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1 if self.dt_uniform else min(self.horizon_steps + 1, self.T - self.start_idx)
                self.ts_base = self._rollout_base_times(self.start_idx, self.L_i)
                self.ts_roll = self._dense_rollout_times(self.ts_base)
                self.ys_roll = self._reference_hybrid_rollout(self.init_states[self.start_idx].detach().cpu().numpy(), self.ts_roll)
                self.rollout_color = self.rollout_color_override if self.rollout_color_override is not None else ROLLOUT_COLORS[self.rollout_idx % len(ROLLOUT_COLORS)]
                if self.ys_roll is None:
                    self.use_reference_hybrid = False
                    print('[viz] rollouts: hybrid stiff solver unavailable; falling back to native hybrid integrator')
                    break
                (self.ts_roll, self.ys_roll, self.trunc_reason, self.trunc_step) = _truncate_rollout_for_plot(self.ts_roll, self.ys_roll, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                if self.trunc_reason is not None:
                    self.truncated_rollouts += 1
                    print(f'[viz] rollouts BDF: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} truncated at local step {self.trunc_step} ({self.trunc_reason})')
                if self.point_stride > 1:
                    self.ts_roll = self.ts_roll[::self.point_stride]
                    self.ys_roll = self.ys_roll[::self.point_stride]
                self.all_rollouts.append({'start_idx': self.start_idx, 't': self.ts_roll, 'y': self.ys_roll})
                if len(self.ts_roll) == 0:
                    self.skipped_rollouts += 1
                    continue
                self.marker_every_rollout = max(1, len(self.ts_roll) // self.rollout_marker_target)
                if not self.draw_hybrid_rollout_lines:
                    continue
                for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                    self.rollout_values_by_state[self.idx].append(np.asarray(self.ys_roll[:, self.idx], dtype=np.float64))
                    self.lbl_roll = self.rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                    if self.plot_markers:
                        self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll, marker='*', markevery=self.marker_every_rollout, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                    else:
                        self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll)
                    self.ax.plot(self.ts_roll[0], self.ys_roll[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
                self.plotted_rollouts += 1
        if not self.use_reference_hybrid:
            print('[viz] rollouts: using slow per-rollout integration fallback')
            for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                print(f'[viz] rollouts fallback: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} start_idx={self.start_idx}')
                self.y0 = self.init_states[self.start_idx].detach().clone()
                self.y_curr = self.y0.unsqueeze(0)
                self.rollout_color = self.rollout_color_override if self.rollout_color_override is not None else ROLLOUT_COLORS[self.rollout_idx % len(ROLLOUT_COLORS)]
                self.ts_roll = [self.t_np[self.start_idx]]
                self.ys_roll = [self.y0.detach().cpu().numpy()]
                self.j = self.start_idx
                self.rollout_trunc_reason = None
                self.rollout_trunc_step = None
                with torch.no_grad():
                    for self.k in range(self.horizon_steps):
                        if self.j >= self.T - 1:
                            break
                        if self.k == 0 or (self.k + 1) % self.progress_every == 0 or self.k + 1 == self.horizon_steps:
                            print(f'[viz] rollouts fallback progress: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} step {self.k + 1}/{self.horizon_steps}')
                        self.dt_step_total = float((self.t[self.j + 1] - self.t[self.j]).item())
                        self.n_plot_sub = self._plot_substep_count(self.dt_step_total)
                        self.dt_step = self.dt_step_total / float(self.n_plot_sub)
                        self.t_curr = self.ts_roll[-1]
                        self.d = self.drive_seq[self.j].unsqueeze(0) if self.drive_seq is not None else None
                        for self._sub in range(self.n_plot_sub):
                            if self.d is not None:
                                if hasattr(self.integrator, 'rk4_step_with_drive'):
                                    self.y_curr = self.integrator.rk4_step_with_drive(self.y_curr, self.dt_step, self.d)
                                elif hasattr(self.integrator, 'rk4_step_with_pin'):
                                    self.y_curr = self.integrator.rk4_step_with_pin(self.y_curr, self.dt_step, self.d)
                                else:
                                    try:
                                        self.y_curr = self.integrator.rk4_step(self.y_curr, self.dt_step, self.d)
                                    except TypeError:
                                        self.y_curr = self.integrator.rk4_step(self.y_curr, self.dt_step)
                            elif self.use_fast_step:
                                self.y_curr = self._fast_step(self.y_curr, self.dt_step)
                            else:
                                self.y_curr = self.integrator.rk4_step(self.y_curr, self.dt_step)
                            self.y_next_np = self.y_curr.squeeze(0).detach().cpu().numpy()
                            if not np.isfinite(self.y_next_np).all():
                                self.rollout_trunc_reason = 'non-finite'
                                self.rollout_trunc_step = self.k + 1
                                self.truncated_rollouts += 1
                                print(f'[viz] rollouts fallback: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} truncated at local step {self.rollout_trunc_step} (non-finite)')
                                break
                            if np.any(self.y_next_np < self.rollout_vis_lower) or np.any(self.y_next_np > self.rollout_vis_upper):
                                self.rollout_trunc_reason = 'left-visible-range'
                                self.rollout_trunc_step = self.k + 1
                                self.truncated_rollouts += 1
                                print(f'[viz] rollouts fallback: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} truncated at local step {self.rollout_trunc_step} (left-visible-range)')
                                break
                            self.t_curr += self.dt_step
                            self.ts_roll.append(self.t_curr)
                            self.ys_roll.append(self.y_next_np)
                        if self.rollout_trunc_reason is not None:
                            break
                        self.j += 1
                self.ts_roll = np.asarray(self.ts_roll, dtype=np.float64)
                self.ys_roll = np.asarray(self.ys_roll, dtype=np.float32)
                (self.ts_roll, self.ys_roll, self.trunc_reason_post, self.trunc_step_post) = _truncate_rollout_for_plot(self.ts_roll, self.ys_roll, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                if self.trunc_reason_post is not None and self.rollout_trunc_reason is None:
                    self.truncated_rollouts += 1
                    print(f'[viz] rollouts fallback: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} truncated at local step {self.trunc_step_post} ({self.trunc_reason_post})')
                if self.point_stride > 1:
                    self.ts_roll = self.ts_roll[::self.point_stride]
                    self.ys_roll = self.ys_roll[::self.point_stride]
                self.all_rollouts.append({'start_idx': self.start_idx, 't': self.ts_roll, 'y': self.ys_roll})
                if len(self.ts_roll) == 0:
                    self.skipped_rollouts += 1
                    print(f'[viz] rollouts fallback: rollout {self.rollout_idx + 1}/{len(self.start_idxs)} has no visible finite prefix; skipping draw')
                    continue
                self.marker_every_rollout = max(1, len(self.ts_roll) // self.rollout_marker_target)
                if not self.draw_hybrid_rollout_lines:
                    continue
                for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                    self.rollout_values_by_state[self.idx].append(np.asarray(self.ys_roll[:, self.idx], dtype=np.float64))
                    self.lbl_roll = self.rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                    if self.plot_markers:
                        self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll, marker='*', markevery=self.marker_every_rollout, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                    else:
                        self.ax.plot(self.ts_roll, self.ys_roll[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color, alpha=self.rollout_alpha, label=self.lbl_roll)
                    self.ax.plot(self.ts_roll[0], self.ys_roll[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
                self.plotted_rollouts += 1


class RolloutPlotSession:
    def __init__(self, values):
        self.__dict__.update(values)

    def run(self):
        prepare_rollouts(self)
        integrate_primary_rollouts(self)
        integrate_comparison_rollouts(self)
        return render_rollouts(self)


def plot_rollouts(t, smoothed_states, integrator, state_indices, state_labels, true_states=None, Ps_s=None, horizon_steps=20, stride=5, max_rollouts=20, title='rollouts', show=True, save_dir='results', save_prefix='experiment', drive_seq=None, epoch=None, epoch_timestamp=None, rollout_color_override=None, physics_integrator=None, physics_system_name=None, physics_system_cfg=None, physics_use_reference_solver=True, hybrid_use_reference_solver=False, reference_solver_method='BDF', physics_rollout_color='purple', rollout_label='hybrid ODE', physics_rollout_label='physics ODE', init_states=None, physics_init_states_alt=None, physics_rollout_color_alt='green', physics_rollout_label_alt=None, init_states_alt=None, rollout_color_alt='orange', rollout_label_alt=None, max_plot_dt=None, allow_extrapolation=False, show_state_context=True):
    """Plot hybrid and optional physics-only rollouts from selected states."""
    return RolloutPlotSession(locals()).run()


__all__ = ['plot_rollouts']

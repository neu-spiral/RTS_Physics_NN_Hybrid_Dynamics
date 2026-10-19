"""Comparison integration and rendering for rollout plots."""

from .common import *


def integrate_comparison_rollouts(self):
    if self.init_states_alt is not None:
        print(f'[viz] rollouts: running hybrid rollouts from alternate init states')
        self.rollout_color_a = self.rollout_color_alt
        with torch.no_grad():
            if self.use_reference_hybrid:
                print(f'[viz] rollouts: using {self.physics_system_name} adaptive {self.solver_method} solver for alternate learned overlays')
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1 if self.dt_uniform else min(self.horizon_steps + 1, self.T - self.start_idx)
                    self.ts_ra = self._dense_rollout_times(self._rollout_base_times(self.start_idx, self.L_i))
                    self.ys_ra = self._reference_hybrid_rollout(self.init_states_alt[self.start_idx].detach().cpu().numpy(), self.ts_ra)
                    if self.ys_ra is None:
                        self.use_reference_hybrid = False
                        print('[viz] rollouts: hybrid stiff solver unavailable for alternate init; falling back to native hybrid integrator')
                        break
                    (self.ts_ra, self.ys_ra, self._, self._) = _truncate_rollout_for_plot(self.ts_ra, self.ys_ra, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_ra = self.ts_ra[::self.point_stride]
                        self.ys_ra = self.ys_ra[::self.point_stride]
                    if len(self.ts_ra) == 0:
                        continue
                    self.me_a = max(1, len(self.ts_ra) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_a = self.rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a, marker='s', markevery=self.me_a, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a)
                        self.ax.plot(self.ts_ra[0], self.ys_ra[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color_a, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            if not self.use_reference_hybrid and self.dt_uniform:
                self.y_a = self.init_states_alt[self.start_t].detach().clone()
                if self.drive_seq is None and hasattr(self.integrator, 'rk4_rollout'):
                    self.ys_a_raw = self.integrator.rk4_rollout(self.y_a, self.dt_step, self.max_h_all)
                    self.ys_a_stack = self.ys_a_raw.detach().cpu().numpy()
                else:
                    self.ys_a_list = [self.y_a]
                    for self.j in range(self.max_h_all):
                        if self.drive_seq is not None:
                            self.d = self.drive_seq[self.start_t + self.j]
                            self.y_a = self.integrator.rk4_step_with_drive(self.y_a, self.dt_step, self.d) if hasattr(self.integrator, 'rk4_step_with_drive') else self.integrator.rk4_step(self.y_a, self.dt_step)
                        else:
                            self.y_a = self.integrator.rk4_step(self.y_a, self.dt_step)
                        self.ys_a_list.append(self.y_a)
                    self.ys_a_stack = torch.stack(self.ys_a_list, dim=1).detach().cpu().numpy()
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1
                    self.ts_ra = self._rollout_base_times(self.start_idx, self.L_i)
                    self.ys_ra = self.ys_a_stack[self.rollout_idx, :self.L_i]
                    (self.ts_ra, self.ys_ra, self._, self._) = _truncate_rollout_for_plot(self.ts_ra, self.ys_ra, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_ra = self.ts_ra[::self.point_stride]
                        self.ys_ra = self.ys_ra[::self.point_stride]
                    if len(self.ts_ra) == 0:
                        continue
                    self.me_a = max(1, len(self.ts_ra) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_a = self.rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a, marker='s', markevery=self.me_a, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a)
                        self.ax.plot(self.ts_ra[0], self.ys_ra[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color_a, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            elif not self.use_reference_hybrid:
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.y0 = self.init_states_alt[self.start_idx].detach().clone()
                    self.y_curr_a = self.y0.unsqueeze(0)
                    self.ts_ra = [self.t_np[self.start_idx]]
                    self.ys_ra = [self.y0.detach().cpu().numpy()]
                    self.j = self.start_idx
                    for self.k in range(self.horizon_steps):
                        if self.j >= self.T - 1:
                            break
                        self.dt_s_total = float((self.t[self.j + 1] - self.t[self.j]).item())
                        self.n_plot_sub = self._plot_substep_count(self.dt_s_total)
                        self.dt_s = self.dt_s_total / float(self.n_plot_sub)
                        self.t_curr = self.ts_ra[-1]
                        self.d = self.drive_seq[self.j].unsqueeze(0) if self.drive_seq is not None else None
                        self.stop_rollout = False
                        for self._sub in range(self.n_plot_sub):
                            if self.d is not None:
                                self.y_curr_a = self.integrator.rk4_step_with_drive(self.y_curr_a, self.dt_s, self.d) if hasattr(self.integrator, 'rk4_step_with_drive') else self.integrator.rk4_step(self.y_curr_a, self.dt_s)
                            else:
                                self.y_curr_a = self.integrator.rk4_step(self.y_curr_a, self.dt_s)
                            self.y_np = self.y_curr_a.squeeze(0).detach().cpu().numpy()
                            if not np.isfinite(self.y_np).all():
                                self.stop_rollout = True
                                break
                            if np.any(self.y_np < self.rollout_vis_lower) or np.any(self.y_np > self.rollout_vis_upper):
                                self.stop_rollout = True
                                break
                            self.t_curr += self.dt_s
                            self.ts_ra.append(self.t_curr)
                            self.ys_ra.append(self.y_np)
                        if self.stop_rollout:
                            break
                        self.j += 1
                    self.ts_ra = np.asarray(self.ts_ra, dtype=np.float64)
                    self.ys_ra = np.asarray(self.ys_ra, dtype=np.float32)
                    (self.ts_ra, self.ys_ra, self._, self._) = _truncate_rollout_for_plot(self.ts_ra, self.ys_ra, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_ra = self.ts_ra[::self.point_stride]
                        self.ys_ra = self.ys_ra[::self.point_stride]
                    if len(self.ts_ra) == 0:
                        continue
                    self.me_a = max(1, len(self.ts_ra) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_a = self.rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a, marker='s', markevery=self.me_a, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_ra, self.ys_ra[:, self.idx], '-', lw=self.rollout_linewidth, color=self.rollout_color_a, alpha=self.rollout_alpha, label=self.lbl_a)
                        self.ax.plot(self.ts_ra[0], self.ys_ra[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.rollout_color_a, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
        print('[viz] rollouts: hybrid rollouts (alt init) done')
    if self.physics_integrator is not None:
        print(f'[viz] rollouts: running physics-only rollouts from same initial points')
        self.phys_color = self.physics_rollout_color
        self.use_reference_solver = False
        with torch.no_grad():
            if self.use_reference_solver:
                print(f'[viz] rollouts: using reference {self.solver_method} solver for physics-only overlays')
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1 if self.dt_uniform else min(self.horizon_steps + 1, self.T - self.start_idx)
                    self.ts_roll_p = self._dense_rollout_times(self._rollout_base_times(self.start_idx, self.L_i))
                    self.ys_roll_p = self._reference_physics_rollout(self.init_states[self.start_idx].detach().cpu().numpy(), self.ts_roll_p)
                    if self.ys_roll_p is None:
                        self.use_reference_solver = False
                        print('[viz] rollouts: reference solver unavailable; falling back to native physics integrator')
                        break
                    (self.ts_roll_p, self.ys_roll_p, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_p, self.ys_roll_p, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_p = self.ts_roll_p[::self.point_stride]
                        self.ys_roll_p = self.ys_roll_p[::self.point_stride]
                    if len(self.ts_roll_p) == 0:
                        continue
                    self.marker_every_p = max(1, len(self.ts_roll_p) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_p = self.physics_rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p, marker='*', markevery=self.marker_every_p, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p)
                        self.ax.plot(self.ts_roll_p[0], self.ys_roll_p[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            if not self.use_reference_solver and self.dt_uniform:
                self.y_phys = self.init_states[self.start_t].detach().clone()
                self.ys_phys = [self.y_phys]
                for self.j in range(self.max_h_all):
                    if self.drive_seq is not None:
                        self.d = self.drive_seq[self.start_t + self.j]
                        if hasattr(self.physics_integrator, 'rk4_step_with_drive'):
                            self.y_phys = self.physics_integrator.rk4_step_with_drive(self.y_phys, self.dt_step, self.d)
                        else:
                            self.y_phys = self.physics_integrator.rk4_step(self.y_phys, self.dt_step)
                    else:
                        self.y_phys = self.physics_integrator.rk4_step(self.y_phys, self.dt_step)
                    self.ys_phys.append(self.y_phys)
                self.ys_phys_stack = torch.stack(self.ys_phys, dim=1).detach().cpu().numpy()
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1
                    self.ts_roll_p = self._rollout_base_times(self.start_idx, self.L_i)
                    self.ys_roll_p = self.ys_phys_stack[self.rollout_idx, :self.L_i]
                    (self.ts_roll_p, self.ys_roll_p, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_p, self.ys_roll_p, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_p = self.ts_roll_p[::self.point_stride]
                        self.ys_roll_p = self.ys_roll_p[::self.point_stride]
                    if len(self.ts_roll_p) == 0:
                        continue
                    self.marker_every_p = max(1, len(self.ts_roll_p) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_p = self.physics_rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p, marker='*', markevery=self.marker_every_p, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p)
                        self.ax.plot(self.ts_roll_p[0], self.ys_roll_p[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            elif not self.use_reference_solver:
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.y0 = self.init_states[self.start_idx].detach().clone()
                    self.y_curr_p = self.y0.unsqueeze(0)
                    self.ts_roll_p = [self.t_np[self.start_idx]]
                    self.ys_roll_p = [self.y0.detach().cpu().numpy()]
                    self.j = self.start_idx
                    for self.k in range(self.horizon_steps):
                        if self.j >= self.T - 1:
                            break
                        self.dt_s_total = float((self.t[self.j + 1] - self.t[self.j]).item())
                        self.n_plot_sub = self._plot_substep_count(self.dt_s_total)
                        self.dt_s = self.dt_s_total / float(self.n_plot_sub)
                        self.t_curr = self.ts_roll_p[-1]
                        self.d = self.drive_seq[self.j].unsqueeze(0) if self.drive_seq is not None else None
                        self.stop_rollout = False
                        for self._sub in range(self.n_plot_sub):
                            if self.d is not None:
                                if hasattr(self.physics_integrator, 'rk4_step_with_drive'):
                                    self.y_curr_p = self.physics_integrator.rk4_step_with_drive(self.y_curr_p, self.dt_s, self.d)
                                else:
                                    self.y_curr_p = self.physics_integrator.rk4_step(self.y_curr_p, self.dt_s)
                            else:
                                self.y_curr_p = self.physics_integrator.rk4_step(self.y_curr_p, self.dt_s)
                            self.y_next_np = self.y_curr_p.squeeze(0).detach().cpu().numpy()
                            if not np.isfinite(self.y_next_np).all():
                                self.stop_rollout = True
                                break
                            if np.any(self.y_next_np < self.rollout_vis_lower) or np.any(self.y_next_np > self.rollout_vis_upper):
                                self.stop_rollout = True
                                break
                            self.t_curr += self.dt_s
                            self.ts_roll_p.append(self.t_curr)
                            self.ys_roll_p.append(self.y_next_np)
                        if self.stop_rollout:
                            break
                        self.j += 1
                    self.ts_roll_p = np.asarray(self.ts_roll_p, dtype=np.float64)
                    self.ys_roll_p = np.asarray(self.ys_roll_p, dtype=np.float32)
                    (self.ts_roll_p, self.ys_roll_p, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_p, self.ys_roll_p, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_p = self.ts_roll_p[::self.point_stride]
                        self.ys_roll_p = self.ys_roll_p[::self.point_stride]
                    if len(self.ts_roll_p) == 0:
                        continue
                    self.marker_every_p = max(1, len(self.ts_roll_p) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_p = self.physics_rollout_label if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p, marker='*', markevery=self.marker_every_p, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_p, self.ys_roll_p[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color, alpha=self.rollout_alpha, label=self.lbl_p)
                        self.ax.plot(self.ts_roll_p[0], self.ys_roll_p[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
        print('[viz] rollouts: physics-only rollouts done')
    if self.physics_integrator is not None and self.physics_init_states_alt is not None:
        print(f'[viz] rollouts: running physics-only rollouts from alternate init states')
        self.phys_color_alt = self.physics_rollout_color_alt
        self.use_reference_solver_alt = False
        with torch.no_grad():
            if self.use_reference_solver_alt:
                print(f'[viz] rollouts: using reference {self.solver_method} solver for alt physics overlays')
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1 if self.dt_uniform else min(self.horizon_steps + 1, self.T - self.start_idx)
                    self.ts_roll_pa = self._dense_rollout_times(self._rollout_base_times(self.start_idx, self.L_i))
                    self.ys_roll_pa = self._reference_physics_rollout(self.physics_init_states_alt[self.start_idx].detach().cpu().numpy(), self.ts_roll_pa)
                    if self.ys_roll_pa is None:
                        self.use_reference_solver_alt = False
                        print('[viz] rollouts: reference solver unavailable for alt init; falling back to native physics integrator')
                        break
                    (self.ts_roll_pa, self.ys_roll_pa, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_pa, self.ys_roll_pa, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_pa = self.ts_roll_pa[::self.point_stride]
                        self.ys_roll_pa = self.ys_roll_pa[::self.point_stride]
                    if len(self.ts_roll_pa) == 0:
                        continue
                    self.marker_every_pa = max(1, len(self.ts_roll_pa) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_pa = self.physics_rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa, marker='^', markevery=self.marker_every_pa, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa)
                        self.ax.plot(self.ts_roll_pa[0], self.ys_roll_pa[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color_alt, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            if not self.use_reference_solver_alt and self.dt_uniform:
                self.y_phys_alt = self.physics_init_states_alt[self.start_t].detach().clone()
                self.ys_phys_alt = [self.y_phys_alt]
                for self.j in range(self.max_h_all):
                    if self.drive_seq is not None:
                        self.d = self.drive_seq[self.start_t + self.j]
                        if hasattr(self.physics_integrator, 'rk4_step_with_drive'):
                            self.y_phys_alt = self.physics_integrator.rk4_step_with_drive(self.y_phys_alt, self.dt_step, self.d)
                        else:
                            self.y_phys_alt = self.physics_integrator.rk4_step(self.y_phys_alt, self.dt_step)
                    else:
                        self.y_phys_alt = self.physics_integrator.rk4_step(self.y_phys_alt, self.dt_step)
                    self.ys_phys_alt.append(self.y_phys_alt)
                self.ys_phys_alt_stack = torch.stack(self.ys_phys_alt, dim=1).detach().cpu().numpy()
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.L_i = int(self.max_h_each_cpu[self.rollout_idx]) + 1
                    self.ts_roll_pa = self._rollout_base_times(self.start_idx, self.L_i)
                    self.ys_roll_pa = self.ys_phys_alt_stack[self.rollout_idx, :self.L_i]
                    (self.ts_roll_pa, self.ys_roll_pa, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_pa, self.ys_roll_pa, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_pa = self.ts_roll_pa[::self.point_stride]
                        self.ys_roll_pa = self.ys_roll_pa[::self.point_stride]
                    if len(self.ts_roll_pa) == 0:
                        continue
                    self.marker_every_pa = max(1, len(self.ts_roll_pa) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_pa = self.physics_rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa, marker='^', markevery=self.marker_every_pa, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa)
                        self.ax.plot(self.ts_roll_pa[0], self.ys_roll_pa[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color_alt, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
            elif not self.use_reference_solver_alt:
                for (self.rollout_idx, self.start_idx) in enumerate(self.start_idxs):
                    self.y0 = self.physics_init_states_alt[self.start_idx].detach().clone()
                    self.y_curr_pa = self.y0.unsqueeze(0)
                    self.ts_roll_pa = [self.t_np[self.start_idx]]
                    self.ys_roll_pa = [self.y0.detach().cpu().numpy()]
                    self.j = self.start_idx
                    for self.k in range(self.horizon_steps):
                        if self.j >= self.T - 1:
                            break
                        self.dt_s_total = float((self.t[self.j + 1] - self.t[self.j]).item())
                        self.n_plot_sub = self._plot_substep_count(self.dt_s_total)
                        self.dt_s = self.dt_s_total / float(self.n_plot_sub)
                        self.t_curr = self.ts_roll_pa[-1]
                        self.d = self.drive_seq[self.j].unsqueeze(0) if self.drive_seq is not None else None
                        self.stop_rollout = False
                        for self._sub in range(self.n_plot_sub):
                            if self.d is not None:
                                if hasattr(self.physics_integrator, 'rk4_step_with_drive'):
                                    self.y_curr_pa = self.physics_integrator.rk4_step_with_drive(self.y_curr_pa, self.dt_s, self.d)
                                else:
                                    self.y_curr_pa = self.physics_integrator.rk4_step(self.y_curr_pa, self.dt_s)
                            else:
                                self.y_curr_pa = self.physics_integrator.rk4_step(self.y_curr_pa, self.dt_s)
                            self.y_next_np = self.y_curr_pa.squeeze(0).detach().cpu().numpy()
                            if not np.isfinite(self.y_next_np).all():
                                self.stop_rollout = True
                                break
                            if np.any(self.y_next_np < self.rollout_vis_lower) or np.any(self.y_next_np > self.rollout_vis_upper):
                                self.stop_rollout = True
                                break
                            self.t_curr += self.dt_s
                            self.ts_roll_pa.append(self.t_curr)
                            self.ys_roll_pa.append(self.y_next_np)
                        if self.stop_rollout:
                            break
                        self.j += 1
                    self.ts_roll_pa = np.asarray(self.ts_roll_pa, dtype=np.float64)
                    self.ys_roll_pa = np.asarray(self.ys_roll_pa, dtype=np.float32)
                    (self.ts_roll_pa, self.ys_roll_pa, self._, self._) = _truncate_rollout_for_plot(self.ts_roll_pa, self.ys_roll_pa, lower_bounds=self.rollout_vis_lower, upper_bounds=self.rollout_vis_upper)
                    if self.point_stride > 1:
                        self.ts_roll_pa = self.ts_roll_pa[::self.point_stride]
                        self.ys_roll_pa = self.ys_roll_pa[::self.point_stride]
                    if len(self.ts_roll_pa) == 0:
                        continue
                    self.marker_every_pa = max(1, len(self.ts_roll_pa) // self.rollout_marker_target)
                    for (self.ax, self.idx) in zip(self.state_axes, self.state_indices):
                        self.lbl_pa = self.physics_rollout_label_alt if self.rollout_idx == 0 and self.idx == self.state_indices[0] else None
                        if self.plot_markers:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa, marker='^', markevery=self.marker_every_pa, markersize=self.rollout_marker_size, markeredgecolor='black', markeredgewidth=0.4)
                        else:
                            self.ax.plot(self.ts_roll_pa, self.ys_roll_pa[:, self.idx], '-', lw=self.rollout_linewidth, color=self.phys_color_alt, alpha=self.rollout_alpha, label=self.lbl_pa)
                        self.ax.plot(self.ts_roll_pa[0], self.ys_roll_pa[0, self.idx], 'D', ms=self.rollout_start_marker_size, color=self.phys_color_alt, markeredgecolor='black', markeredgewidth=1.0, zorder=5)
        print('[viz] rollouts: physics-only rollouts (alt init) done')

def render_rollouts(self):
    self.t_integ = time.perf_counter() - self.t_integ_start
    print(f'[viz] rollouts integrated in {self.t_integ:.2f}s (batch={len(self.start_idxs)}, horizon={self.horizon_steps}, dt_uniform={self.dt_uniform})')
    print(f'[viz] rollouts summary: plotted={self.plotted_rollouts} truncated={self.truncated_rollouts} skipped={self.skipped_rollouts}')
    self.Ps_s_np = None
    if self.show_state_context and self.Ps_s is not None:
        self.Ps_s_np = self.Ps_s.detach().cpu().numpy()
    try:
        self.marker_target_main = max(1, int(os.getenv('ODE_MAIN_MARKER_TARGET', '8')))
    except Exception:
        self.marker_target_main = 8
    self.marker_every = max(1, self.T // self.marker_target_main)
    try:
        self.marker_size_main = float(os.getenv('ODE_MAIN_MARKER_SIZE', '3.2'))
        self.marker_size_true = float(os.getenv('ODE_MAIN_TRUE_MARKER_SIZE', '3.0'))
        self.marker_size_drive = float(os.getenv('ODE_MAIN_DRIVE_MARKER_SIZE', '3.0'))
    except Exception:
        self.marker_size_main = 3.2
        self.marker_size_true = 3.0
        self.marker_size_drive = 3.0
    self.marker_size_main *= self.marker_scale
    self.marker_size_true *= self.marker_scale
    self.marker_size_drive *= self.marker_scale
    try:
        self.nonrollout_marker_alpha = float(os.getenv('ODE_NONROLLOUT_MARKER_ALPHA', '0.5'))
    except Exception:
        self.nonrollout_marker_alpha = 0.5
    self.nonrollout_marker_alpha = max(0.0, min(1.0, self.nonrollout_marker_alpha))
    self.edge_alpha = min(1.0, self.nonrollout_marker_alpha + 0.2)
    self.smoother_marker_face = matplotlib.colors.to_rgba(COLORS['smoother'], self.nonrollout_marker_alpha)
    self.true_marker_face = matplotlib.colors.to_rgba(COLORS['true'], self.nonrollout_marker_alpha)
    self.drive_marker_face = matplotlib.colors.to_rgba(COLORS['drive'], self.nonrollout_marker_alpha)
    self.marker_edge_rgba = matplotlib.colors.to_rgba('black', self.edge_alpha)
    self.first_state_idx = self.state_indices[0] if self.state_indices else None
    for (self.ax, self.idx, self.label) in zip(self.state_axes, self.state_indices, self.state_labels):
        if self.Ps_s_np is not None:
            self.std_s = np.sqrt(np.clip(self.Ps_s_np[:, self.idx, self.idx], 0.0, None))
            self.ax.fill_between(self.t_np, self.states_np[:, self.idx] - 2 * self.std_s, self.states_np[:, self.idx] + 2 * self.std_s, color=COLORS['smoother'], alpha=0.15, label='smoother Â±2Ïƒ' if self.idx == self.state_indices[0] else None)
        self.ax.plot(self.t_np, self.states_np[:, self.idx], '-', lw=3.8 + self.line_width_bump, color=COLORS['smoother'], label='smoothed states' if self.show_state_context and self.idx == self.state_indices[0] else None, marker='o', markevery=self.marker_every, markersize=self.marker_size_main, markerfacecolor=self.smoother_marker_face, markeredgecolor=self.marker_edge_rgba, markeredgewidth=0.4, visible=self.show_state_context)
        if self.true_np is not None:
            self.ax.plot(self.t_np, self.true_np[:, self.idx], '--', lw=3.2 + self.line_width_bump, color=COLORS['true'], label='true states' if self.show_state_context and self.idx == self.state_indices[0] else None, marker='^', markevery=self.marker_every, markersize=self.marker_size_true, markerfacecolor=self.true_marker_face, markeredgecolor=self.marker_edge_rgba, markeredgewidth=0.4, visible=self.show_state_context)
        self.ax.set_ylabel(self.label, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linewidth=0.9)
        self.y_parts = [self.states_np[:, self.idx]] if self.show_state_context else []
        if self.show_state_context and self.true_np is not None:
            self.y_parts.append(self.true_np[:, self.idx])
        self.y_parts.extend(self.rollout_values_by_state.get(self.idx, []))
        self.finite_parts = []
        self.y_all = None
        for self.part in self.y_parts:
            self.arr = np.asarray(self.part, dtype=np.float64)
            self.arr = self.arr[np.isfinite(self.arr)]
            if self.arr.size > 0:
                self.finite_parts.append(self.arr)
        if self.finite_parts:
            self.y_all = np.concatenate(self.finite_parts)
            self.y_min = float(np.min(self.y_all))
            self.y_max = float(np.max(self.y_all))
            if np.isfinite(self.y_min) and np.isfinite(self.y_max):
                if abs(self.y_max - self.y_min) < 1e-12:
                    self.pad = max(1e-06, 0.05 * max(1.0, abs(self.y_max)))
                else:
                    self.pad = 0.08 * (self.y_max - self.y_min)
                self.ax.set_ylim(self.y_min - self.pad, self.y_max + self.pad)
    if self.drive_seq is not None and self.ax_drive is not None:
        self.drive_np = self.drive_seq.detach().cpu().numpy()
        if self.drive_np.ndim == 1:
            self.drive_np = self.drive_np[:, None]
        self.ax_drive.plot(self.t_np, self.drive_np[:, 0], '-', lw=3.2 + self.line_width_bump, color=COLORS['drive'], label='Pin', marker='d', markevery=self.marker_every, markersize=self.marker_size_drive, markerfacecolor=self.drive_marker_face, markeredgecolor=self.marker_edge_rgba, markeredgewidth=0.4)
        self.ax_drive.set_ylabel('Pin', fontweight='bold')
        self.ax_drive.grid(True, alpha=0.3, linewidth=0.9)
    if self.axes_all:
        self.axes_all[-1].set_xlabel('Time', fontweight='bold')
    self.fig.suptitle(self.title, fontweight='bold', fontsize=22)
    (self.handles, self.labels) = self.state_axes[0].get_legend_handles_labels()
    if self.handles:
        self.fig.legend(self.handles, self.labels, loc='lower center', ncol=len(self.handles), fontsize=22, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    self.fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    self.roll_t_list = [self.r['t'] for self.r in self.all_rollouts]
    self.roll_y_list = [self.r['y'] for self.r in self.all_rollouts]
    self.start_idx_arr = np.array([self.r['start_idx'] for self.r in self.all_rollouts], dtype=np.int64)
    self.arrays = {'t': self.t_np, 'seed_states': self.states_np, 'rollout_start_indices': self.start_idx_arr, 'rollout_t_list': np.array(self.roll_t_list, dtype=object), 'rollout_y_list': np.array(self.roll_y_list, dtype=object)}
    if self.show_state_context:
        self.arrays['smoothed_states'] = self.states_np
        self.arrays['true_states'] = self.true_np if self.true_np is not None else np.array([])
    elif self.true_np is not None:
        self.arrays['reference_true_states'] = self.true_np
    if self.drive_seq is not None:
        self.arrays['drive_seq'] = self.drive_seq.detach().cpu().numpy()
    if self.epoch_timestamp is None and self.epoch is not None:
        self.epoch_timestamp = _timestamp_tag()
    _save_fig_and_arrays(base_dir=self.save_dir, prefix=self.save_prefix, tag=self.title.replace(' ', '_'), fig=self.fig, arrays_dict=self.arrays, epoch=self.epoch, epoch_timestamp=self.epoch_timestamp)
    _show_if_interactive(self.show)
    plt.close(self.fig)

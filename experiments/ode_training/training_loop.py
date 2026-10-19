"""Per-epoch smoothing, optimization, logging, plotting, and checkpoints."""

from __future__ import annotations

from .environment import *
from .configuration import *


def run_epoch_smoother(self):
    """Refresh or reuse the CKF/RTS targets and schedule the rollout horizon."""
    self.tic = time.time()
    print(f"\n[Epoch {self.epoch}] start: n_epochs={self.tc.n_epochs}")
    self.smooth_time_s = self.ckf_time_s = self.rts_time_s = 0.0
    self.smoother_cached = False
    self.rollout_time_s = 0.0
    self.stop_after_epoch = False
    self.do_smoother = not self.freeze_smoother and (
        self.epoch == 1
        or self.epoch % self.smoother_every == 0
        or self.last_ms_s is None
    )
    print(f"[Epoch {self.epoch}] smoother decision: do_smoother={self.do_smoother} (every={self.smoother_every})")
    smoother_ctx = torch.inference_mode() if self.smoother_inference else torch.no_grad()
    with smoother_ctx:
        if self.do_smoother:
            self._sync()
            started = time.perf_counter()
            filter_kwargs = {"drive_seq": self.drive_seq} if self.drive_seq is not None else {}
            self.ms, self.Ps = ckf_filter(
                self.smoother_integrator, self.m0, self.P0, self.y_meas, self.t,
                nonneg_indices=self.nonneg_idx, **filter_kwargs,
            )
            self._sync()
            self.ckf_time_s = time.perf_counter() - started
            self.ms_s, self.Ps_s, self.rts_time_s = self._run_guidance(
                self.smoother_integrator, self.ms, self.Ps, t_seq=self.t,
                drive_seq_arg=self.drive_seq, nonneg_indices_arg=self.nonneg_idx,
            )
            self.last_smoother_epoch = self.epoch
            self.last_ms, self.last_Ps = self.ms, self.Ps
            self.last_ms_s, self.last_Ps_s = self.ms_s, self.Ps_s
        else:
            self.ms, self.Ps = self.last_ms, self.last_Ps
            self.ms_s, self.Ps_s = self.last_ms_s, self.last_Ps_s
            self.smoother_cached = True
            print(f"[Epoch {self.epoch}] smoother: using cached outputs from epoch {self.last_smoother_epoch}")
    self.smooth_time_s = self.ckf_time_s + self.rts_time_s
    self.horizon_steps = min(
        compute_horizon_steps(
            self.epoch, self.tc.n_epochs,
            self.tc.min_rollout_horizon, self.tc.max_rollout_horizon,
        ),
        self.global_max_h,
    )
    print(f"[Epoch {self.epoch}] rollout horizon={self.horizon_steps} stride={self.tc.stride}")


def optimize_epoch(self):
    """Optimize the learned derivative using measured and smoother-derived targets."""
    learned_by_system = {
        "ho": [1], "pendulum": [1], "hh": [1],
        "brusselator": [0], "yeast": [3],
    }
    if self.exp.system_name == "retina":
        self.learned_idx = list(
            getattr(self.exp.system_cfg, "replaced_indices", getattr(self.exp, "pseudo_idx", (2,)))
        )
    else:
        self.learned_idx = learned_by_system[self.exp.system_name]
    self.full_state_wt = float(getattr(self.tc, "full_state_weight", 0.0))
    self.t_rollout_start = time.perf_counter()
    self.amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    n_horizons = max(1, _arg_int(self.exp, "multi_horizon_steps_arg", 4))
    self.intermediate_wt = max(0.0, _arg_float(self.exp, "intermediate_weight_arg", 1.0))
    if self.epoch == 1 or n_horizons == 1 or self.horizon_steps <= self.tc.min_rollout_horizon:
        horizons = [self.horizon_steps]
    else:
        horizons = sorted(
            {
                max(
                    self.tc.min_rollout_horizon,
                    int(round(
                        self.tc.min_rollout_horizon
                        + i * (self.horizon_steps - self.tc.min_rollout_horizon) / max(1, n_horizons - 1)
                    )),
                )
                for i in range(n_horizons)
            }
        )
    print(f"[Epoch {self.epoch}] loss: multi-horizon rollouts {horizons}")
    losses = []
    self.nonfinite_loss = False
    for horizon in horizons:
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=self.amp_dtype, enabled=self.scaler.is_enabled()):
            if self.loss_mode == "cov_weighted":
                meas_cov = getattr(getattr(self.integrator_for_loss, "physics", None), "R", None)
                loss = self.loss_fn(
                    t=self.t, meas=self.y_meas, smoothed_states=self.ms_s,
                    smoother_covs=self.Ps_s, integrator=self.integrator_for_loss,
                    meas_idx=self.meas_idx, horizon_steps=horizon, stride=self.tc.stride,
                    pseudo_idx=self.learned_idx if self.tc.use_pseudo_latent else None,
                    pseudo_weight=self.tc.pseudo_weight, meas_cov=meas_cov,
                    use_meas_weight=True,
                )
            else:
                loss = self.loss_fn(
                    t=self.t, meas=self.y_meas, smoothed_states=self.ms_s,
                    integrator=self.integrator_for_loss, meas_idx=self.meas_idx,
                    horizon_steps=horizon, stride=self.tc.stride,
                    pseudo_idx=self.learned_idx if self.tc.use_pseudo_latent else None,
                    pseudo_weight=self.tc.pseudo_weight,
                    intermediate_weight=self.intermediate_wt,
                    drive_seq=self.drive_seq,
                    full_state_weight=self.full_state_wt,
                )
        if not torch.isfinite(loss):
            self.nonfinite_loss = True
            print(f"[Epoch {self.epoch}] non-finite loss at horizon {horizon}; skipped")
            break
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        bad_gradient = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in self.params
        )
        if bad_gradient:
            self.optimizer.zero_grad(set_to_none=True)
            self.nonfinite_loss = True
            print(f"[Epoch {self.epoch}] non-finite gradient at horizon {horizon}; skipped")
            break
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.nnc.grad_clip)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        losses.append(float(loss.item()))
        print(f"[Epoch {self.epoch}] h={horizon} loss={losses[-1]:.6f}")

    self.rollout_time_s = time.perf_counter() - self.t_rollout_start
    self.dur = time.time() - self.tic
    self.epoch_mean_rollout_loss = float(np.mean(losses)) if losses else float("nan")
    if self.nonfinite_loss:
        self.saw_nonfinite_loss = True
        self.nonfinite_epochs.append(self.epoch)
        if not self.continue_on_nonfinite_loss:
            self.stop_after_epoch = True


def report_epoch(self):
    """Compute state errors, update the scheduler, and write compact logs."""
    with torch.no_grad():
        self.state_errors = torch.mean((self.ms_s - self.true_states) ** 2, dim=0).sqrt()
        self.total_rmse = torch.mean(self.state_errors).item()
    self.final_loss = float(self.epoch_mean_rollout_loss)
    self.final_total_rmse = float(self.total_rmse)
    self.final_state_errors = {
        label: float(value) for label, value in zip(self.state_labels, self.state_errors)
    }
    if self.scheduler is not None and not self.nonfinite_loss and np.isfinite(self.final_loss):
        self.scheduler.step(self.final_loss)
    print(
        f"{self.exp.system_name:12s} | epoch {self.epoch:03d}/{self.tc.n_epochs:03d} "
        f"h={self.horizon_steps:03d} loss={self.final_loss:.6f} RMSE={self.total_rmse:.6f} "
        f"[{self.dur:.2f}s] smooth={self.smooth_time_s:.2f}s rollout={self.rollout_time_s:.2f}s"
    )
    should_log = self.epoch % self.log_every == 0 or self.epoch in (1, self.tc.n_epochs)
    if not should_log:
        return
    epoch_dir = self.run_root / f"epoch{self.epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    with open(epoch_dir / "error_summary.txt", "w") as handle:
        handle.write(f"epoch: {self.epoch}\n")
        handle.write(f"horizon_steps: {self.horizon_steps}\n")
        handle.write(f"loss: {self.final_loss:.6f}\n")
        handle.write(f"total_rmse: {self.total_rmse:.6f}\n")
        for label, value in zip(self.state_labels, self.state_errors):
            handle.write(f"{label}_rmse: {float(value):.6f}\n")
    import csv
    exists = self.log_file.exists()
    with open(self.log_file, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "total_rmse", "horizon", "duration_s"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": self.epoch, "loss": self.final_loss,
                "total_rmse": self.total_rmse, "horizon": self.horizon_steps,
                "duration_s": self.dur,
            }
        )
    _save_training_curves(self.log_file, epoch_dir, self.title_prefix_rep)


def plot_epoch(self):
    """Save state-estimation, rollout, and optional phase-space figures."""
    self.do_plot = self.plot_every > 0 and (
        self.epoch == 1 or self.epoch == self.tc.n_epochs or self.epoch % self.plot_every == 0
    )
    if not self.do_plot:
        return
    self.epoch_timestamp = self.ts_tag
    plot_filter_smoother_uncertainty(
        t=self.t, true_states=self.true_states, meas=self.y_meas,
        meas_idx=self.meas_idx, ms=self.ms, Ps=self.Ps, ms_s=self.ms_s, Ps_s=self.Ps_s,
        state_labels=self.state_labels,
        title=f"{self.title_prefix_rep}: filter/smoother @ epoch {self.epoch}",
        show=True, save_dir="results", save_prefix=self.artifact_system_name,
        drive_seq=self.drive_seq, epoch=self.epoch, epoch_timestamp=self.epoch_timestamp,
    )
    plot_horizon = (
        len(self.t) - 1
        if self.epoch == self.tc.n_epochs
        else compute_plot_horizon_steps(len(self.t), max(1, self.tc.max_rollout_horizon * 3))
    )
    rollout_seed_states = self.true_states if self.use_true_rollouts else self.ms_s
    plot_rollouts(
        t=self.t, smoothed_states=rollout_seed_states, integrator=self.integrator,
        state_indices=list(range(self.true_states.shape[1])), state_labels=self.state_labels,
        true_states=self.true_states, Ps_s=self.Ps_s, horizon_steps=plot_horizon,
        stride=self.pc.plot_stride, max_rollouts=self.pc.max_rollouts,
        title=f"{self.title_prefix_rep}: rollouts @ epoch {self.epoch}", show=True,
        save_dir="results", save_prefix=self.artifact_system_name,
        drive_seq=self.drive_seq, epoch=self.epoch, epoch_timestamp=self.epoch_timestamp,
    )
    self.phase_errors = None
    if not self.skip_phase_portrait:
        phase_pairs = {
            "ho": [(0, 1)], "pendulum": [(0, 1)], "brusselator": [(0, 1)],
            "yeast": [(3, 5)], "hh": [(0, 1), (1, 2), (1, 3)],
            "retina": [(0, 2), (1, 2)],
        }
        results = []
        for i, j in phase_pairs[self.exp.system_name]:
            result = plot_phase_portrait(
                t=self.t, true_states=self.true_states, integrator=self.integrator,
                state_indices=(i, j), state_labels=(self.state_labels[i], self.state_labels[j]),
                n_trajectories=5,
                title=f"{self.title_prefix_rep}: {self.state_labels[i]} vs {self.state_labels[j]} @ epoch {self.epoch}",
                show=True, save_dir="results", save_prefix=self.artifact_system_name,
                epoch=self.epoch, epoch_timestamp=self.epoch_timestamp,
            )
            if result:
                results.append(result)
        if results:
            self.phase_errors = {
                "mean_phase_error": float(np.mean([r["mean_phase_error"] for r in results])),
                "hausdorff_phase_error": float(np.mean([r.get("hausdorff_phase_error", 0.0) for r in results])),
            }
            self.final_phase_errors = dict(self.phase_errors)

    if self.long_rollout_only_every > 0 and self.epoch % self.long_rollout_only_every == 0:
        long_horizon = max(
            int(self.horizon_steps) * self.long_rollout_only_multiplier,
            int(self.tc.max_rollout_horizon) * self.long_rollout_only_multiplier,
            len(self.t) - 1,
        )
        if long_horizon != plot_horizon:
            plot_rollouts(
                t=self.t, smoothed_states=rollout_seed_states, integrator=self.integrator,
                state_indices=list(range(self.true_states.shape[1])), state_labels=self.state_labels,
                true_states=self.true_states, Ps_s=None, horizon_steps=long_horizon,
                stride=self.pc.plot_stride, max_rollouts=self.pc.max_rollouts,
                title=f"{self.title_prefix_rep}: long rollout only @ epoch {self.epoch}",
                show=True, save_dir="results", save_prefix=self.artifact_system_name,
                drive_seq=self.drive_seq, epoch=self.epoch, epoch_timestamp=self.epoch_timestamp,
                rollout_label="long learned rollout", allow_extrapolation=self.drive_seq is None,
                show_state_context=False,
            )


def save_epoch_artifacts(self):
    """Persist smoother arrays and a chainable checkpoint."""
    try:
        path = self._save_epoch_smoother_outputs(self.epoch, self.ms, self.Ps, self.ms_s, self.Ps_s)
        note = f"cached_from={self.last_smoother_epoch}" if self.smoother_cached else "fresh"
        print(f"  [smoother] saved epoch {self.epoch:03d} -> {path} ({note})")
    except Exception as exc:
        print(f"  [warn] failed to save smoother outputs at epoch {self.epoch:03d}: {exc}")
    checkpoint = self._save_epoch_checkpoint(self.epoch, self.ms, self.Ps, self.ms_s, self.Ps_s)
    print(f"  [checkpoint] saved epoch {self.epoch:03d} -> {checkpoint}")
    self.last_completed_epoch = self.epoch
    return bool(self.stop_after_epoch)

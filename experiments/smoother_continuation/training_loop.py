"""Continuation epoch execution, diagnostics, and checkpoints."""

from __future__ import annotations

import csv
import json

from .support import *


def _loss_kwargs(self, horizon_steps: int):
    """Build the public continuation loss call for one rollout horizon."""
    common = dict(
        t=self.t,
        meas=self.y_meas,
        smoothed_states=self.ms_s_rollout,
        integrator=self.integrator,
        meas_idx=self.meas_idx,
        horizon_steps=horizon_steps,
        stride=self.stride,
        pseudo_idx=self.learned_idx if self.use_pseudo_latent else None,
        pseudo_weight=self.pseudo_weight,
        full_state_weight=self.full_state_weight,
        state_scales=None,
        tbptt_steps=self.tbptt_steps,
    )
    if self.loss_mode == "cov_weighted":
        common.update(
            smoother_covs=self.Ps_s,
            meas_cov=getattr(getattr(self.integrator, "physics", None), "R", None),
            use_meas_weight=True,
        )
    else:
        common.update(
            intermediate_weight=self.intermediate_weight,
            drive_seq=self.drive_seq if self.system_name == "retina" else None,
            point_weights=None,
        )
    return common


def optimize_epoch(self):
    """Run the scheduled multi-horizon optimization passes for one epoch."""
    started = time.time()
    print(f"\n[Epoch {self.epoch}/{self.n_epochs}] start")
    print(f"[Epoch {self.epoch}] smoother: frozen (checkpoint epoch {self.loaded_epoch})")
    self.horizon_steps = compute_horizon_steps(
        self.epoch, self.n_epochs, self.min_h, self.max_h
    )
    requested_horizons = self.args.multi_horizon_steps
    self.n_horizons = max(
        1,
        int(requested_horizons if requested_horizons is not None else os.getenv("ODE_MULTI_HORIZON_STEPS", "6")),
    )
    self.intermediate_weight = max(0.0, float(os.getenv("ODE_INTERMEDIATE_WEIGHT", "1.0")))
    if self.horizon_steps <= self.min_h or self.n_horizons == 1:
        self.rollout_horizons = [self.horizon_steps]
    else:
        self.rollout_horizons = sorted(
            {
                max(
                    self.min_h,
                    int(
                        round(
                            self.min_h
                            + index * (self.horizon_steps - self.min_h)
                            / max(1, self.n_horizons - 1)
                        )
                    ),
                )
                for index in range(self.n_horizons)
            }
        )
    self.rollout_repeats = max(
        1,
        int(
            self.args.rollout_repeats
            if self.args.rollout_repeats is not None
            else os.getenv("ODE_ROLLOUT_REPEATS", "1")
        ),
    )
    self.long_horizon_boost = max(
        0,
        int(
            self.args.long_horizon_boost
            if self.args.long_horizon_boost is not None
            else os.getenv("ODE_LONG_HORIZON_BOOST", "1")
        ),
    )
    print(
        f"[Epoch {self.epoch}] horizons={self.rollout_horizons} "
        f"repeats={self.rollout_repeats} long_boost={self.long_horizon_boost}"
    )
    amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
    epoch_losses = []
    self.nonfinite_loss = False

    def optimize_once(horizon_steps: int, label: str):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=self.scaler.is_enabled(),
        ):
            loss = self.loss_fn(**_loss_kwargs(self, horizon_steps))
        if not torch.isfinite(loss) or not loss.requires_grad:
            print(f"[Epoch {self.epoch}] invalid loss at {label}; skipping step")
            return None
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in self._guarded_params
        ):
            self.optimizer.zero_grad(set_to_none=True)
            print(f"[Epoch {self.epoch}] non-finite gradients at {label}; skipping step")
            return None
        torch.nn.utils.clip_grad_norm_(self._guarded_params, max_norm=self.grad_clip)
        snapshot = self._snapshot_params()
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if not self._params_finite():
            self._restore_params(snapshot)
            print(f"[Epoch {self.epoch}] non-finite parameters at {label}; reverted")
            return None
        value = float(loss.detach().item())
        print(f"[Epoch {self.epoch}] {label} h={horizon_steps} loss={value:.6g}")
        return value

    for repeat in range(self.rollout_repeats):
        for horizon in self.rollout_horizons:
            value = optimize_once(horizon, f"repeat={repeat + 1}")
            if value is None:
                self.nonfinite_loss = True
                break
            epoch_losses.append(value)
        if self.nonfinite_loss:
            break
    if not self.nonfinite_loss and self.rollout_horizons:
        longest = self.rollout_horizons[-1]
        for boost in range(self.long_horizon_boost):
            value = optimize_once(longest, f"long-boost={boost + 1}")
            if value is None:
                self.nonfinite_loss = True
                break
            epoch_losses.append(value)

    self.dur = time.time() - started
    if self.nonfinite_loss:
        self.saw_nonfinite = True
        self.nonfinite_epochs.append(self.epoch)
    self.final_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    with torch.no_grad():
        if self.true_states is None:
            self.state_errors = torch.full((self.n_state,), float("nan"), device=self.device)
            self.total_rmse = float("nan")
        else:
            self.state_errors = torch.mean((self.ms_s - self.true_states) ** 2, dim=0).sqrt()
            self.total_rmse = float(torch.mean(self.state_errors).item())
    self.final_total_rmse = self.total_rmse
    print(
        f"{self.system_name:12s} | epoch {self.epoch:03d}/{self.n_epochs:03d} "
        f"h={self.horizon_steps:03d} loss={self.final_loss:.6g} "
        f"RMSE={self.total_rmse:.6g} [{self.dur:.2f}s]"
    )


def log_epoch(self):
    """Append scalar diagnostics and decide whether to plot/save this epoch."""
    if self.epoch % self.log_every == 0 or self.epoch in (1, self.n_epochs):
        exists = self.log_file.exists()
        with self.log_file.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["epoch", "loss", "total_rmse", "horizon", "duration_s"],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": self.epoch,
                    "loss": self.final_loss,
                    "total_rmse": self.total_rmse,
                    "horizon": self.horizon_steps,
                    "duration_s": self.dur,
                }
            )
    self.should_save_checkpoint = (
        self.epoch % self.checkpoint_every == 0 or self.epoch == self.n_epochs
    )
    self.do_plot = not self.args.no_plots and (
        self.should_save_checkpoint
        or self.epoch % self.plot_every == 0
        or self.epoch == 1
    )


def _plot_training_curve(self, output_dir: Path):
    if not self.log_file.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with self.log_file.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    rmses = [float(row["total_rmse"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, losses, "b-o", ms=3)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[0].set_yscale("log")
    axes[1].plot(epochs, rmses, "r-o", ms=3)
    axes[1].set(xlabel="Epoch", ylabel="RMSE", title="Frozen-smoother RMSE")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(self.title_prefix_rep)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_epoch(self):
    """Save the standard public diagnostics for selected epochs."""
    if not self.do_plot:
        return
    epoch_dir = self.run_root / f"epoch{self.epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    timestamp = self.ts_tag
    save_dir = str(self.plot_save_dir)
    observations = {
        state_index: self.y_meas[:, column]
        for column, state_index in enumerate(self.meas_idx)
    }
    try:
        plot_filter_smoother_uncertainty(
            t=self.t, ms=self.ms, Ps=self.Ps, ms_s=self.ms_s, Ps_s=self.Ps_s,
            true_states=self.true_states, observations_by_state=observations,
            state_indices=list(range(self.n_state)), state_labels=self.state_labels,
            title=f"{self.title_prefix_rep}: frozen smoother @ epoch {self.epoch}",
            show=False, save_dir=save_dir, save_prefix=self.artifact_name,
            drive_seq=self.drive_seq if self.system_name == "retina" else None,
            epoch=self.epoch, epoch_timestamp=timestamp,
        )
    except Exception as exc:
        print(f"[viz] filter/smoother plot failed: {exc}")

    default_horizon = min(len(self.t) - 1, 200)
    plot_horizon = min(
        len(self.t) - 1,
        max(1, int(self.args.plot_rollout_steps or default_horizon)),
    )
    long_horizon = min(
        len(self.t) - 1,
        max(plot_horizon, int(self.args.long_rollout_steps or 500)),
    )
    plot_stride = max(1, int(self.args.plot_stride or 2))
    max_rollouts = max(1, int(self.args.plot_max_rollouts or 10))
    max_plot_dt = float(self.args.plot_max_dt or 0.1)
    use_reference_solver = self.args.plot_stiff_solver.lower() in {"bdf", "radau"}
    rollout_kwargs = dict(
        t=self.t, smoothed_states=self.ms_s_rollout, integrator=self.integrator,
        state_indices=list(range(self.n_state)), state_labels=self.state_labels,
        true_states=self.true_states, Ps_s=self.Ps_s, stride=plot_stride,
        max_rollouts=max_rollouts, show=False, save_dir=save_dir,
        save_prefix=self.artifact_name,
        drive_seq=self.drive_seq if self.system_name == "retina" else None,
        epoch=self.epoch, epoch_timestamp=timestamp, rollout_color_override="blue",
        physics_integrator=self.integrator.physics, physics_system_name=self.system_name,
        physics_system_cfg=self.system_cfg,
        physics_use_reference_solver=use_reference_solver,
        hybrid_use_reference_solver=use_reference_solver,
        reference_solver_method=self.args.plot_stiff_solver,
        rollout_label="learned rollout (from smoother IC)",
        physics_rollout_label="mechanistic model (from smoother IC)",
        max_plot_dt=max_plot_dt,
    )
    try:
        plot_rollouts(
            horizon_steps=plot_horizon,
            title=f"{self.title_prefix_rep}: rollouts @ epoch {self.epoch}",
            **rollout_kwargs,
        )
        if long_horizon > plot_horizon:
            plot_rollouts(
                horizon_steps=long_horizon,
                title=f"{self.title_prefix_rep}: long rollouts @ epoch {self.epoch}",
                allow_extrapolation=True,
                **rollout_kwargs,
            )
    except Exception as exc:
        print(f"[viz] rollout plot failed: {exc}")
    if not self.args.skip_full_ode_plot:
        try:
            plot_model_vs_true_ode(
                t=self.t, integrator=self.integrator,
                physics_integrator=self.integrator.physics,
                state_indices=list(range(self.n_state)), state_labels=self.state_labels,
                smoothed_states=self.ms_s_rollout, true_states=self.true_states,
                init_from="smoother",
                title=f"{self.title_prefix_rep}: learned vs mechanistic ODE @ epoch {self.epoch}",
                show=False, save_dir=save_dir, save_prefix=self.artifact_name,
                epoch=self.epoch, epoch_timestamp=timestamp,
            )
        except Exception as exc:
            print(f"[viz] model comparison failed: {exc}")
    try:
        _plot_training_curve(self, epoch_dir)
    except Exception as exc:
        print(f"[viz] loss curve failed: {exc}")


def _checkpoint_payload(self):
    return {
        "latent_net_state_dict": self.latent_net.state_dict(),
        "system_name": self.system_name,
        "nn_config": self.nn_cfg,
        "integrator_config": self.ic_cfg,
        "ckf_config": self.ck_cfg,
        "data_config": self.data_config,
        "system_config": self.system_cfg,
        "meas_idx": self.meas_idx,
        "state_labels": self.state_labels,
        "run_tag": self.ts_tag,
        "run_root": str(self.run_root),
        "epoch": self.epoch,
        "replaced_indices": list(self.learned_idx),
        "optimizer_state_dict": _sanitized_optimizer_state(self.optimizer),
        "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        "ms": self.ms.detach().cpu(),
        "Ps": self.Ps.detach().cpu(),
        "ms_s": self.ms_s.detach().cpu(),
        "Ps_s": self.Ps_s.detach().cpu(),
        "t": self.t.detach().cpu(),
        "y_meas": self.y_meas.detach().cpu(),
        "true_states": self.true_states.detach().cpu() if self.true_states is not None else None,
        "drive_seq": self.drive_seq.detach().cpu() if self.drive_seq is not None else None,
        "prior_vec": list(self.prior_vec) if self.prior_vec else [],
        "train_config": {
            "min_rollout_horizon": self.min_h,
            "max_rollout_horizon": self.max_h,
            "stride": self.stride,
            "tbptt_steps": self.tbptt_steps,
            "use_pseudo_latent": self.use_pseudo_latent,
            "pseudo_weight": self.pseudo_weight,
            "full_state_weight": self.full_state_weight,
        },
        "loss_mode": self.loss_mode,
        "title_prefix": self.title_prefix,
        "lr": float(self.optimizer.param_groups[0]["lr"]),
        "weight_decay": self.weight_decay,
        "grad_clip": self.grad_clip,
    }


def save_epoch(self):
    """Save self-contained checkpoints at regular intervals and the final epoch."""
    if not self.should_save_checkpoint:
        self.last_completed_epoch = self.epoch
        return
    epoch_dir = self.run_root / f"epoch{self.epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = epoch_dir / "checkpoint.pt"
    torch.save(_checkpoint_payload(self), checkpoint_path)
    plot_artifacts = sorted(
        path.name for path in epoch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".npz"}
    )
    (epoch_dir / "metrics.json").write_text(
        json.dumps(
            {
                "epoch": self.epoch, "loss": self.final_loss,
                "total_rmse": self.total_rmse, "horizon": self.horizon_steps,
                "duration_s": self.dur, "plots_requested": not self.args.no_plots,
                "plot_artifacts": plot_artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  [checkpoint] epoch {self.epoch:03d} -> {checkpoint_path}")
    self.last_completed_epoch = self.epoch


def finalize(self):
    """Report the completed continuation run and return its final metrics."""
    if self.saw_nonfinite:
        print(f"\n[FromSmoother] completed with skipped epochs/passes: {self.nonfinite_epochs}")
    else:
        print("\n[FromSmoother] training done.")
    final_dir = self.run_root / f"epoch{self.last_completed_epoch:03d}"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        _plot_training_curve(self, final_dir)
    except Exception as exc:
        print(f"[warn] Could not save final training curve: {exc}")
    print(f"Final loss={self.final_loss:.6g} RMSE={self.final_total_rmse:.6g}")
    print(f"Artifacts: {self.run_root}")
    return self.final_loss, self.final_total_rmse

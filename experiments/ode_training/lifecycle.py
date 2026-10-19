"""Run creation, data preparation, and initial smoother diagnostics."""

from __future__ import annotations

from .environment import *
from .configuration import *
from .data import *
from .systems import *


def prepare_run(self):
    """Create run metadata and checkpoint helpers."""
    import shutil

    self.artifact_system_name = _artifact_system_name(self.exp)
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.backup_dir = Path("code_backups") / self.artifact_system_name / self.timestamp
    self.backup_enabled = not _arg_bool(self.exp, "no_code_backup_arg", False)
    if self.backup_enabled:
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            py_files = [
                "ode_experiments.py", "integrate_saved_model.py", "ckf.py",
                "loss_rollout.py", "nn_ode_models.py", "viz_ode.py",
                "models/__init__.py", "models/base.py", "models/_template.py",
                "models/ho.py", "models/pendulum.py", "models/hh.py",
                "models/brusselator.py", "models/yeast.py", "models/retina.py",
            ]
            for filename in py_files:
                source = Path(filename)
                if source.exists():
                    destination = self.backup_dir / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            print(f"Code backup saved to: {self.backup_dir}")
        except Exception as exc:
            self.backup_enabled = False
            print(f"[warn] code backup disabled: {exc}")

    self.loss_mode = str(self.loss_mode).strip().lower()
    if self.loss_mode not in ("standard", "cov_weighted"):
        raise ValueError("loss_mode must be 'standard' or 'cov_weighted'")
    self.guidance_label = "RTS" if self.use_rts_guidance else "CKF"
    print(f"[Train] loss_mode={self.loss_mode} guidance={self.guidance_label}")
    def _checkpoint_payload(epoch_num=None):
        replaced = _get_replaced_indices(self.exp)
        return {
            "latent_net_state_dict": self.latent_net.state_dict(),
            "system_name": self.exp.system_name,
            "nn_config": {
                "in_dim": self.nn_in_dim,
                "out_dim": self.nn_out_dim,
                "hidden_dims": self.nnc.hidden_dims,
                "activation": self.nnc.activation,
                "input_scale": self.input_scale.tolist() if isinstance(self.input_scale, torch.Tensor) else self.nnc.input_scale,
                "input_shift": self.input_shift.tolist() if isinstance(self.input_shift, torch.Tensor) else self.nnc.input_shift,
            },
            "integrator_config": {
                "default_dt": self.ic.default_dt,
                "max_dt": self.ic.max_dt,
                "min_dt": self.ic.min_dt,
                "n_substeps": self.ic.n_substeps,
            },
            "ckf_config": {
                "Q_diag": self.ck.Q_diag,
                "R_diag": self.ck.R_diag,
                "P0_diag": self.ck.P0_diag,
            },
            "data_config": _data_config_payload(getattr(self.exp, "data_cfg", None)),
            "system_config": self.exp.system_cfg,
            "meas_idx": list(self.meas_idx),
            "state_labels": list(self.state_labels),
            "run_tag": self.ts_tag,
            "run_root": str(_safe_resolve_path(self.run_root)),
            "epoch": epoch_num,
            "replaced_indices": list(replaced),
        }

    self._checkpoint_payload = _checkpoint_payload

    def _save_epoch_checkpoint(epoch_num, ms_epoch=None, Ps_epoch=None, ms_s_epoch=None, Ps_s_epoch=None):
        epoch_dir = self.run_root / f"epoch{epoch_num:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = epoch_dir / "checkpoint.pt"
        payload = self._checkpoint_payload(epoch_num)
        payload["optimizer_state_dict"] = self.optimizer.state_dict()
        payload["scaler_state_dict"] = self.scaler.state_dict() if self.scaler.is_enabled() else None
        if ms_s_epoch is not None:
            payload.update(
                ms=ms_epoch.detach().cpu(), Ps=Ps_epoch.detach().cpu(),
                ms_s=ms_s_epoch.detach().cpu(), Ps_s=Ps_s_epoch.detach().cpu(),
            )
        payload.update(
            t=self.t.detach().cpu(),
            y_meas=self.y_meas.detach().cpu(),
            true_states=self.true_states.detach().cpu() if self.true_states is not None else None,
            drive_seq=self.drive_seq.detach().cpu() if self.drive_seq is not None else None,
            prior_vec=list(self.prior_vec),
            train_config={
                "T_train": float(self.tc.T_train),
                "n_epochs": int(self.tc.n_epochs),
                "min_rollout_horizon": int(self.tc.min_rollout_horizon),
                "max_rollout_horizon": int(self.tc.max_rollout_horizon),
                "stride": int(self.tc.stride),
                "use_pseudo_latent": bool(self.tc.use_pseudo_latent),
                "pseudo_weight": float(self.tc.pseudo_weight),
                "meas_noise_std": float(self.tc.meas_noise_std),
                "full_state_weight": float(getattr(self.tc, "full_state_weight", 0.0)),
            },
            loss_mode=self.loss_mode,
            title_prefix=self.title_prefix,
            lr=float(self.optimizer.param_groups[0]["lr"]),
            weight_decay=float(self.nnc.weight_decay),
            grad_clip=float(self.nnc.grad_clip),
        )
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    self._save_epoch_checkpoint = _save_epoch_checkpoint

    def _save_epoch_smoother_outputs(epoch_num, ms_epoch, Ps_epoch, ms_s_epoch, Ps_s_epoch):
        epoch_dir = self.run_root / f"epoch{epoch_num:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        output_path = epoch_dir / "smoother_outputs.pt"
        torch.save(
            {
                "epoch": int(epoch_num), "t": self.t.detach().cpu(),
                "measurements": self.y_meas.detach().cpu(),
                "true_states": self.true_states.detach().cpu() if self.true_states is not None else None,
                "ms": ms_epoch.detach().cpu(), "Ps": Ps_epoch.detach().cpu(),
                "ms_s": ms_s_epoch.detach().cpu(), "Ps_s": Ps_s_epoch.detach().cpu(),
                "state_labels": list(self.state_labels), "meas_idx": list(self.meas_idx),
                "run_tag": self.ts_tag, "system_name": self.exp.system_name,
            },
            output_path,
        )
        return output_path

    self._save_epoch_smoother_outputs = _save_epoch_smoother_outputs
    if getattr(self.exp.train_cfg, "meas_snr", None):
        snr = self.exp.train_cfg.meas_snr
        if f"_SNR{snr:g}" not in self.exp.title_prefix:
            self.exp.title_prefix = f"{self.exp.title_prefix}_SNR{snr:g}"


def prepare_data(self):
    """Generate a benchmark data set and create its run directory."""
    self.ic, self.ck = self.exp.integrator_cfg, self.exp.ckf_cfg
    self.nnc, self.tc, self.pc = self.exp.nn_cfg, self.exp.train_cfg, self.exp.plot_cfg
    self.dc = self.exp.data_cfg
    self.adaptive_time_grid = bool(getattr(self.dc, "keep_adaptive_grid", False))
    if self.adaptive_time_grid:
        self.dc.uniform_dt = 0.0
        print("[Data] Adaptive time grid enabled")

    (
        self.physics, self.data_bundle, self.state_labels, self.title_prefix,
        self.prior_vec, self.meas_idx, self.nn_in_dim, self.drive_seq,
    ) = _choose_system(self.exp)
    self.run_tag = _make_run_tag(self.exp, self.tc)
    snr_suffix = f"_SNR{self.tc.meas_snr:g}" if getattr(self.tc, "meas_snr", None) else ""
    self.ts_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_random_tag3()}{snr_suffix}"

    if self.exp.system_name in ("retina", "hh", "brusselator", "yeast"):
        self.t, self.meas, self.s0_true, self.s1_true, self.true_states = self.data_bundle
    else:
        self.t, self.meas, self.s0_true, self.s1_true = self.data_bundle
        self.true_states = torch.stack([self.s0_true, self.s1_true], dim=-1)

    self.replaced_label = _get_replaced_state_label(self.exp, self.state_labels)
    self.title_prefix_rep = (
        f"{self.title_prefix} [replaced: {self.replaced_label}]"
        if self.replaced_label else self.title_prefix
    )
    self.nn_out_dim = int(getattr(self.nnc, "out_dim", 1))
    self.run_root = Path("results") / self.artifact_system_name / self.ts_tag
    self.run_root.mkdir(parents=True, exist_ok=True)
    _save_step_size_diagnostic(self.t, self.run_root, title=self.title_prefix_rep)
    _save_generated_data_diagnostic(
        t=self.t,
        true_states=self.true_states,
        meas=self.meas,
        meas_idx=self.meas_idx,
        state_labels=self.state_labels,
        out_dir=self.run_root,
        title=f"{self.title_prefix_rep}: generated data",
        drive_seq=self.drive_seq,
    )
    log_override = str(_arg_value(self.exp, "training_log_file_arg", "") or "").strip()
    self.log_file = Path(log_override) if log_override else Path("results") / "logs" / f"{self.artifact_system_name}_training_log.csv"
    self.log_file.parent.mkdir(parents=True, exist_ok=True)
    plot_dataset_ode(
        t=self.t, true_states=self.true_states, meas=self.meas,
        meas_idx=self.meas_idx, state_labels=self.state_labels,
        title=f"{self.title_prefix_rep}: dataset (full ODE)", show=True,
        save_dir="results", save_prefix=self.artifact_system_name,
        drive_seq=self.drive_seq, epoch_timestamp=self.ts_tag,
    )


def initial_rollout(self):
    """Run the initial filter/smoother pass and save epoch-zero artifacts."""
    print("\n[Epoch 0] Generating initial rollouts with initialized network...")
    self.epoch_timestamp = self.ts_tag
    self.use_true_rollouts = self.rollouts_from_true
    self.nonneg_idx = None
    with torch.no_grad():
        kwargs = {"drive_seq": self.drive_seq} if self.drive_seq is not None else {}
        print("[Epoch 0] CKF begin")
        started = time.perf_counter()
        self.ms, self.Ps = ckf_filter(
            self.smoother_integrator, self.m0, self.P0, self.y_meas, self.t,
            nonneg_indices=self.nonneg_idx, **kwargs,
        )
        print(f"[Epoch 0] CKF done in {time.perf_counter() - started:.2f}s")
        self.ms_s, self.Ps_s, guide_time = self._run_guidance(
            self.smoother_integrator, self.ms, self.Ps, t_seq=self.t,
            drive_seq_arg=self.drive_seq, nonneg_indices_arg=self.nonneg_idx,
        )
        print(f"[Epoch 0] Guidance done in {guide_time:.2f}s")
        checkpoint = self._save_epoch_checkpoint(0, self.ms, self.Ps, self.ms_s, self.Ps_s)
        print(f"  [checkpoint] saved epoch 000 -> {checkpoint}")

    self.base_plot_horizon = self.tc.max_rollout_horizon * 9 if self.tc.max_rollout_horizon > 0 else 90
    self.plot_horizon = compute_plot_horizon_steps(len(self.t), self.base_plot_horizon)
    self.rollout_seed_states = self.true_states if self.use_true_rollouts else self.ms_s
    plot_rollouts(
        t=self.t, smoothed_states=self.rollout_seed_states, integrator=self.integrator,
        state_indices=list(range(self.true_states.shape[1])), state_labels=self.state_labels,
        true_states=self.true_states, Ps_s=self.Ps_s, horizon_steps=self.plot_horizon,
        stride=self.pc.plot_stride, max_rollouts=self.pc.max_rollouts,
        title=f"{self.title_prefix_rep}: rollouts @ epoch 0 (initialized)", show=True,
        save_dir="results", save_prefix=self.artifact_system_name,
        drive_seq=self.drive_seq, epoch=0, epoch_timestamp=self.epoch_timestamp,
    )
    print("[Epoch 0] Initial rollout visualization complete.\n")

    self.freeze_smoother = _arg_bool(self.exp, "freeze_smoother_arg", False)
    self.frozen_smoother_loaded = False
    self.frozen_smoother_epoch = 0
    self.smoother_every = max(1, _arg_int(self.exp, "smoother_every_arg", 1))
    self.last_smoother_epoch = 0
    self.last_ms, self.last_Ps = self.ms, self.Ps
    self.last_ms_s, self.last_Ps_s = self.ms_s, self.Ps_s
    self.timing_sync = device.type == "cuda" and _arg_bool(self.exp, "timing_sync_arg", False)

    def _sync():
        if self.timing_sync:
            torch.cuda.synchronize()

    self._sync = _sync
    self.smoother_amp = device.type == "cuda" and _arg_bool(self.exp, "smoother_amp_arg", False)
    self.smoother_inference = _arg_bool(self.exp, "smoother_inference_arg", True)
    self.final_loss = float("nan")
    self.final_total_rmse = float("nan")
    self.final_state_errors = {}
    self.final_phase_errors = None
    self.continue_on_nonfinite_loss = _arg_bool(self.exp, "continue_on_nonfinite_loss_arg", True)
    self.saw_nonfinite_loss = False
    self.nonfinite_epochs = []
    self.last_completed_epoch = 0

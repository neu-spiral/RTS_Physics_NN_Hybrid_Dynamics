"""Load and validate a frozen-smoother checkpoint."""

from .support import *
from .configuration import apply_model_continuation_config


def load_checkpoint(self):
    """Load model metadata and frozen state targets without rerunning CKF/RTS."""
    self.checkpoint_path = _resolve_checkpoint_path(self.checkpoint_path)
    requested = (self.args.device or "").strip().lower()
    if self.args.cpu:
        self.device = torch.device("cpu")
    elif requested in ("", "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        self.device = torch.device(requested)
    print(f"[Device] {self.device}")

    _install_checkpoint_compat_modules()
    self.ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
    self.system_name = str(self.ckpt["system_name"]).lower()
    if self.system_name not in {"ho", "pendulum", "hh", "brusselator", "yeast", "retina"}:
        raise ValueError(f"Unsupported public checkpoint system: {self.system_name}")
    self.model_continuation_config = apply_model_continuation_config(self.args, self.system_name)
    self.nn_cfg = dict(self.ckpt["nn_config"])
    self.ic_cfg = dict(self.ckpt["integrator_config"])
    self.ck_cfg = dict(self.ckpt["ckf_config"])
    self.system_cfg = self.ckpt["system_config"]
    self.meas_idx = list(self.ckpt["meas_idx"])
    self.state_labels = list(self.ckpt["state_labels"])
    self.loaded_epoch = int(self.ckpt.get("epoch", 0) or 0)
    self.train_config = dict(self.ckpt.get("train_config") or {})
    self.data_config = dict(self.ckpt.get("data_config") or {})

    self.nn_arch_overridden = False
    if self.model_continuation_config.nn_linear_layers is not None:
        target_layers = int(self.model_continuation_config.nn_linear_layers)
        previous = self.nn_cfg.get("hidden_dims")
        self.nn_cfg["hidden_dims"] = _expand_hidden_dims_for_linear_layers(previous or (32, 16), target_layers)
        self.nn_arch_overridden = True
        print(f"[FromSmoother] NN architecture override: {previous} -> {self.nn_cfg['hidden_dims']}")

    self.rollout_max_dt = self.args.rollout_max_dt
    if self.rollout_max_dt is not None:
        self.rollout_max_dt = float(self.rollout_max_dt)
        if self.rollout_max_dt <= 0.0:
            raise ValueError("--rollout-max-dt must be positive.")
        previous = self.ic_cfg.get("max_dt")
        self.ic_cfg["max_dt"] = self.rollout_max_dt
        print(f"[FromSmoother] Rollout max_dt override: {previous} -> {self.rollout_max_dt}")

    self.ms_s = self.ckpt["ms_s"].to(self.device)
    self.Ps_s = self.ckpt["Ps_s"].to(self.device)
    self.ms = self.ckpt.get("ms", self.ms_s).to(self.device)
    self.Ps = self.ckpt.get("Ps", self.Ps_s).to(self.device)
    self.t = self.ckpt["t"].to(self.device)
    self.y_meas = self.ckpt["y_meas"].to(self.device)
    self.true_states = self.ckpt.get("true_states")
    if self.true_states is not None:
        self.true_states = self.true_states.to(self.device)
    self.drive_seq = self.ckpt.get("drive_seq")
    if self.drive_seq is not None:
        self.drive_seq = self.drive_seq.to(self.device)
    self.prior_vec = self.ckpt.get("prior_vec", [])
    print(f"[Load] checkpoint={self.checkpoint_path}")
    print(f"[Load] system={self.system_name} epoch={self.loaded_epoch} state_labels={self.state_labels} meas_idx={self.meas_idx}")
    print("[FromSmoother] using loaded state estimates as frozen targets")

    smoother_checkpoint = getattr(self.args, "smoother_checkpoint", None)
    if smoother_checkpoint:
        path = _resolve_checkpoint_path(smoother_checkpoint)
        override = torch.load(str(path), map_location="cpu", weights_only=False)
        if str(override.get("system_name", self.system_name)).lower() != self.system_name:
            raise ValueError("--smoother-checkpoint system mismatch")
        self.t = override["t"].to(self.device)
        self.ms_s = override["ms_s"].to(self.device)
        self.Ps_s = override["Ps_s"].to(self.device)
        self.ms = override.get("ms", self.ms_s).to(self.device)
        self.Ps = override.get("Ps", self.Ps_s).to(self.device)
        self.y_meas = override.get("y_meas", self.y_meas).to(self.device)
        self.true_states = override.get("true_states", self.true_states)
        if self.true_states is not None:
            self.true_states = self.true_states.to(self.device)
        self.drive_seq = override.get("drive_seq", self.drive_seq)
        if self.drive_seq is not None:
            self.drive_seq = self.drive_seq.to(self.device)
        self.loaded_epoch = int(override.get("epoch", self.loaded_epoch) or self.loaded_epoch)
        print(f"[SmootherOverride] loaded frozen targets from {path}")

    self.t_full_for_plot = self.t.detach().clone()
    self.y_meas_full_for_plot = self.y_meas.detach().clone()
    self.ms_full_for_plot = self.ms.detach().clone()
    self.Ps_full_for_plot = self.Ps.detach().clone()
    self.ms_s_full_for_plot = self.ms_s.detach().clone()
    self.Ps_s_full_for_plot = self.Ps_s.detach().clone()
    self.true_states_full_for_plot = self.true_states.detach().clone() if self.true_states is not None else None

    crop_start = getattr(self.args, "crop_start", None)
    crop_end = getattr(self.args, "crop_end", None)
    if crop_start is not None or crop_end is not None:
        start = float(self.t.min().item()) if crop_start is None else float(crop_start)
        end = float(self.t.max().item()) if crop_end is None else float(crop_end)
        mask = (self.t >= start) & (self.t <= end)
        if int(mask.sum().item()) < 3:
            raise RuntimeError(f"Crop window [{start}, {end}] contains fewer than three samples")
        for name in ("t", "ms_s", "Ps_s", "ms", "Ps", "y_meas"):
            setattr(self, name, getattr(self, name)[mask])
        if self.true_states is not None:
            self.true_states = self.true_states[mask]
        if self.drive_seq is not None:
            self.drive_seq = self.drive_seq[mask]
        print(f"[CROP] training on [{start}, {end}] with {int(mask.sum().item())} samples")
    else:
        print(f"[CROP] disabled; using {self.t.shape[0]} samples")

"""Rebuild the continuation model and configure its training run."""

from .support import *


def build_model(self):
    """Recreate physics, learned network, and hybrid integrator."""
    class _Namespace:
        def __init__(namespace_self, values):
            for key, value in values.items():
                setattr(namespace_self, key, value)

    self.ic_ns = _Namespace(self.ic_cfg)
    self.ck_ns = _Namespace(self.ck_cfg)
    self.physics = _build_physics(
        self.system_name, self.system_cfg, self.ic_ns, self.ck_ns, self.device
    )
    self.input_scale = (
        torch.tensor(self.nn_cfg["input_scale"], device=self.device, dtype=torch.float32)
        if self.nn_cfg.get("input_scale") else None
    )
    self.input_shift = (
        torch.tensor(self.nn_cfg["input_shift"], device=self.device, dtype=torch.float32)
        if self.nn_cfg.get("input_shift") else None
    )
    self.latent_net = build_scripted_state_mlp(
        in_dim=self.nn_cfg["in_dim"],
        out_dim=self.nn_cfg["out_dim"],
        hidden_dims=self.nn_cfg["hidden_dims"],
        activation=self.nn_cfg["activation"],
        input_scale=self.input_scale,
        input_shift=self.input_shift,
        device=self.device,
    )
    self._fresh_nn_init = self.nn_arch_overridden or not _load_compatible_state_dict(
        self.latent_net, self.ckpt.get("latent_net_state_dict"), "FromSmoother"
    )
    self.integrator = _build_integrator(
        self.system_name, self.physics, self.latent_net,
        self.ic_cfg, self.system_cfg, self.device,
    )
    print(
        f"       NN ready ({sum(parameter.numel() for parameter in self.latent_net.parameters())} params) "
        f"source={'fresh_random' if self._fresh_nn_init else 'checkpoint'}"
    )


def configure_training(self):
    """Configure continuation objectives, optimizer state, and artifact paths."""
    self.lr = self.args.lr if self.args.lr is not None else float(self.ckpt.get("lr", 0.001))
    self.weight_decay = float(self.ckpt.get("weight_decay", 0.0))
    self.grad_clip = float(self.ckpt.get("grad_clip", 10.0))
    self.params = [parameter for parameter in self.latent_net.parameters() if parameter.requires_grad]
    self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
    optimizer_state = self.ckpt.get("optimizer_state_dict")
    if self._fresh_nn_init and optimizer_state is not None:
        print("       [warn] Skipping optimizer state because the network was reinitialized.")
    elif _optimizer_state_has_nonfinite(optimizer_state):
        print("       [warn] Skipping non-finite optimizer state.")
    elif not self.args.no_resume_optimizer and optimizer_state is not None:
        try:
            self.optimizer.load_state_dict(optimizer_state)
            if self.args.lr is not None:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.args.lr
            print(f"       Optimizer state restored (lr={self.optimizer.param_groups[0]['lr']:.2e})")
        except Exception as exc:
            print(f"       [warn] Could not restore optimizer state: {exc}")

    self.amp_enabled = self.device.type == "cuda" and os.getenv("ODE_AMP", "0") == "1"
    try:
        self.scaler = GradScaler(device_type=self.device.type, enabled=self.amp_enabled)
    except TypeError:
        self.scaler = GradScaler(enabled=self.amp_enabled)
    if self.ckpt.get("scaler_state_dict") is not None and self.amp_enabled:
        try:
            self.scaler.load_state_dict(self.ckpt["scaler_state_dict"])
        except Exception:
            pass

    self.loss_mode = self.args.loss_mode or self.ckpt.get("loss_mode", "standard")
    self.base_loss_fn = (
        cov_weighted_multi_step_rollout_loss
        if self.loss_mode == "cov_weighted" else multi_step_rollout_loss
    )
    self.loss_fn = self.base_loss_fn
    self.compile_enabled = (
        self.device.type == "cuda"
        and hasattr(torch, "compile")
        and os.getenv("ODE_TORCH_COMPILE", "0") == "1"
    )
    if self.compile_enabled:
        try:
            backend = "inductor" if sys.platform.startswith("linux") else "aot_eager"
            self.loss_fn = torch.compile(self.base_loss_fn, mode="reduce-overhead", backend=backend)
        except Exception as exc:
            print(f"[Compile] unavailable; using eager loss: {exc}")
            self.loss_fn = self.base_loss_fn

    self.n_epochs = int(self.args.n_epochs)
    self.min_h = int(
        self.args.min_horizon
        if self.args.min_horizon is not None
        else self.train_config.get("min_rollout_horizon", 5)
    )
    self.max_h = int(
        self.args.max_horizon
        if self.args.max_horizon is not None
        else self.train_config.get("max_rollout_horizon", 30)
    )
    self.min_h = max(1, min(self.min_h, self.max_h))
    self.stride = int(
        self.args.stride if self.args.stride is not None else self.train_config.get("stride", 1)
    )
    self.tbptt_steps = max(1, int(self.args.tbptt_steps or 5))
    self.full_state_weight = max(
        0.0,
        float(
            self.args.full_state_weight
            if self.args.full_state_weight is not None
            else self.train_config.get("full_state_weight", 0.0)
        ),
    )
    self.pseudo_weight = max(
        0.0,
        float(
            self.args.pseudo_weight
            if self.args.pseudo_weight is not None
            else self.train_config.get("pseudo_weight", 0.0)
        ),
    )
    self.use_pseudo_latent = self.pseudo_weight > 0.0
    self.learned_idx = _get_learned_idx(self.system_name, self.system_cfg)
    self.replaced_label = _get_replaced_label(self.system_name, self.state_labels, self.system_cfg)
    self.title_prefix = self.ckpt.get("title_prefix", self.system_name.upper())
    self.title_prefix_rep = f"{self.title_prefix} [replaced: {self.replaced_label}]"
    self.n_state = self.ms_s.shape[1]
    self.ms_s_rollout = self.ms_s

    self.run_suffix = str(getattr(self.args, "run_suffix", "") or "").strip()
    self.artifact_root = _normalize_repo_path(self.args.artifact_root)
    self.resume_run_root = os.getenv("ODE_CONTINUATION_RUN_ROOT", "").strip()
    if self.resume_run_root:
        self.run_root = Path(self.resume_run_root).resolve()
        self.ts_tag = self.run_root.name
    else:
        suffix = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in self.run_suffix
        )
        tag = suffix or _random_tag3()
        self.ts_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}_fromsmoother"
        self.run_root = self.artifact_root / "results" / self.system_name / self.ts_tag
    self.run_root.mkdir(parents=True, exist_ok=True)
    self.artifact_name = self.system_name
    if (
        self.run_root.parent.name == self.system_name
        and self.run_root.parent.parent.name == "results"
    ):
        self.plot_save_dir = self.run_root.parent.parent
    else:
        self.plot_save_dir = self.artifact_root / "results"
    self.log_file = self.run_root / "training_log.csv"
    print(
        f"\n[FromSmoother] Training {self.n_epochs} epochs lr={self.lr:.2e} "
        f"horizon=[{self.min_h},{self.max_h}] stride={self.stride}\n"
        f"               loss_mode={self.loss_mode} device={self.device} amp={self.amp_enabled}\n"
        f"               output -> {self.run_root}\n"
    )


def prepare_epochs(self):
    """Set epoch numbering and install finite-parameter rollback guards."""
    self.epoch_offset = max(0, int(os.getenv("ODE_CONTINUATION_EPOCH_OFFSET", "0")))
    if self.epoch_offset:
        self.n_epochs += self.epoch_offset
    self.plot_every = max(1, int(os.getenv("ODE_PLOT_EVERY", "5")))
    self.log_every = max(1, int(os.getenv("ODE_LOG_EVERY", "1")))
    self.checkpoint_every = 5
    self._guarded_params = list(self.latent_net.parameters())

    def _snapshot_params():
        return [parameter.detach().clone() for parameter in self._guarded_params]

    def _restore_params(snapshot):
        with torch.no_grad():
            for parameter, value in zip(self._guarded_params, snapshot):
                parameter.copy_(value)

    def _params_finite():
        return all(
            not parameter.is_floating_point() or torch.isfinite(parameter).all().item()
            for parameter in self._guarded_params
        )

    self._snapshot_params = _snapshot_params
    self._restore_params = _restore_params
    self._params_finite = _params_finite
    self.final_loss = float("nan")
    self.final_total_rmse = float("nan")
    self.saw_nonfinite = False
    self.nonfinite_epochs = []
    self.last_completed_epoch = 0

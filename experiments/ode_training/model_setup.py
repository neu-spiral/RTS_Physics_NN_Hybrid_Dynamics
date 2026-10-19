"""Build hybrid models, optimizers, and CKF/RTS guidance wrappers."""

from __future__ import annotations

from .environment import *
from .configuration import *
from .data import *


def build_network(self):
    """Build the learned derivative and the system-specific hybrid integrator."""
    self.input_scale = (
        torch.tensor(self.nnc.input_scale, device=device, dtype=torch.float32)
        if self.nnc.input_scale else None
    )
    self.input_shift = (
        torch.tensor(self.nnc.input_shift, device=device, dtype=torch.float32)
        if self.nnc.input_shift else None
    )
    self.latent_net = build_scripted_state_mlp(
        in_dim=self.nn_in_dim,
        out_dim=self.nn_out_dim,
        hidden_dims=self.nnc.hidden_dims,
        activation=self.nnc.activation,
        input_scale=self.input_scale,
        input_shift=self.input_shift,
        device=device,
    )
    if self.exp.system_name == "retina":
        with torch.no_grad():
            last_layer = None
            for module in self.latent_net.net.modules():
                if isinstance(module, torch.nn.Linear):
                    last_layer = module
            if last_layer is not None:
                last_layer.weight.zero_()
                last_layer.bias.zero_()
        replaced = tuple(
            int(i) for i in getattr(
                self.exp.system_cfg, "replaced_indices", getattr(self.exp, "pseudo_idx", (2,))
            )
        )
        self.integrator = LearnedODERetina4WithPin(
            physics=self.physics,
            latent_model=self.latent_net,
            max_dt=self.ic.max_dt,
            min_dt=self.ic.min_dt,
            replaced_index=replaced[0],
            replaced_indices=replaced,
        ).to(device)
    elif self.exp.system_name == "hh":
        self.integrator = LearnedODE4D(
            physics=self.physics, latent_model=self.latent_net,
            max_dt=self.ic.max_dt, min_dt=self.ic.min_dt, replaced_index=1,
        ).to(device)
    elif self.exp.system_name == "brusselator":
        self.integrator = LearnedODE3D(
            physics=self.physics, latent_model=self.latent_net,
            max_dt=self.ic.max_dt, min_dt=self.ic.min_dt, replaced_index=0,
        ).to(device)
    elif self.exp.system_name == "yeast":
        self.integrator = LearnedODEYeast(
            physics=self.physics, latent_model=self.latent_net,
            max_dt=self.ic.max_dt, min_dt=self.ic.min_dt, replaced_index=3,
        ).to(device)
    else:
        self.integrator = LearnedODE2D(
            physics=self.physics, latent_model=self.latent_net,
            max_dt=self.ic.max_dt, min_dt=self.ic.min_dt,
        ).to(device)

    init_path = str(_arg_value(self.exp, "init_checkpoint_arg", "") or "").strip()
    if init_path:
        checkpoint = torch.load(init_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("latent_net_state_dict", checkpoint)
        self.latent_net.load_state_dict(state_dict, strict=False)
        print(f"[Init] loaded learned-derivative weights from {init_path}")


def configure_training(self):
    """Create optimizer, optional compilation, scheduling, and logging controls."""
    self.lr = self.nnc.lr[self.exp.system_name] if isinstance(self.nnc.lr, dict) else self.nnc.lr
    self.params = [parameter for parameter in self.latent_net.parameters() if parameter.requires_grad]
    self.optimizer = torch.optim.Adam(
        self.params, lr=self.lr, weight_decay=self.nnc.weight_decay
    )
    self.amp_enabled = device.type == "cuda" and _arg_bool(self.exp, "amp_arg", False)
    try:
        self.scaler = GradScaler(device_type=device.type, enabled=self.amp_enabled)
    except TypeError:
        self.scaler = GradScaler(enabled=self.amp_enabled)

    self.compile_enabled = (
        device.type == "cuda"
        and hasattr(torch, "compile")
        and _arg_bool(self.exp, "torch_compile_arg", False)
    )
    self.compile_backend = "inductor" if sys.platform.startswith("linux") else "aot_eager"
    self.base_loss_fn = (
        cov_weighted_multi_step_rollout_loss
        if self.loss_mode == "cov_weighted"
        else multi_step_rollout_loss
    )
    self.loss_fn = self.base_loss_fn
    if self.compile_enabled:
        try:
            self.loss_fn = torch.compile(
                self.base_loss_fn, mode="reduce-overhead", backend=self.compile_backend
            )
            print("[Compile] torch.compile enabled for rollout loss")
        except Exception as exc:
            print(f"[Compile] unavailable; using eager loss: {exc}")
            self.loss_fn = self.base_loss_fn
            self.compile_enabled = False
    self.integrator_for_loss = self.integrator
    self.scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )
        if self.exp.system_name == "retina" else None
    )
    self.plot_every = max(0, int(getattr(self.pc, "show_every", 1)))
    self.long_rollout_only_every = 5
    self.long_rollout_only_multiplier = 10
    self.log_every = max(1, _arg_int(self.exp, "log_every_arg", 1))
    self.skip_phase_portrait = _arg_bool(self.exp, "skip_phase_portrait_arg", True)


def initialize_smoother(self):
    """Initialize the filter from configured priors and observed channels only."""
    self.y_meas = self.meas if self.meas.dim() == 2 else self.meas.unsqueeze(-1)
    self.m0 = torch.tensor(self.prior_vec, device=device, dtype=torch.float32)
    self.y0 = self.y_meas[0]
    for k, idx in enumerate(self.meas_idx):
        self.m0[idx] = self.y0[k]
    self.P0 = torch.diag(torch.tensor(self.ck.P0_diag, device=device, dtype=torch.float32))
    self.Tn = self.t.shape[0]
    if self.Tn < 2:
        raise ValueError(f"t has too few points: {self.Tn}")
    self.global_max_h = max(1, self.Tn - 1)
    print(f"\n=== system={self.exp.system_name}  T={self.tc.T_train:.2f}  N={len(self.t)} ===")

    self.fast_smoother = _arg_bool(self.exp, "smoother_fast_arg", False)
    self.fast_method = str(_arg_value(self.exp, "smoother_fast_method_arg", "euler")).lower()
    if self.fast_method not in ("euler", "rk2"):
        self.fast_method = "euler"

    class _FastStepWrapper:
        def __init__(wrapper_self, base, method="euler"):
            wrapper_self.base = base
            wrapper_self.physics = base.physics
            wrapper_self.method = method

        @property
        def Q(wrapper_self):
            return wrapper_self.base.Q

        @property
        def R(wrapper_self):
            return wrapper_self.base.R

        def propagate_batch(wrapper_self, states, dt):
            if wrapper_self.method == "rk2":
                k1 = wrapper_self.base.f_full(states)
                k2 = wrapper_self.base.f_full(states + 0.5 * dt * k1)
                next_states = states + dt * k2
            else:
                next_states = states + dt * wrapper_self.base.f_full(states)
            if hasattr(wrapper_self.physics, "clip_states"):
                next_states = wrapper_self.physics.clip_states(next_states)
            return next_states

        def h(wrapper_self, states):
            if hasattr(wrapper_self.base, "h"):
                return wrapper_self.base.h(states)
            return wrapper_self.physics.h(states)

    self.smoother_integrator = (
        _FastStepWrapper(self.integrator, method=self.fast_method)
        if self.fast_smoother else self.integrator
    )

    def _run_guidance(base_integrator, means, covariances, *, t_seq, drive_seq_arg=None, nonneg_indices_arg=None):
        if not self.use_rts_guidance:
            print("[Guidance] RTS disabled; using CKF filter states as targets")
            return means.clone(), covariances.clone(), 0.0
        started = time.perf_counter()
        guided_means, guided_covariances = rts_smoother(
            base_integrator,
            means,
            covariances,
            t_seq,
            drive_seq=drive_seq_arg,
            nonneg_indices=nonneg_indices_arg,
        )
        return guided_means, guided_covariances, time.perf_counter() - started

    self._run_guidance = _run_guidance

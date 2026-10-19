"""Integrate a saved hybrid ODE checkpoint without CKF/RTS or training."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

import models as _models_pkg
from models import (
    BrusselatorPhysics,
    CKFConfig,
    HOPhysics,
    HodgkinHuxleyPhysics,
    IntegratorConfig,
    LearnedODE2D,
    LearnedODE3D,
    LearnedODE4D,
    LearnedODERetina4WithPin,
    LearnedODEYeast,
    PendulumPhysics,
    RetinaPhysics,
    YeastPhysics,
)
from nn_ode_models import build_scripted_state_mlp


def _install_checkpoint_compat_modules():
    """Let old checkpoints that referenced removed root modules unpickle cleanly."""
    if "ode_config" not in sys.modules:
        compat = types.ModuleType("ode_config")
        for name in dir(_models_pkg):
            setattr(compat, name, getattr(_models_pkg, name))
        sys.modules["ode_config"] = compat


def _as_config(config, config_type):
    """Return dataclass config objects from checkpoint dicts or existing objects."""
    if isinstance(config, config_type):
        return config
    if isinstance(config, dict):
        return config_type(**config)
    raise TypeError(f"Unsupported {config_type.__name__} payload: {type(config)}")


def _cfg_get_replaced_indices(system_cfg, default_idx: int) -> tuple[int, ...]:
    """Read one or more learned state indices from checkpoint system config."""
    raw = getattr(system_cfg, "replaced_indices", None)
    if raw is None:
        raw = (getattr(system_cfg, "replaced_index", default_idx),)
    if isinstance(raw, (int, float, str)):
        raw = (raw,)
    return tuple(int(i) for i in raw)


def _build_physics(system_name, system_cfg, integrator_cfg, ckf_cfg, device):
    """Rebuild the physics object saved checkpoints expect."""
    builders = {
        "ho": lambda: HOPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "pendulum": lambda: PendulumPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "hh": lambda: HodgkinHuxleyPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "brusselator": lambda: BrusselatorPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "retina": lambda: RetinaPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "yeast": lambda: YeastPhysics(system_cfg, integrator_cfg, ckf_cfg),
    }
    if system_name not in builders:
        raise ValueError(f"Unsupported system_name in checkpoint: {system_name}")
    return builders[system_name]().to(device)


def _build_integrator(system_name, physics, latent_net, integrator_cfg, system_cfg, device):
    """Rebuild the learned hybrid ODE wrapper around physics and NN weights."""
    max_dt = float(integrator_cfg.max_dt)
    min_dt = float(integrator_cfg.min_dt)
    if system_name == "retina":
        replaced = _cfg_get_replaced_indices(system_cfg, 2)
        return LearnedODERetina4WithPin(
            physics, latent_net, max_dt=max_dt, min_dt=min_dt,
            replaced_index=replaced[0], replaced_indices=replaced,
        ).to(device)
    if system_name == "hh":
        return LearnedODE4D(physics, latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=1).to(device)
    if system_name == "brusselator":
        return LearnedODE3D(physics, latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=0).to(device)
    if system_name == "yeast":
        return LearnedODEYeast(physics, latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=3).to(device)
    return LearnedODE2D(physics, latent_net, max_dt=max_dt, min_dt=min_dt).to(device)


def _parse_y0(raw: str | None, device) -> torch.Tensor | None:
    """Parse a comma-separated initial condition from the command line."""
    if raw is None or not raw.strip():
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _default_y0(ckpt, source: str, device) -> torch.Tensor:
    """Choose an initial condition from checkpoint tensors when --y0 is absent."""
    candidates = {
        "true": ("true_states",),
        "smoothed": ("ms_s", "smoothed_states"),
        "filtered": ("ms",),
        "prior": ("prior_vec",),
        "auto": ("true_states", "ms_s", "smoothed_states", "ms", "prior_vec"),
    }
    for key in candidates[source]:
        value = ckpt.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim >= 2:
            tensor = tensor[0]
        return tensor.flatten()
    raise ValueError("No initial condition found in checkpoint. Pass --y0.")


def _checkpoint_dt_and_duration(ckpt, requested_dt, requested_duration):
    """Use CLI values first, then infer dt/duration from checkpoint time if present."""
    t = ckpt.get("t")
    if t is not None:
        t = torch.as_tensor(t, dtype=torch.float32)
    if requested_dt is not None:
        dt = float(requested_dt)
    elif t is not None and t.numel() > 1:
        dt = float(torch.diff(t).mean().item())
    else:
        dt = 0.01
    if requested_duration is not None:
        duration = float(requested_duration)
    elif t is not None and t.numel() > 1:
        duration = float((t[-1] - t[0]).item())
    else:
        duration = 10.0
    return dt, duration


def _hidden_dims_from_config(nn_cfg) -> tuple[int, ...]:
    """Normalize hidden_dims from checkpoint metadata."""
    hidden_dims = nn_cfg.get("hidden_dims", (32, 32))
    if isinstance(hidden_dims, int):
        return (int(hidden_dims),)
    return tuple(int(v) for v in hidden_dims)


def _build_latent_net(ckpt, nn_cfg, device):
    """Create the learned derivative MLP and load checkpoint weights."""
    input_scale = torch.tensor(nn_cfg["input_scale"], device=device, dtype=torch.float32) if nn_cfg.get("input_scale") else None
    input_shift = torch.tensor(nn_cfg["input_shift"], device=device, dtype=torch.float32) if nn_cfg.get("input_shift") else None
    net = build_scripted_state_mlp(
        in_dim=int(nn_cfg["in_dim"]),
        out_dim=int(nn_cfg["out_dim"]),
        hidden_dims=_hidden_dims_from_config(nn_cfg),
        activation=str(nn_cfg.get("activation", "tanh")),
        input_scale=input_scale,
        input_shift=input_shift,
        device=device,
    )
    state_dict = ckpt.get("latent_net_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain latent_net_state_dict.")
    net.load_state_dict(state_dict, strict=False)
    return net


def integrate_checkpoint(args) -> Path:
    """Load a saved hybrid model checkpoint, integrate it, and save a trajectory."""
    device = torch.device(args.device)
    _install_checkpoint_compat_modules()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload: {type(ckpt)}")

    system_name = str(ckpt["system_name"]).lower()
    system_cfg = ckpt["system_config"]
    integrator_cfg = _as_config(ckpt["integrator_config"], IntegratorConfig)
    ckf_cfg = _as_config(ckpt["ckf_config"], CKFConfig)
    nn_cfg = dict(ckpt["nn_config"])

    latent_net = _build_latent_net(ckpt, nn_cfg, device)
    physics = _build_physics(system_name, system_cfg, integrator_cfg, ckf_cfg, device)
    integrator = _build_integrator(system_name, physics, latent_net, integrator_cfg, system_cfg, device)
    integrator.eval()

    y0 = _parse_y0(args.y0, device)
    if y0 is None:
        y0 = _default_y0(ckpt, args.init_source, device)
    if y0.numel() != len(ckpt.get("state_labels", [])) and ckpt.get("state_labels"):
        raise ValueError(f"--y0 has {y0.numel()} values but checkpoint state has {len(ckpt['state_labels'])} states.")

    dt, duration = _checkpoint_dt_and_duration(ckpt, args.dt, args.duration)
    n_steps = max(1, int(round(duration / dt)))
    t_out = torch.linspace(float(args.t0), float(args.t0) + n_steps * dt, n_steps + 1, device=device)
    states = torch.empty((n_steps + 1, y0.numel()), dtype=torch.float32, device=device)
    states[0] = y0

    drive_seq = ckpt.get("drive_seq")
    if drive_seq is not None:
        drive_seq = torch.as_tensor(drive_seq, dtype=torch.float32, device=device)

    with torch.no_grad():
        y = y0.unsqueeze(0)
        for step in range(n_steps):
            if drive_seq is not None and hasattr(integrator, "rk4_step_with_drive"):
                drive = drive_seq[min(step, drive_seq.shape[0] - 1)]
                y = integrator.rk4_step_with_drive(y, dt, drive.unsqueeze(0))
            else:
                y = integrator.rk4_step(y, dt)
            if hasattr(integrator, "physics") and hasattr(integrator.physics, "clip_states"):
                y = integrator.physics.clip_states(y)
            states[step + 1] = y.squeeze(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array(ckpt.get("state_labels", [f"x{i}" for i in range(states.shape[1])]), dtype=object)
    np.savez(
        out_path,
        t=t_out.detach().cpu().numpy(),
        states=states.detach().cpu().numpy(),
        state_labels=labels,
        system_name=system_name,
        checkpoint=str(Path(args.checkpoint)),
        y0=y0.detach().cpu().numpy(),
        dt=float(dt),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Integrate a saved hybrid ODE checkpoint.")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint.pt or saved epoch .pt.")
    parser.add_argument("--output", "-o", default="results/standalone_integrated_trajectory.npz", help="Output .npz path.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Torch device for integration.")
    parser.add_argument("--y0", default=None, help="Comma-separated initial state. Default: checkpoint initial state.")
    parser.add_argument("--init-source", default="auto", choices=("auto", "true", "smoothed", "filtered", "prior"), help="Checkpoint tensor used when --y0 is omitted.")
    parser.add_argument("--duration", type=float, default=None, help="Integration duration. Default: checkpoint time span.")
    parser.add_argument("--dt", type=float, default=None, help="Integration step. Default: checkpoint mean dt.")
    parser.add_argument("--t0", type=float, default=0.0, help="Output start time.")
    args = parser.parse_args()

    out_path = integrate_checkpoint(args)
    print(f"saved integrated trajectory: {out_path}")


if __name__ == "__main__":
    main()

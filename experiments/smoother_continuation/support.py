"""Shared checkpoint and model-construction support."""

from __future__ import annotations

import argparse
import os
import secrets
import string
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

import models as _models_pkg
from loss_rollout import cov_weighted_multi_step_rollout_loss, multi_step_rollout_loss
from models import (
    BrusselatorPhysics, HOPhysics, HodgkinHuxleyPhysics, PendulumPhysics,
    RetinaPhysics, YeastPhysics, LearnedODE2D, LearnedODE3D, LearnedODE4D,
    LearnedODEYeast, LearnedODERetina4WithPin,
)
from nn_ode_models import build_scripted_state_mlp
from viz_ode import (
    plot_dataset_ode, plot_filter_smoother_uncertainty, plot_model_vs_true_ode,
    plot_phase_portrait, plot_rollouts,
)


DEFAULT_CHECKPOINT = ""
_here = Path(__file__).resolve().parents[2]


def _safe_resolve_path(path_like) -> Path:
    path = Path(path_like)
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _install_checkpoint_compat_modules():
    """Let checkpoints that referenced the former root config module unpickle."""
    if "ode_config" not in sys.modules:
        compat = types.ModuleType("ode_config")
        for name in dir(_models_pkg):
            setattr(compat, name, getattr(_models_pkg, name))
        sys.modules["ode_config"] = compat


def _tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cfg_get_replaced_indices(system_cfg, default_idx):
    if isinstance(system_cfg, dict):
        raw = system_cfg.get("replaced_indices")
        if raw is None:
            raw = (system_cfg.get("replaced_index", default_idx),)
    else:
        raw = getattr(system_cfg, "replaced_indices", None)
        if raw is None:
            raw = (getattr(system_cfg, "replaced_index", default_idx),)
    if isinstance(raw, (int, float, str)):
        raw = (raw,)
    return tuple(int(value) for value in raw)


def _cfg_set_replaced_indices(system_cfg, indices):
    cleaned = []
    for value in indices:
        idx = int(value)
        if idx not in cleaned:
            cleaned.append(idx)
    if not cleaned:
        raise ValueError("At least one replaced index is required.")
    if isinstance(system_cfg, dict):
        system_cfg["replaced_index"] = cleaned[0]
        system_cfg["replaced_indices"] = tuple(cleaned)
    else:
        setattr(system_cfg, "replaced_index", cleaned[0])
        setattr(system_cfg, "replaced_indices", tuple(cleaned))
    return tuple(cleaned)


def _parse_replaced_indices_arg(raw, name_to_idx, *, allowed):
    if raw is None:
        return None
    items = [part.strip().lower() for part in str(raw).replace(";", ",").replace("+", ",").split(",")]
    output = []
    for item in items:
        if not item:
            continue
        idx = int(name_to_idx[item]) if item in name_to_idx else int(item)
        if idx not in allowed:
            raise ValueError(f"Unsupported replaced index {idx}; allowed indices: {sorted(allowed)}")
        if idx not in output:
            output.append(idx)
    if not output:
        raise ValueError(f"Could not parse replaced-index value: {raw!r}")
    return tuple(output)


def _random_tag3() -> str:
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(3))


def _normalize_repo_path(path_like) -> Path:
    raw = str(path_like).strip().strip('"').strip("'")
    if not raw:
        raise ValueError("A checkpoint path is required.")
    path = Path(raw)
    if path.is_absolute():
        return _safe_resolve_path(path)
    repo_relative = _safe_resolve_path(_here / path)
    return repo_relative if repo_relative.exists() else _safe_resolve_path(path)


def _resolve_checkpoint_path(path_like) -> Path:
    checkpoint_path = _normalize_repo_path(path_like)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return _safe_resolve_path(checkpoint_path)


def _sanitized_optimizer_state(optimizer):
    """Drop optimizer moments when any saved tensor is non-finite."""
    state_dict = optimizer.state_dict()
    if _optimizer_state_has_nonfinite(state_dict):
        print("[FromSmoother] WARNING: dropping non-finite optimizer state.")
        state_dict["state"] = {}
    return state_dict


def _optimizer_state_has_nonfinite(opt_state_dict) -> bool:
    if not isinstance(opt_state_dict, dict):
        return False
    for parameter_state in opt_state_dict.get("state", {}).values():
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                return True
    return False


def _load_compatible_state_dict(model, loaded_state, label: str) -> bool:
    """Load finite, shape-compatible parameters from a checkpoint."""
    if not isinstance(loaded_state, dict):
        print(f"[{label}] WARNING: missing checkpoint weights; starting fresh.")
        return False
    model_state = model.state_dict()
    loadable = {}
    for key, value in loaded_state.items():
        if key not in model_state or model_state[key].shape != value.shape:
            continue
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            continue
        loadable[key] = value
    if not loadable:
        print(f"[{label}] WARNING: no compatible finite weights; starting fresh.")
        return False
    model.load_state_dict(loadable, strict=False)
    return True


def _expand_hidden_dims_for_linear_layers(base_hidden_dims, n_linear_layers: int):
    if n_linear_layers < 2:
        raise ValueError("--nn-linear-layers must be at least 2 (hidden + output).")
    n_hidden = n_linear_layers - 1
    base = list(base_hidden_dims) if isinstance(base_hidden_dims, (list, tuple)) else [int(base_hidden_dims)]
    if not base:
        base = [32]
    if len(base) == n_hidden:
        return tuple(int(value) for value in base)
    if len(base) == 1:
        return tuple([int(base[0])] * n_hidden)
    if n_hidden > len(base):
        return tuple([int(base[0])] * (n_hidden - 1) + [int(base[-1])])
    return tuple(int(value) for value in base[:n_hidden])


def compute_horizon_steps(epoch: int, n_epochs: int, min_h: int, max_h: int) -> int:
    if n_epochs <= 1:
        return max_h
    fraction = float(epoch - 1) / float(max(1, n_epochs - 1))
    value = int(round(min_h + fraction * (max_h - min_h)))
    return max(min_h, min(max_h, value))


def compute_plot_horizon_steps(total_steps, base, min_fraction=0.0, min_steps=0):
    horizon = max(base, min_steps) if min_steps > 0 else base
    if min_fraction > 0:
        horizon = max(horizon, int(total_steps * min_fraction))
    return min(horizon, total_steps - 1) if total_steps > 1 else 1


def _build_physics(system_name, system_cfg, integrator_cfg, ckf_cfg, device):
    builders = {
        "ho": lambda: HOPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "pendulum": lambda: PendulumPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "hh": lambda: HodgkinHuxleyPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "brusselator": lambda: BrusselatorPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "retina": lambda: RetinaPhysics(system_cfg, integrator_cfg, ckf_cfg),
        "yeast": lambda: YeastPhysics(system_cfg, integrator_cfg, ckf_cfg),
    }
    if system_name not in builders:
        raise ValueError(f"Unknown system: {system_name}")
    return builders[system_name]().to(device)


def _build_integrator(system_name, physics, latent_net, integrator_cfg, system_cfg, device):
    max_dt = integrator_cfg["max_dt"] if isinstance(integrator_cfg, dict) else integrator_cfg.max_dt
    min_dt = integrator_cfg["min_dt"] if isinstance(integrator_cfg, dict) else integrator_cfg.min_dt
    if system_name == "retina":
        replaced = _cfg_get_replaced_indices(system_cfg, 2)
        return LearnedODERetina4WithPin(
            physics=physics, latent_model=latent_net, max_dt=max_dt, min_dt=min_dt,
            replaced_index=replaced[0], replaced_indices=replaced,
        ).to(device)
    if system_name == "hh":
        return LearnedODE4D(physics=physics, latent_model=latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=1).to(device)
    if system_name == "brusselator":
        return LearnedODE3D(physics=physics, latent_model=latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=0).to(device)
    if system_name == "yeast":
        return LearnedODEYeast(physics=physics, latent_model=latent_net, max_dt=max_dt, min_dt=min_dt, replaced_index=3).to(device)
    return LearnedODE2D(physics=physics, latent_model=latent_net, max_dt=max_dt, min_dt=min_dt).to(device)


def _get_learned_idx(system_name, system_cfg):
    mapping = {
        "ho": [1], "pendulum": [1], "hh": [1],
        "brusselator": [0], "yeast": [3],
    }
    if system_name == "retina":
        return list(_cfg_get_replaced_indices(system_cfg, 2))
    return mapping[system_name]


def _get_replaced_label(system_name, state_labels, system_cfg):
    indices = _get_learned_idx(system_name, system_cfg)
    return ", ".join(state_labels[idx] if 0 <= idx < len(state_labels) else f"idx{idx}" for idx in indices)


__all__ = [name for name in globals() if not name.startswith("__")]

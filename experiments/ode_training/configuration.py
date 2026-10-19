"""Experiment configuration, CLI-value helpers, and run naming."""

from __future__ import annotations

from .environment import *

VALID_MODELS = {'ho', 'pendulum', 'hh', 'brusselator', 'retina', 'retina2', 'yeast'}


def _safe_resolve_path(path_like) -> Path:
    """Resolve paths robustly on flaky network mounts (e.g., sshfs on Windows)."""
    p = Path(path_like)
    try:
        return p.resolve()
    except Exception:
        try:
            return p.absolute()
        except Exception:
            return p

def _data_config_payload(data_cfg):
    if data_cfg is None:
        return {}
    keys = ('max_step_scipy', 'uniform_dt', 'keep_adaptive_grid', 'scipy_method', 'seed')
    return {key: getattr(data_cfg, key, None) for key in keys}


def _random_tag3() -> str:
    return ''.join((secrets.choice(string.ascii_lowercase) for _ in range(3)))

def _sanitize_tag_text(text) -> str:
    text = str(text)
    return ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text))

def _artifact_system_name(exp) -> str:
    requested = getattr(exp, 'requested_model_name', '')
    if requested:
        return str(requested)
    return exp.system_name

def compute_horizon_steps(epoch: int, n_epochs: int, min_h: int, max_h: int) -> int:
    """Linear schedule between min_h and max_h over epochs."""
    if n_epochs <= 1:
        return max_h
    frac = float(epoch - 1) / float(max(1, n_epochs - 1))
    h = int(round(min_h + frac * (max_h - min_h)))
    return max(min_h, min(max_h, h))

def compute_plot_horizon_steps(total_steps: int, base_horizon: int, min_fraction: float=0.0, min_steps: int=0) -> int:
    """Keep rollout plots visible by enforcing a minimum fraction of the trajectory."""
    if total_steps <= 1:
        return 0
    plot_horizon = int(base_horizon)
    min_steps_floor = int(min_steps)
    if min_fraction > 0.0:
        min_steps_frac = int(round((total_steps - 1) * float(min_fraction)))
        plot_horizon = max(plot_horizon, min_steps_frac)
    if min_steps_floor > 0:
        plot_horizon = max(plot_horizon, min_steps_floor)
    return max(1, min(plot_horizon, total_steps - 1))

def _fmt_diag(diag_vals) -> str:
    return '-'.join((f'{float(v):.2e}'.replace('+', '') for v in diag_vals))

def _range_tag(diag_vals) -> str:
    vals = [float(v) for v in diag_vals]
    vmin = min(vals)
    vmax = max(vals)
    return f'{vmin:.1e}-{vmax:.1e}'.replace('+', '')

def _make_run_tag(exp, tc) -> str:
    from datetime import datetime
    import hashlib
    ck = exp.ckf_cfg
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    q_tag = _range_tag(ck.Q_diag)
    r_tag = _range_tag(ck.R_diag)
    p_tag = _range_tag(ck.P0_diag)
    snr_tag = ''
    if hasattr(tc, 'meas_snr') and tc.meas_snr:
        snr_tag = f'_SNR{tc.meas_snr:g}'
    rep_tag = ''
    if exp.system_name == 'retina':
        rep_raw = getattr(exp.system_cfg, 'replaced_indices', None)
        if rep_raw is None:
            rep_raw = getattr(exp, 'pseudo_idx', None)
        if rep_raw:
            rep_tag = '_rep' + '-'.join((str(int(v)) for v in rep_raw))
        else:
            rep_tag = f'_rep{int(exp.system_cfg.replaced_index)}'
    full = f'Q={_fmt_diag(ck.Q_diag)};R={_fmt_diag(ck.R_diag)};P0={_fmt_diag(ck.P0_diag)}'
    h = hashlib.sha1(full.encode()).hexdigest()[:8]
    tag = f'{ts}_E{tc.n_epochs}_Q{q_tag}_R{r_tag}_P0{p_tag}{snr_tag}{rep_tag}_ckf{h}'
    if len(tag) > 120:
        tag = f'{ts}_E{tc.n_epochs}{snr_tag}{rep_tag}_ckf{h}'
    return tag

def _get_replaced_state_label(exp, state_labels):
    if hasattr(exp.system_cfg, 'replaced_indices'):
        try:
            labels = []
            for idx_raw in exp.system_cfg.replaced_indices:
                idx = int(idx_raw)
                labels.append(state_labels[idx] if 0 <= idx < len(state_labels) else f'idx{idx}')
            return ', '.join(labels)
        except Exception:
            return None
    if hasattr(exp.system_cfg, 'replaced_index'):
        try:
            idx = int(exp.system_cfg.replaced_index)
            if 0 <= idx < len(state_labels):
                return state_labels[idx]
            return f'idx{idx}'
        except Exception:
            return None
    return None

def _get_replaced_indices(exp):
    raw = getattr(exp.system_cfg, 'replaced_indices', None)
    if raw is None:
        raw = getattr(exp, 'pseudo_idx', None)
    if raw:
        if isinstance(raw, (int, float, str)):
            raw = (raw,)
        return tuple((int(v) for v in raw))
    if hasattr(exp.system_cfg, 'replaced_index'):
        return (int(exp.system_cfg.replaced_index),)
    return tuple()


def _set_replaced_indices(exp, indices):
    """Synchronize learned-state metadata after a model config is normalized."""
    cleaned = []
    for idx in indices:
        idx_i = int(idx)
        if idx_i not in cleaned:
            cleaned.append(idx_i)
    if not cleaned:
        raise ValueError('At least one replaced index is required.')
    exp.system_cfg.replaced_index = int(cleaned[0])
    exp.system_cfg.replaced_indices = tuple(cleaned)
    exp.pseudo_idx = tuple(cleaned)
    return tuple(cleaned)

def _dependency_union(dependency_map, replaced_indices):
    deps = []
    for idx in replaced_indices:
        for dep_idx in dependency_map[int(idx)]:
            if dep_idx not in deps:
                deps.append(dep_idx)
    return tuple(deps)

def _save_training_curves(log_file: Path, out_dir: Path, title_prefix_rep: str):
    try:
        if not log_file.exists():
            return
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(log_file)
        agg = {'loss': 'mean', 'total_rmse': 'mean', 'horizon': 'first', 'duration_s': 'sum'}
        df_grouped = df.groupby('epoch').agg(agg).reset_index()
        (fig, axes) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{title_prefix_rep}: Training Metrics', fontsize=16, fontweight='bold')
        axes[0, 0].plot(df_grouped['epoch'], df_grouped['loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 1].plot(df_grouped['epoch'], df_grouped['total_rmse'], 'r-', linewidth=2, marker='o', markersize=4, label='Total RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('State Estimation RMSE Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[1, 0].plot(df_grouped['epoch'], df_grouped['horizon'], 'g-', linewidth=2, marker='s', markersize=4, label='Rollout Horizon')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Horizon Steps')
        axes[1, 0].set_title('Training Horizon Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 1].plot(df_grouped['epoch'], df_grouped['duration_s'], 'purple', linewidth=2, marker='^', markersize=4, label='Epoch Duration')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Duration (s)')
        axes[1, 1].set_title('Training Time Per Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        curves_path = out_dir / 'training_curves.png'
        fig.canvas.draw()
        plt.savefig(str(curves_path), dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {curves_path}')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] training curves plot skipped: {e}')

def _parse_diag_values(raw, expected_len: int, label: str):
    raw = '' if raw is None else str(raw).strip()
    if not raw:
        return None
    parts = [p for p in raw.replace(',', ' ').split() if p]
    vals = [float(p) for p in parts]
    if not vals:
        return None
    if expected_len is None:
        return tuple(vals)
    n = len(vals)
    if n == expected_len:
        return tuple(vals)
    if n > expected_len:
        trimmed = tuple(vals[:expected_len])
        print(f'[warn] {label}: got {n} values, expected {expected_len}; truncating to {trimmed}')
        return trimmed
    pad_val = vals[-1]
    padded = tuple(vals + [pad_val] * (expected_len - n))
    print(f'[warn] {label}: got {n} values, expected {expected_len}; padding with {pad_val} -> {padded}')
    return padded

def _arg_value(exp, name: str, default=None):
    val = getattr(exp, name, None)
    return default if val is None else val

def _arg_bool(exp, name: str, default: bool=False) -> bool:
    val = getattr(exp, name, None)
    return bool(default if val is None else val)

def _arg_int(exp, name: str, default: int) -> int:
    val = getattr(exp, name, None)
    return int(default if val is None else val)

def _arg_float(exp, name: str, default: float) -> float:
    val = getattr(exp, name, None)
    return float(default if val is None else val)

def _apply_ckf_overrides(ckf_cfg, *, q_diag=None, r_diag=None, p0_diag=None):
    q_vals = _parse_diag_values(q_diag, len(ckf_cfg.Q_diag), '--ckf-q-diag')
    r_vals = _parse_diag_values(r_diag, len(ckf_cfg.R_diag), '--ckf-r-diag')
    p_vals = _parse_diag_values(p0_diag, len(ckf_cfg.P0_diag), '--ckf-p0-diag')
    q_source = '--ckf-q-diag'
    r_source = '--ckf-r-diag'
    p_source = '--ckf-p0-diag'
    if q_vals is not None:
        ckf_cfg.Q_diag = q_vals
        print(f'[CKF] Q_diag overridden via {q_source}: {q_vals}')
    if r_vals is not None:
        ckf_cfg.R_diag = r_vals
        print(f'[CKF] R_diag overridden via {r_source}: {r_vals}')
    if p_vals is not None:
        ckf_cfg.P0_diag = p_vals
        print(f'[CKF] P0_diag overridden via {p_source}: {p_vals}')


__all__ = [name for name in globals() if not name.startswith("__")]

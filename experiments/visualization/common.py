"""Shared plotting setup, filenames, and rollout safety utilities."""

from __future__ import annotations

import math
import os
import secrets
import string
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

from distancecheck import compute_phase_portrait_error

DEFAULT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "figure.titlesize": 24,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "axes.linewidth": 1.6,
    "grid.linewidth": 0.9,
    "grid.alpha": 0.25,
    "axes.grid": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COLORS = {
    "filter": "#6b7280",
    "smoother": "#d62728",
    "true": "#2ca02c",
    "meas": "#4b5563",
    "rollout": "#9467bd",
    "uncertainty": "#6b7280",
    "drive": "#8c564b",
}
ROLLOUT_COLORS = [
    "#9467bd", "#e377c2", "#8c564b", "#17becf", "#bcbd22",
    "#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4",
]
plt.rcParams.update(DEFAULT_RCPARAMS)

def _safe_resolve_path(path_like) -> Path:
    """Resolve path without hard-failing on transient network filesystem errors."""
    p = Path(path_like)
    try:
        return p.resolve()
    except Exception:
        try:
            return p.absolute()
        except Exception:
            return p

def _state_grid_shape(n_states: int):
    """
    Choose a subplot grid focused on horizontal layouts for paper figures.
    Special cases requested by user:
      - 4 states -> 1 row x 4 cols
      - 3 states -> 1 row x 3 cols
      - 10 states -> 2 rows x 5 cols
      - 7 states -> 2 rows x 4 cols (one empty slot)
    Fallback: up to 4 columns, rows filled in order.
    """
    if n_states <= 0:
        return (1, 1)
    if n_states == 4:
        return (1, 4)
    if n_states == 3:
        return (1, 3)
    if n_states == 10:
        return (2, 5)
    if n_states == 7:
        return (2, 4)
    n_cols = min(max(1, n_states), 4)
    n_rows = math.ceil(n_states / n_cols)
    return (n_rows, n_cols)

def _random_tag3() -> str:
    return ''.join((secrets.choice(string.ascii_lowercase) for _ in range(3)))

def _timestamp_tag() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_random_tag3()}"

def reset_plot_params():
    """Reset matplotlib rcParams to ensure consistent plotting across Monte Carlo runs."""
    plt.rcParams.update(DEFAULT_RCPARAMS)

def _show_if_interactive(show: bool) -> None:
    """Avoid noisy warnings when using a non-interactive backend like Agg on Slurm."""
    if not show:
        return
    try:
        backend = str(matplotlib.get_backend()).lower()
    except Exception:
        backend = ''
    if any((tag in backend for tag in ('agg', 'pdf', 'svg', 'ps', 'template'))):
        return
    plt.show()

def _viz_progress_every(default: int=500) -> int:
    """Progress print cadence for long-running visualization loops."""
    raw = os.getenv('ODE_VIZ_PROGRESS_EVERY', str(default)).strip()
    try:
        return max(1, int(raw))
    except Exception:
        return default

def _finite_prefix_length(arr: np.ndarray) -> int:
    """Number of consecutive leading entries whose full row stays finite."""
    arr_np = np.asarray(arr)
    if arr_np.ndim <= 1:
        finite_rows = np.isfinite(arr_np)
    else:
        finite_rows = np.all(np.isfinite(arr_np), axis=tuple(range(1, arr_np.ndim)))
    bad = np.flatnonzero(~finite_rows)
    return int(bad[0]) if bad.size else int(arr_np.shape[0])

def _rollout_visibility_bounds(states_np: np.ndarray, true_np: np.ndarray | None=None):
    """
    Reference envelope used only for plotting. This prevents one exploding
    rollout from blowing out autoscaling and hiding all useful traces.
    """
    try:
        bound_mult = max(1.0, float(os.getenv('ODE_PLOT_BLOWUP_MULT', '500')))
    except Exception:
        bound_mult = 50.0
    ref_parts = [np.asarray(states_np)]
    if true_np is not None and np.size(true_np) > 0:
        ref_parts.append(np.asarray(true_np))
    ref = np.concatenate(ref_parts, axis=0)
    if ref.ndim == 1:
        ref = ref[:, None]
    finite_rows = np.all(np.isfinite(ref), axis=1)
    ref = ref[finite_rows]
    n_states = states_np.shape[1] if np.asarray(states_np).ndim > 1 else 1
    if ref.size == 0:
        base = np.ones(n_states, dtype=np.float64)
        return (-bound_mult * base, bound_mult * base, bound_mult)
    ref_min = np.min(ref, axis=0)
    ref_max = np.max(ref, axis=0)
    ref_span = np.maximum(ref_max - ref_min, 1e-06)
    ref_scale = np.maximum(np.maximum(np.abs(ref_min), np.abs(ref_max)), np.maximum(ref_span, 1.0))
    lower = ref_min - bound_mult * ref_scale
    upper = ref_max + bound_mult * ref_scale
    return (lower, upper, bound_mult)

def _truncate_rollout_for_plot(ts_roll: np.ndarray, ys_roll: np.ndarray, lower_bounds: np.ndarray | None=None, upper_bounds: np.ndarray | None=None):
    """
    Keep only the leading visible prefix of a rollout so saved plots remain
    readable even when the model diverges.
    """
    ts_np = np.asarray(ts_roll)
    ys_np = np.asarray(ys_roll)
    if ts_np.size == 0 or ys_np.size == 0:
        return (ts_np, ys_np, None, None)
    ys_rows = ys_np[:, None] if ys_np.ndim == 1 else ys_np
    prefix_len = _finite_prefix_length(ys_rows)
    reason = None
    bad_step = None
    if prefix_len < ys_rows.shape[0]:
        reason = 'non-finite'
        bad_step = prefix_len
    if prefix_len > 0 and lower_bounds is not None and (upper_bounds is not None):
        ys_prefix = ys_rows[:prefix_len]
        in_bounds = np.all((ys_prefix >= lower_bounds[None, :]) & (ys_prefix <= upper_bounds[None, :]), axis=1)
        bad_bounds = np.flatnonzero(~in_bounds)
        if bad_bounds.size:
            prefix_len = int(bad_bounds[0])
            reason = 'left-visible-range'
            bad_step = prefix_len
    return (ts_np[:prefix_len], ys_np[:prefix_len], reason, bad_step)

def _get_run_dir(base_dir='results'):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = _timestamp_tag()
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _save_fig_and_arrays(base_dir, prefix, tag, fig, arrays_dict, epoch=None, epoch_timestamp=None):
    """Save figure and data arrays to timestamped folder.

    Args:
        base_dir: Base results directory
        prefix: Model/experiment name (e.g., 'retina')
        tag: Figure tag/name
        fig: Matplotlib figure object
        arrays_dict: Dictionary of arrays to save as npz
        epoch: Optional epoch number to include in folder name
        epoch_timestamp: Optional timestamp string (reused across multiple saves in same epoch)
    """
    try:
        if prefix.lower().startswith('hh_'):
            prefix = 'hh'
        base = _safe_resolve_path(base_dir) / prefix
        base.mkdir(parents=True, exist_ok=True)
        if epoch_timestamp is None:
            epoch_timestamp = _timestamp_tag()
        if epoch is not None:
            out_dir = base / epoch_timestamp / f'epoch{epoch:03d}'
        else:
            out_dir = base / epoch_timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        sanitized_tag = tag.replace(':', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('?', '_').replace('*', '_')
        pdf_path = out_dir / f'{sanitized_tag}.pdf'
        npz_path = out_dir / f'{sanitized_tag}.npz'
        print(f'[viz] attempting to save: {str(pdf_path)}')
        print(f'[viz] figure object: {fig}')
        print(f'[viz] figure has {len(fig.axes)} axes')
        fig.canvas.draw()
        fig.savefig(str(pdf_path), bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'[viz] PDF saved successfully')
        np.savez(str(npz_path), **arrays_dict)
        print(f'[viz] NPZ saved successfully')
        print(f'[viz] saved fig:  {pdf_path}')
        print(f'[viz] saved data: {npz_path}')
        print(f'[viz] absolute dir: {_safe_resolve_path(out_dir)}')
    except Exception as e:
        print(f'[viz] ERROR saving figure: {e}')
        import traceback
        traceback.print_exc()

__all__ = ['_finite_prefix_length', '_get_run_dir', '_random_tag3', '_rollout_visibility_bounds', '_safe_resolve_path', '_save_fig_and_arrays', '_show_if_interactive', '_state_grid_shape', '_timestamp_tag', '_truncate_rollout_for_plot', '_viz_progress_every', 'reset_plot_params', 'DEFAULT_RCPARAMS', 'COLORS', 'ROLLOUT_COLORS', 'matplotlib', 'math', 'np', 'torch', 'plt', 'Path', 'time', 'os', 'compute_phase_portrait_error']

"""Distance utilities for phase-portrait diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def _directed_mean_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Support the directed mean min distance step in this module.
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return float(np.mean(np.min(dist, axis=1)))


def _clean_points(path: np.ndarray, indices: tuple[int, int]) -> np.ndarray:
    # Support the clean points step in this module.
    pts = np.asarray(path, dtype=np.float64)[:, indices]
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def compute_phase_portrait_error(
    true_path: np.ndarray,
    learned_path: np.ndarray,
    state_indices: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute symmetric nearest-neighbor phase-portrait errors.

    Returns one entry per 2D state-pair plus an ``overall`` mean. The function is
    intentionally dependency-light because it is used only for diagnostics.
    """
    true_arr = np.asarray(true_path, dtype=np.float64)
    learned_arr = np.asarray(learned_path, dtype=np.float64)
    if true_arr.ndim != 2 or learned_arr.ndim != 2:
        raise ValueError("true_path and learned_path must be 2D arrays")

    n_states = min(true_arr.shape[1], learned_arr.shape[1])
    if state_indices is None:
        indices = tuple(range(n_states))
    else:
        indices = tuple(int(i) for i in state_indices if 0 <= int(i) < n_states)

    if len(indices) < 2:
        return {"overall": float("nan")}

    errors: dict[str, float] = {}
    values: list[float] = []
    for i, j in combinations(indices, 2):
        true_pts = _clean_points(true_arr, (i, j))
        learned_pts = _clean_points(learned_arr, (i, j))
        forward = _directed_mean_min_distance(true_pts, learned_pts)
        backward = _directed_mean_min_distance(learned_pts, true_pts)
        value = float(np.nanmean([forward, backward]))
        key = f"{i}-{j}"
        errors[key] = value
        if np.isfinite(value):
            values.append(value)

    errors["overall"] = float(np.mean(values)) if values else float("nan")
    return errors

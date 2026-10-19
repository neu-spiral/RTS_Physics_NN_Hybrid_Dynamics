"""Estimate CKF Q_diag and P0_diag for configured ODE experiments.

This script is intentionally independent of ode_experiments.py. It uses the
model package lookup and data generators only, then estimates full-state CKF
process noise (Q) and initial covariance (P0) from each generated trajectory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from models import (
    get_experiment,
    make_data_brusselator,
    make_data_hh,
    make_data_ho,
    make_data_pendulum,
    make_data_retina,
    make_data_yeast,
)
from experiments.ode_training.estimation_config import CovarianceEstimationConfig


DATA_GENERATORS = {
    "ho": make_data_ho,
    "pendulum": make_data_pendulum,
    "hh": make_data_hh,
    "brusselator": make_data_brusselator,
    "yeast": make_data_yeast,
    "retina": make_data_retina,
}

ESTIMABLE_MODELS = tuple(sorted(DATA_GENERATORS))


PRIOR_FIELDS = {
    "ho": ("x0_prior", "v0_prior"),
    "pendulum": ("theta0_prior", "omega0_prior"),
    "hh": ("V0_prior", "m0_prior", "h0_prior", "n0_prior"),
    "brusselator": ("x0_prior", "y0_prior", "z0_prior"),
    "retina": ("p1_prior", "p2_prior", "p4_prior", "p5_prior"),
}


def _as_2d_state(data: tuple, n_state: int) -> torch.Tensor:
    """Extract full true state from a make_data_* return tuple."""
    for item in data[2:]:
        if torch.is_tensor(item):
            x = item
            if x.dim() == 3 and x.size(1) == 1:
                x = x[:, 0, :]
            if x.dim() == 2 and x.shape[1] == n_state:
                return x

    state_cols = []
    for item in data[2:]:
        if torch.is_tensor(item):
            x = item
            if x.dim() == 1:
                state_cols.append(x)
            elif x.dim() == 2 and x.shape[1] == 1:
                state_cols.append(x[:, 0])
    if len(state_cols) == n_state:
        return torch.stack(state_cols, dim=1)

    raise ValueError(f"Could not extract a {n_state}D true state from data generator output.")


def _prior_vector(model: str, exp, n_state: int) -> np.ndarray:
    # Support the prior vector step in this module.
    sys_cfg = exp.system_cfg
    if model == "yeast" and hasattr(sys_cfg, "x0_prior"):
        prior = np.asarray(sys_cfg.x0_prior, dtype=np.float64)
    else:
        fields = PRIOR_FIELDS.get(model, ())
        prior = np.asarray([getattr(sys_cfg, name) for name in fields], dtype=np.float64)
    if prior.size != n_state:
        raise ValueError(f"{model}: expected {n_state} prior values, got {prior.size}.")
    return prior


def _learned_indices(model: str, exp, n_state: int) -> list[int]:
    # Support the learned indices step in this module.
    pseudo_idx = list(getattr(exp, "pseudo_idx", ()) or ())
    if pseudo_idx:
        return pseudo_idx
    meas_idx = list(getattr(exp, "meas_idx", ()) or ())
    if meas_idx:
        return [idx for idx in range(n_state) if idx not in set(meas_idx)]
    return []


def _finite_trajectory(t: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    # Support the finite trajectory step in this module.
    t_np = t.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)
    mask = np.isfinite(t_np) & np.all(np.isfinite(y_np), axis=1)
    t_np = t_np[mask]
    y_np = y_np[mask]
    if t_np.size < 3:
        raise ValueError("Need at least three finite trajectory samples.")
    order = np.argsort(t_np)
    return t_np[order], y_np[order]


def _uniform_estimation_grid(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate to a uniform grid before estimating time-scaled covariance.

    Adaptive stiff solvers often emit tiny startup steps and very large later
    steps. Estimating Q directly from those raw intervals makes the result
    depend on solver bookkeeping instead of trajectory scale.
    """
    unique_t, unique_idx = np.unique(t, return_index=True)
    t = unique_t
    y = y[unique_idx]
    if len(t) < 3:
        raise ValueError("Need at least three unique time samples.")

    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0.0]
    if positive_dt.size == 0:
        raise ValueError("Time vector has no positive steps.")

    if max_points <= 0:
        return t, y, {"grid_kind": "native", "grid_n_points": int(len(t))}

    t0 = float(t[0])
    t1 = float(t[-1])
    span = max(t1 - t0, 0.0)
    if span <= 0.0:
        raise ValueError("Time vector has zero span.")

    median_dt = float(np.median(positive_dt))
    target_n = int(np.ceil(span / max(median_dt, 1e-12))) + 1
    target_n = max(len(t), target_n)
    target_n = min(max_points, target_n)
    target_n = max(3, target_n)

    t_grid = np.linspace(t0, t1, target_n, dtype=np.float64)
    y_grid = np.empty((target_n, y.shape[1]), dtype=np.float64)
    for j in range(y.shape[1]):
        y_grid[:, j] = np.interp(t_grid, t, y[:, j])

    return t_grid, y_grid, {
        "grid_kind": "uniform_interp",
        "grid_n_points": int(target_n),
        "native_n_points": int(len(t)),
        "native_dt_median": median_dt,
    }


def estimate_q_p0(
    t: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    learned_idx: Iterable[int],
    q_step_fraction: float,
    q_state_floor_fraction: float,
    p0_state_fraction: float,
    learned_q_multiplier: float,
    learned_p0_multiplier: float,
    learned_diffusion_time: float,
    learned_p0_std_fraction: float,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Estimate q p0.
    dt = np.diff(t)
    dy = np.diff(y, axis=0)
    positive = dt > 0.0
    positive_dt = dt[positive]
    if positive_dt.size == 0:
        raise ValueError("Time vector has no positive steps.")

    dy = dy[positive]
    # ckf.py adds Q * dt in the prediction step, so Q_diag is a continuous-time
    # variance intensity. Estimate a robust derivative scale and convert the
    # desired reference-step variance into variance per second.
    rate_abs_p90 = np.percentile(np.abs(dy / positive_dt[:, None]), 90.0, axis=0)
    dt_reference = float(np.median(positive_dt))
    if positive_dt.size >= 10:
        dt_reference = float(np.percentile(positive_dt, 50.0))
    dt_reference = max(dt_reference, 1e-12)

    state_std = np.std(y, axis=0)
    state_abs_mean = np.maximum(np.mean(np.abs(y), axis=0), min_variance)
    state_scale = np.maximum(state_std, state_abs_mean)
    state_floor = q_state_floor_fraction * np.maximum(state_std, state_abs_mean)

    q_step_intensity = (q_step_fraction * rate_abs_p90) ** 2 * dt_reference
    q_floor_intensity = (state_floor**2) / dt_reference
    q_diag = np.maximum(q_step_intensity, q_floor_intensity)

    initial_error = prior - y[0]
    p0_floor = p0_state_fraction * np.maximum(state_std, state_abs_mean)
    p0_diag = np.maximum(initial_error**2, p0_floor**2)

    learned = sorted({int(i) for i in learned_idx if 0 <= int(i) < y.shape[1]})
    if learned:
        diffusion_time = max(float(learned_diffusion_time), dt_reference)
        learned_std = np.maximum(state_std[learned], p0_floor[learned])
        learned_q_floor = (learned_std**2) / diffusion_time
        learned_p0_floor = (float(learned_p0_std_fraction) * learned_std) ** 2
        q_diag[learned] = np.maximum(q_diag[learned], learned_q_floor)
        p0_diag[learned] = np.maximum(p0_diag[learned], learned_p0_floor)
        q_diag[learned] *= learned_q_multiplier
        p0_diag[learned] *= learned_p0_multiplier

    q_diag = np.maximum(q_diag, min_variance)
    p0_diag = np.maximum(p0_diag, min_variance)

    diagnostics = {
        "n_points": int(len(t)),
        "dt_min": float(np.min(positive_dt)),
        "dt_median": float(np.median(positive_dt)),
        "dt_max": float(np.max(positive_dt)),
        "dt_reference": dt_reference,
        "state_abs_mean": state_abs_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_scale": state_scale.tolist(),
        "rate_abs_p90": rate_abs_p90.tolist(),
        "initial_error": initial_error.tolist(),
        "learned_diffusion_time": float(learned_diffusion_time),
        "learned_p0_std_fraction": float(learned_p0_std_fraction),
    }
    return q_diag, p0_diag, diagnostics


def _format_tuple(values: Iterable[float]) -> str:
    # Support the format tuple step in this module.
    return "(" + ", ".join(f"{float(v):.6g}" for v in values) + ")"


def _make_exp(model: str):
    # Build and return exp.
    exp = copy.deepcopy(get_experiment(model))
    return exp


def run_model(model: str, config: CovarianceEstimationConfig, device: torch.device) -> dict:
    # Run model.
    exp = _make_exp(model)
    generator = DATA_GENERATORS[model]
    data = generator(exp, device)

    t_t = data[0]
    n_state = len(exp.state_labels)
    y_t = _as_2d_state(data, n_state)
    t, y = _finite_trajectory(t_t, y_t)
    t, y, grid_info = _uniform_estimation_grid(t, y, config.max_points)

    prior = _prior_vector(model, exp, n_state)
    learned_idx = _learned_indices(model, exp, n_state)
    q_diag, p0_diag, diagnostics = estimate_q_p0(
        t=t,
        y=y,
        prior=prior,
        learned_idx=learned_idx,
        q_step_fraction=config.q_step_fraction,
        q_state_floor_fraction=config.q_state_floor_fraction,
        p0_state_fraction=config.p0_state_fraction,
        learned_q_multiplier=config.learned_q_multiplier,
        learned_p0_multiplier=config.learned_p0_multiplier,
        learned_diffusion_time=config.learned_diffusion_time,
        learned_p0_std_fraction=config.learned_p0_std_fraction,
        min_variance=config.min_variance,
    )
    diagnostics.update(grid_info)

    return {
        "model": model,
        "state_labels": list(exp.state_labels),
        "learned_idx": learned_idx,
        "meas_idx": list(getattr(exp, "meas_idx", ()) or []),
        "current_Q_diag": list(map(float, exp.ckf_cfg.Q_diag)),
        "current_P0_diag": list(map(float, exp.ckf_cfg.P0_diag)),
        "estimated_Q_diag": q_diag.tolist(),
        "estimated_P0_diag": p0_diag.tolist(),
        "diagnostics": diagnostics,
    }


def write_outputs(results: list[dict], output_dir: Path) -> None:
    # Support the write outputs step in this module.
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "estimated_q_p0.json"
    csv_path = output_dir / "estimated_q_p0.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "state_index", "state_label", "learned", "estimated_Q", "estimated_P0", "current_Q", "current_P0"])
        for result in results:
            learned = set(result["learned_idx"])
            for i, label in enumerate(result["state_labels"]):
                writer.writerow(
                    [
                        result["model"],
                        i,
                        label,
                        i in learned,
                        result["estimated_Q_diag"][i],
                        result["estimated_P0_diag"][i],
                        result["current_Q_diag"][i],
                        result["current_P0_diag"][i],
                    ]
                )


def parse_args() -> argparse.Namespace:
    """Parse only the run-wide estimator controls.

    Numerical covariance-heuristic settings live in
    ``experiments/ode_training/estimation_config.py``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(ESTIMABLE_MODELS), choices=ESTIMABLE_MODELS)
    parser.add_argument("--output-dir", default="results/q_p0_estimates")
    parser.add_argument("--no-save", action="store_true", help="Print estimates without writing JSON/CSV files.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first model that cannot be estimated.")
    return parser.parse_args()


def main() -> int:
    # Run the module's command-line entry point.
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    device = torch.device(args.device)
    config = CovarianceEstimationConfig()

    results = []
    failures = []
    for model in args.models:
        print(f"\n=== {model} ===")
        try:
            result = run_model(model, config, device)
            results.append(result)
            print(f"states: {result['state_labels']}")
            print(f"learned_idx: {result['learned_idx']}")
            print(f"estimated Q_diag:  {_format_tuple(result['estimated_Q_diag'])}")
            print(f"estimated P0_diag: {_format_tuple(result['estimated_P0_diag'])}")
        except Exception as exc:
            failures.append((model, str(exc)))
            print(f"[error] {model}: {exc}")
            if args.fail_fast:
                raise

    if not args.no_save:
        output_dir = Path(args.output_dir)
        write_outputs(results, output_dir)
        print(f"\nWrote {output_dir / 'estimated_q_p0.json'}")
        print(f"Wrote {output_dir / 'estimated_q_p0.csv'}")

    if failures:
        print("\nFailures:")
        for model, message in failures:
            print(f"  {model}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

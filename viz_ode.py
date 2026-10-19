"""Compatibility-facing visualization API.

Implementation is grouped under :mod:`experiments.visualization` by plot type.
Existing imports from ``viz_ode`` remain supported."""

from __future__ import annotations

from experiments.visualization import (
    plot_dataset_ode,
    plot_filter_smoother_uncertainty,
    plot_model_vs_true_ode,
    plot_phase_portrait,
    plot_rollouts,
)

__all__ = [
    'plot_filter_smoother_uncertainty', 'plot_rollouts', 'plot_model_vs_true_ode',
    'plot_dataset_ode', 'plot_phase_portrait',
]

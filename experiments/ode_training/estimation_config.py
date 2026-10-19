"""Fixed settings for CKF covariance estimation.

These describe the estimation method, not an individual command invocation.
Edit this preset when the covariance heuristic itself changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CovarianceEstimationConfig:
    """Stable defaults for :mod:`estimate_q_p0`."""

    max_points: int = 20_000
    q_step_fraction: float = 0.25
    q_state_floor_fraction: float = 1e-3
    p0_state_fraction: float = 0.05
    learned_q_multiplier: float = 1.0
    learned_p0_multiplier: float = 1.0
    learned_diffusion_time: float = 5.0
    learned_p0_std_fraction: float = 1.0
    min_variance: float = 1e-14

"""Model profiles for the CKF-based RBSE baseline."""

from dataclasses import dataclass


VALID_MODELS = ('ho', 'pendulum', 'hh', 'brusselator', 'yeast', 'retina')


@dataclass(frozen=True)
class RBSEModelConfig:
    """RBSE-specific model settings, separate from neural-ODE presets."""

    hidden_dims: tuple[int, ...] = (8, 8)
    q_weight_scale: float = 1e-6
    p_weight_scale: float = 1e-2
    q_state_scale: float = 1.0
    r_scale: float = 1.0
    minimum_epochs: int = 1


_DEFAULT_CONFIG = RBSEModelConfig()

MODEL_CONFIGS = {
    # The augmented CKF carries every NN weight as a state, so these models
    # use a deliberately shallow learned residual.
    'hh': RBSEModelConfig(hidden_dims=(2, 2), r_scale=0.5, minimum_epochs=3),
    'retina': RBSEModelConfig(hidden_dims=(2, 2), q_weight_scale=1e-8, p_weight_scale=1e-3, q_state_scale=0.5, r_scale=0.5, minimum_epochs=2),
}


def get_model_config(model_name: str) -> RBSEModelConfig:
    """Return the immutable RBSE profile for ``model_name``."""
    return MODEL_CONFIGS.get(str(model_name).lower(), _DEFAULT_CONFIG)

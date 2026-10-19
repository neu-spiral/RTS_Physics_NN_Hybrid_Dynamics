"""Model-owned settings for frozen-smoother continuation.

Keep choices that describe a model (learned state, auxiliary networks, or
data-specific safeguards) here.  ``ode_from_smoother.py`` deliberately owns
only controls that apply to every continuation run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ContinuationModelConfig:
    """Continuation behaviour that is intrinsic to one model family."""

    replaced_indices: Optional[Tuple[int, ...]] = None
    nn_linear_layers: Optional[int] = None
    loss_mode: Optional[str] = None
    pseudo_weight: Optional[float] = None


# State indices are intentionally numeric: a saved checkpoint records its
# state order, while the continuation loader validates it against the model.
# Edit this table when changing a model's continuation strategy.
MODEL_CONTINUATION_CONFIGS = {}

_DEFAULT_CONFIG = ContinuationModelConfig()


def get_model_continuation_config(system_name: str) -> ContinuationModelConfig:
    """Return immutable continuation defaults for the saved model."""
    return MODEL_CONTINUATION_CONFIGS.get(str(system_name).lower(), _DEFAULT_CONFIG)


def apply_model_continuation_config(args, system_name: str) -> ContinuationModelConfig:
    """Expose compatibility attributes used by the continuation session.

    These are deliberately set after reading the checkpoint, so a caller does
    not need model-specific command-line arguments to resume a run.
    """
    config = get_model_continuation_config(system_name)
    for name in ('pseudo_weight',):
        setattr(args, name, getattr(config, name))
    # Keep this universal command-line option overridable, while making the
    # safe model default independent of a historical checkpoint's setting.
    if getattr(args, 'loss_mode', None) is None and config.loss_mode is not None:
        args.loss_mode = config.loss_mode
    return config

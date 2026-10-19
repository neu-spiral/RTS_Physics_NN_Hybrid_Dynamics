"""Model package exports and experiment lookup.

Each model module contains its own SystemConfig, Physics, data-generation, and
learned-ODE code.  Use get_experiment/make_default_experiment for configured
runs.
"""

from copy import deepcopy

from .base import (
    BasePhysics2D,
    CKFConfig,
    DataGenConfig,
    ExperimentConfig,
    IntegratorConfig,
    NNConfig,
    PlotConfig,
    TrainConfig,
    LearnedODE2D,
    LearnedODE3D,
    LearnedODE4D,
)
from .ho import HOSystemConfig, HOPhysics, LearnedODEHO, integrate_ho_adaptive, make_data_ho
from .pendulum import PendulumSystemConfig, PendulumPhysics, LearnedODEPendulum, integrate_pendulum_adaptive, make_data_pendulum
from .hh import HodgkinHuxleySystemConfig, HodgkinHuxleyPhysics, LearnedODEHH, make_data_hh
from .brusselator import BrusselatorSystemConfig, BrusselatorPhysics, LearnedODEBrusselator, make_data_brusselator
from .yeast import YeastSystemConfig, YeastPhysics, LearnedODEYeast, YEAST_DEPENDENCIES, YEAST_STATE_NAMES, get_yeast_input_mask, make_data_yeast
from .retina import RetinaSystemConfig, RetinaPhysics, LearnedODERetina4WithPin, integrate_retina_adaptive, make_data_retina
from . import brusselator, hh, ho, pendulum, retina, yeast


EXPERIMENTS = {
    "ho": ho.make_experiment(),
    "pendulum": pendulum.make_experiment(),
    "hh": hh.make_experiment(),
    "brusselator": brusselator.make_experiment(),
    "yeast": yeast.make_experiment(),
    "retina": retina.make_experiment(),
    "retina2": retina.make_retina2_experiment(),
}


def get_experiment(model_name: str) -> ExperimentConfig:
    """Return a fresh experiment config for a registered model name."""
    model_name = model_name.lower()
    if model_name not in EXPERIMENTS:
        valid = list(EXPERIMENTS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")
    exp = deepcopy(EXPERIMENTS[model_name])
    exp.requested_model_name = model_name
    if model_name == "retina2":
        exp.system_cfg.replaced_indices = (1, 2)
    return exp


def make_default_experiment(model: str) -> ExperimentConfig:
    """Compatibility name used by the main training script."""
    return get_experiment(model)

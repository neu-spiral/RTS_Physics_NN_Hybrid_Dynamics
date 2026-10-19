"""Process setup, shared dependencies, device selection, and AMP helpers."""

from __future__ import annotations

import os
import sys

def preparse_cli_value(flag_name: str):
    argv = sys.argv[1:]
    for index, argument in enumerate(argv):
        if argument.startswith(flag_name + "="):
            return argument.split("=", 1)[1]
        if argument == flag_name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def configure_process_environment() -> str:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    threads = preparse_cli_value("--cpu-threads")
    value = threads or "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = value
    return threads or ""


configure_process_environment()

import argparse
import copy
import os
import secrets
import shutil
import string
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ckf import ckf_filter, rts_smoother
from loss_rollout import cov_weighted_multi_step_rollout_loss, multi_step_rollout_loss
from models import (
    BasePhysics2D, BrusselatorPhysics, HOPhysics, HodgkinHuxleyPhysics,
    PendulumPhysics, RetinaPhysics, YeastPhysics,
    LearnedODE2D, LearnedODE3D, LearnedODE4D, LearnedODEYeast,
    LearnedODERetina4WithPin, make_data_brusselator, make_data_hh,
    make_data_ho, make_data_pendulum, make_data_retina, make_data_yeast,
)
from nn_ode_models import build_scripted_state_mlp
from viz_ode import plot_dataset_ode, plot_filter_smoother_uncertainty, plot_phase_portrait, plot_rollouts


try:
    from torch.amp import GradScaler, autocast
except (ImportError, ModuleNotFoundError):
    try:
        from torch.cuda.amp import GradScaler as _CudaGradScaler
        from torch.cuda.amp import autocast as _cuda_autocast
    except (ImportError, ModuleNotFoundError):
        _CudaGradScaler = None
        _cuda_autocast = None

    def autocast(device_type="cuda", dtype=None, enabled=True, **kwargs):
        if not enabled or device_type != "cuda" or _cuda_autocast is None:
            return nullcontext()
        if dtype is None:
            return _cuda_autocast(enabled=True, **kwargs)
        try:
            return _cuda_autocast(enabled=True, dtype=dtype, **kwargs)
        except TypeError:
            return _cuda_autocast(enabled=True, **kwargs)

    if _CudaGradScaler is not None:
        class GradScaler(_CudaGradScaler):
            def __init__(self, device_type=None, enabled=True, *args, **kwargs):
                super().__init__(*args, enabled=enabled and device_type in (None, "cuda"), **kwargs)
    else:
        class GradScaler:
            def __init__(self, *args, enabled=False, **kwargs):
                self._enabled = False

            def is_enabled(self):
                return False

            def state_dict(self):
                return {}


def requested_device():
    requested = ""
    argv = [] if any(key in sys.modules for key in ("spyder_kernels",)) else sys.argv[1:]
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--cpu":
            requested = "cpu"
        elif argument.startswith("--device="):
            requested = argument.split("=", 1)[1].strip().lower()
        elif argument == "--device" and index + 1 < len(argv):
            requested = argv[index + 1].strip().lower()
            index += 1
        index += 1
    return requested or "auto"


def resolve_device():
    requested = requested_device()
    if requested in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), requested
    if requested == "cpu":
        return torch.device("cpu"), requested
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), requested
    if requested == "cuda":
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    raise ValueError(f"Unsupported device override {requested!r}. Use auto, cpu, or cuda.")


torch.set_default_dtype(torch.float32)
device, requested_device_name = resolve_device()
print(f"[Device] Using: {device}")
if device.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

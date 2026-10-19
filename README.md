# RTS Smoother-Guided Learning of Physics-Based Neural Differential Models

This repository contains the implementation for the paper
`RTS Smoother-Guided Learning of Physics-Based Neural Differential Models`.
The setting is partially observed, noisy dynamical systems in which one state
derivative is treated as unknown and replaced by a neural component, while the
remaining dynamics stay mechanistic. An RTS smoother estimates latent
trajectories from partial measurements, and those trajectories supervise a
hybrid neural differential model through multi-step rollout losses.

The repository currently supports:

- end-to-end RTS-guided hybrid ODE training via `ode_experiments.py`
- frozen-smoother continuation via `ode_from_smoother.py`
- synthetic benchmarks for harmonic oscillator, pendulum, Brusselator,
  Hodgkin--Huxley, yeast glycolysis, and retinal circulation

The final learned model is a standalone hybrid ODE that can be simulated
forward without filtering or smoothing at test time.

## High-Level Map

`ode_experiments.py`, `ode_from_smoother.py`, and `viz_ode.py` are the public
entry points. Physics and integrators live in `models/`; CKF/RTS and rollout
losses live in `ckf.py` and `loss_rollout.py`.

```text
experiments/
├── ode_training/                 # Standard end-to-end ODE experiments
│   ├── environment.py            # Device and runtime setup
│   ├── configuration.py          # CLI/configuration and restart settings
│   ├── data.py                   # Dataset loading and preparation
│   ├── systems.py                # ODE system construction
│   ├── lifecycle.py              # Run and checkpoint lifecycle management
│   ├── model_setup.py            # Model and optimizer setup
│   ├── training_loop.py          # Training and evaluation loop
│   ├── outputs.py                # Saving results and summaries
│   └── session.py                # Connects the workflow together
├── smoother_continuation/        # Continue from smoother checkpoints
│   ├── support.py                # Shared continuation utilities
│   ├── loading.py                # Checkpoint and data loading
│   ├── model_setup.py            # Continued-model setup
│   ├── training_loop.py          # Continuation training loop
│   └── session.py                # Connects the workflow together
└── visualization/                # Plotting only
    ├── common.py                 # Shared plotting utilities
    ├── plots.py                  # General experiment plots
    ├── phase.py                  # Phase-space plots
    ├── rollout_plots.py          # Rollout-specific plots
    └── rollout_rendering.py      # Figure construction and rendering
```

There is no fragment loader or runtime source compilation. TorchScript model
files remain normal `.py` files because TorchScript requires source access.

For maintainers, see [Model Class Guide and Refactoring Map](docs/model-class-guide.md).
It documents the model/configuration classes, their public interfaces, and
the repeated patterns that are candidates for safe refactoring.

For the relationship between this Git repository and the newer unversioned
experiment tree, see [Project Locations and Migration Status](docs/project-locations-and-migration.md).
For paper-result selection, oracle boundaries, and figure provenance, see
[Oracle-Free Evaluation and Paper Provenance](docs/oracle-free-evaluation-and-paper-provenance.md).

## Requirements

- Anaconda or Miniconda
- Python `3.9+`; Python `3.10` is a good default
- Packages listed in `requirements.txt`: `torch`, `numpy`, `scipy`,
  `matplotlib`, and `pandas`
- Optional GPU runs require a CUDA-capable PyTorch install

### Non-CUDA / CPU Setup

Use this path for laptops, desktops, or servers without GPU training.

```bash
conda create -n rts-hybrid-cpu python=3.10 -y
conda activate rts-hybrid-cpu
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install numpy scipy matplotlib pandas
```

Check the install:

```bash
python -c "import torch, numpy, scipy, matplotlib, pandas; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python ode_experiments.py --model ho --device cpu \
  --n-epochs 25 \
  --loss-mode cov_weighted \
  --min-horizon 5 \
  --max-horizon 25 \
  --plot-every 5
```

Expected CUDA check for this setup: `cuda False`.

### CUDA Setup

Use this path for GPU training. The example uses CUDA 12.1 wheels; change the
PyTorch wheel index if the machine or cluster uses a different supported CUDA
version.

```bash
conda create -n rts-hybrid-cuda python=3.10 -y
conda activate rts-hybrid-cuda
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
python -m pip install numpy scipy matplotlib pandas
```

Check the install:

```bash
python -c "import torch, numpy, scipy, matplotlib, pandas; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
python ode_experiments.py --model ho --device cuda \
  --n-epochs 25 \
  --loss-mode cov_weighted \
  --min-horizon 5 \
  --max-horizon 25 \
  --plot-every 5
```

## Method Summary

For a dynamical system

```text
dy/dt = f(y)
```

all but one derivative are assumed to be known from physics. The missing
derivative is replaced by a neural network:

```text
dy_j/dt = physics_j(y)   for j != k
dy_k/dt = NN(y; theta)   for learned index k
```

Training in this repository follows the workflow below.

1. Either simulate ground-truth trajectories with noisy partial observations or load an experimental data window.
2. Run CKF/RTS smoothing with the **physics-only** model to estimate full
   latent trajectories from the noisy partial observations.
3. Train the hybrid neural ODE using rollout losses against measurements and
   smoother-inferred latent states.
4. Optionally continue training from a frozen smoother checkpoint using
   `ode_from_smoother.py`.
5. Integrate the learned hybrid ODE as a standalone simulator from chosen initial conditions.

CKF/RTS uses physics-only integrators; training and standalone rollout use the
hybrid integrators through the same `rk4_step` interface. Data generation
lives in `models/<model>.py`.

### Positivity-Constrained CKF Sigma Points

Several systems have non-negative or bounded states such as pressures,
volumes, and neural activations. To keep the CKF/RTS physically plausible,
`ckf.py` uses two safeguards:

```text
cap covariance for non-negative states -> generate sigma points
    -> clip out-of-bounds points with physics.clip_states(...)
    -> propagate valid sigma points
```

`_enforce_min_covariance(...)` keeps sigma-point spread reasonable near zero,
and `_clip_sigma_points(...)` clamps any remaining invalid samples before the
physics model evaluates them.

## Supported Systems

| System | States | Measured states | Learned derivative |
|------|------|------|------|
| Harmonic oscillator (`ho`) | `x, v` | `x` | `dv/dt` |
| Pendulum (`pendulum`) | `theta, omega` | `theta` | `domega/dt` |
| Brusselator (`brusselator`) | `x, y, z` | `y, z` | `dx/dt` |
| Hodgkin-Huxley (`hh`) | `V, m, h, n` | `V, h, n` | `dm/dt` |
| Retina (`retina`) | `P1, P2, P4, P5` | `P1, P2, P5` | `dP4/dt` |
| Yeast glycolysis (`yeast`) | `x1-x7` | `x1, x2, x3, x5, x6, x7` | `dx4/dt` |

## Neural Network Inputs

The learned derivative is not always given every state. The input rules live in
`models/base.py` and `models/<model>.py`.

For systems without a dependency mask, the NN receives the full state shown
below.

| System | Learned derivative | NN input |
|------|------|------|
| `ho` | `dv/dt` | `x, v` |
| `pendulum` | `domega/dt` | `theta, omega` |
| `brusselator` | `dx/dt` | `x, y, z` |
| `hh` | `dm/dt` | `V, m, h, n` |
| `retina` | `dP4/dt` by default | `P1, P2, P4, P5, Pin` |

For systems with masks, the dependency maps specify which state channels are
kept and which are zeroed before the NN sees the input.

| System | Code location | Learned derivative | NN input |
|------|------|------|------|
| `yeast` | `YEAST_DEPENDENCIES` in `models/yeast.py` | `dx4/dt` | `x3, x4, x5, x6, x7` |

## Quick Start

### Train a single system

All systems in the table above use the same command shape. Start with a short
CPU run, then replace `ho` with the desired preset. Edit its model config for
state selection or data-source changes.

```bash
# Synthetic baseline
python ode_experiments.py --model ho --device cpu \
  --n-epochs 25 \
  --loss-mode cov_weighted \
  --min-horizon 5 \
  --max-horizon 25 \
  --plot-every 5
```

### Continue from a frozen smoother checkpoint

Continuation-specific model defaults (such as the learned derivative) are in
`experiments/smoother_continuation/configuration.py`; this command only takes
run-wide controls.

```bash
python ode_from_smoother.py \
  -c ./results/retina/<run_tag>/epoch001/checkpoint.pt \
  --n-epochs 100 \
  --loss-mode standard \
  --min-horizon 10 \
  --max-horizon 80 \
  --device cuda
```

### Integrate a saved learned model

Use this when training is finished and you only want to load the saved NN
weights, choose an initial condition, and integrate the hybrid ODE forward.

```bash
python integrate_saved_model.py \
  --checkpoint ./results/ho/<run_tag>/epoch005/checkpoint.pt \
  --y0 1.0,0.0 \
  --duration 120 \
  --dt 0.01 \
  --output ./results/standalone_ho_rollout.npz
```

### Important shell scripts

```bash
bash scripts/submit_qpr_grid.sh retina
```

## Main Scripts

### Core training and modeling

| File | Purpose |
|------|---------|
| `models/__init__.py` and `models/<model>.py` | Experiment presets, system metadata, data generation, physics definitions, and training defaults |
| `models/base.py` and `models/<model>.py` | Hybrid ODE wrappers combining physics with a learned derivative |
| `nn_ode_models.py` | MLP architectures for the learned derivative |
| `ckf.py` | Forward filter and RTS smoother |
| `loss_rollout.py` | Rollout losses used for hybrid training |
| `ode_experiments.py` | Universal CLI parser for end-to-end training |
| `experiments/ode_training/` | End-to-end training session: data, system selection, network setup, CKF/RTS guidance, epochs, and saved artifacts |
| `ode_from_smoother.py` | CLI parser for continuation training from saved checkpoints |
| `experiments/smoother_continuation/` | Frozen-smoother continuation session: checkpoint loading, rebuilt model, epochs, and continuation artifacts |
| `integrate_saved_model.py` | Standalone integration from saved learned NN weights and an initial condition |
| `viz_ode.py` and `experiments/visualization/` | Stable plotting API plus general, phase-space, and rollout plot implementations |

### Where To Edit Hyperparameters

Most defaults are in `models/__init__.py` and `models/<model>.py`; command-line
flags in `ode_experiments.py` override them for a run. System/data selection is
implemented in `experiments/ode_training/systems.py`.

| Target setting | Edit this file | Default field | CLI override |
|------|------|------|------|
| Filter process noise | `models/<model>.py` | `CKFConfig.Q_diag` | edit config directly |
| Measurement noise covariance | `models/<model>.py` | `CKFConfig.R_diag` | edit config directly |
| Initial state covariance | `models/<model>.py` | `CKFConfig.P0_diag` | edit config directly |
| Training epochs | `models/__init__.py` and `models/<model>.py` | `TrainConfig.n_epochs` | `--n-epochs` |
| Training time horizon | `models/__init__.py` and `models/<model>.py` | `TrainConfig.T_train` | `--t-train` |
| Rollout-loss horizon | `models/__init__.py` and `models/<model>.py` | `TrainConfig.min_rollout_horizon`, `TrainConfig.max_rollout_horizon` | `--min-horizon`, `--max-horizon` |
| Pseudo-state loss weight | `models/__init__.py` and `models/<model>.py` | `TrainConfig.pseudo_weight` | `--pseudo-weight` |
| Full-state rollout loss weight | `models/__init__.py` and `models/<model>.py` | `TrainConfig.full_state_weight` | `--full-state-weight` |
| NN architecture and input scaling | `models/<model>.py` | `NNConfig.hidden_dims`, `activation`, `input_*` | edit config directly |
| NN learning rate, weight decay, gradient clipping | `models/<model>.py` | `NNConfig.lr`, `weight_decay`, `grad_clip` | `--lr`, `--weight-decay`, `--grad-clip` |
| Measurement noise used for generated data | `models/__init__.py` and `models/<model>.py` | `TrainConfig.meas_noise_std`, `TrainConfig.meas_snr` | `--meas-noise-std`, `--meas-snr` |
| Data source, sampling, and SciPy settings | `models/<model>.py` | `DataGenConfig` | edit config directly |
| Plot frequency | `models/__init__.py` and `models/<model>.py` | `PlotConfig.show_every` | `--plot-every` |
| Learned state(s) | `models/<model>.py` | `SystemConfig.replaced_index`, `replaced_indices` | edit config directly |

For model equations, edit `models/<model>.py`. For NN input channels, edit the
dependency maps or learned wrappers in `models/base.py` and `models/<model>.py`.

### Diagnostics

| File | Purpose |
|------|---------|
| `distancecheck.py` | Rollout and phase-portrait error utilities |
| `estimate_q_p0.py` | Estimate practical CKF `Q_diag` and `P0_diag` values from generated trajectories |

### Baselines

| File / Directory | Purpose |
|------|---------|
| `RBSE/` | CKF-based joint state/parameter estimation baseline |

## Adding a New ODE

Minimum checklist:

1. `models/<model>.py`: add the physics class, `rhs_true(...)`, `h(...)`,
   state clipping if needed, and `make_data_<system>(...)`.
2. `models/__init__.py` and `models/<model>.py`: define labels, measured
   states, learned states, `Q/R/P0`, NN dimensions, and training defaults.
3. `experiments/ode_training/systems.py` and the relevant data module: add the
   model name, data generator, physics selection, and tensors consumed by the
   training session.
4. `models/base.py` and `models/<model>.py`: add a custom learned wrapper only
   if the existing hybrid wrappers do not match the new state layout.
5. `ode_from_smoother.py`: optional continuation support for checkpoints that
   should train from frozen smoother targets.

## Using Experimental Data With A New Model

For real-data runs, the pipeline needs a time vector, measured channels, a
physics model for CKF/RTS prediction, and state metadata that marks which
states are observed versus inferred.

Short checklist:

1. Convert recordings to a simple NumPy format with aligned time and measured
   channels.
2. Add a loader that returns `t`, `meas`, and any available state targets in
   the same shape used by `ode_experiments.py`.
3. Set `meas_idx`, `pseudo_idx`, state labels, initial priors, bounds, `Q_diag`,
   `R_diag`, and `P0_diag` in `models/__init__.py` and `models/<model>.py` or during loader setup.
4. Register the model in `experiments/ode_training/systems.py`, then choose the
   learned derivative in `models/base.py` and `models/<model>.py`.
5. Start with short windows and conservative rollout horizons. Tune covariance
   defaults in the model preset and use `--min-horizon` / `--max-horizon` for
   a specific run.

## Repository Workflow

```text
models/<model>.py
    -> generate true trajectories + noisy partial measurements

ckf.py  (uses physics-only propagators from models/<model>.py)
    -> forward CKF pass: filter noisy measurements into state estimates
    -> backward RTS pass: smooth forward estimates into full latent trajectories

ode_experiments.py  (universal CLI options)
    -> experiments/ode_training/TrainingSession
       -> train the hybrid neural differential model against smoother targets

ode_from_smoother.py  (continuation CLI options)
    -> experiments/smoother_continuation/ContinuationSession
       -> train from frozen smoother guidance without rerunning CKF/RTS
```

## Output Layout

Current runs are organized by timestamped directories.

```text
code_backups/
  <system_or_artifact>/
    <timestamp>/
      *.py

results/
  <system_or_artifact>/
    <run_tag>/
      config.txt
      training_log.csv
      epoch001/
        checkpoint.pt
        *.npz
        *.png
        *.pdf
      epoch005/
        checkpoint.pt
        ...

  logs/
    *_training_log.csv

  paper_figures/
    *.pdf
    *.tex
    *.txt

saved_models/
  <system_or_artifact>/
    <run_tag>/
      epoch005.pt
      epoch010.pt
      ...
```

Notes:

- `ode_experiments.py` saves epoch-level checkpoints inside `results/.../epochXXX/`.
- `ode_from_smoother.py` additionally saves chainable epoch checkpoints under
  `saved_models/.../epochXXX.pt`.
- Most figures are saved together with `.npz` payloads for downstream analysis.

## HPC / Slurm

The repository includes Slurm launchers for sweeps and frozen-smoother
continuation:

```bash
bash scripts/submit_qpr_grid.sh retina
```

## Common CLI Options

Use argparse for a run's universal controls. Keep model choices in the model
preset so one model definition is reproducible without a long command line:

| Flag | Purpose |
|------|------|
| `--device {auto,cpu,cuda}` | Choose the execution device |
| `--n-epochs N` | Override the number of training epochs |
| `--loss-mode {standard,cov_weighted}` | Choose the rollout loss family |
| `--t-train T` | Override the generated-data time horizon |
| `--min-horizon N`, `--max-horizon N` | Set rollout-loss horizon scheduling |
| `--stride N` | Set rollout-start stride for loss windows |
| `--plot-every N` | Control plotting frequency |
| `--lr X` | Override the learned model learning rate |
| Model-specific choices | Set learned states, data files, filter covariances, NN architecture, and physics options in `models/<model>.py` |

## Code Pointers

| Area | Main files | What to read there |
|------|------|------|
| Experiment setup | `models/__init__.py`, `models/<model>.py`, `experiments/ode_training/` | Model defaults, measured states, learned derivative choices, and data generation |
| CKF/RTS smoother | `ckf.py` | `ckf_filter(...)`, `rts_smoother(...)`, sigma-point clipping, and covariance safeguards |
| Hybrid ODE wrappers | `models/base.py`, `models/<model>.py`, `nn_ode_models.py` | How physics derivatives and NN derivatives are combined, scripted, and rolled forward |
| Rollout losses | `loss_rollout.py` | Standard and covariance-weighted rollout objectives used to train the NN |
| Frozen smoother continuation | `ode_from_smoother.py`, `experiments/smoother_continuation/` | Reloads saved smoother tensors and trains without rerunning CKF/RTS |
| CKF covariance estimation | `estimate_q_p0.py`, `experiments/ode_training/estimation_config.py` | Estimates Q/P0 using a fixed estimator preset; the CLI selects models and output only |
| RBSE baseline | `RBSE/train.py`, `RBSE/configuration.py` | Uses model-owned shallow-NN and augmented-CKF defaults; the CLI selects models and run output only |
| Diagnostics | `viz_ode.py`, `experiments/visualization/`, `distancecheck.py`, `estimate_q_p0.py` | Plots, phase-portrait errors, rollout comparisons, and practical covariance estimates |
| RBSE baseline | `RBSE/*` | CKF-based joint state/parameter estimation baseline |
| Data utilities | `data/*`, `scripts/*` | Data helpers and cluster/batch launchers |

Training gradients come from rollout losses: the hybrid model is integrated
forward, predictions are compared to measured channels and smoother-derived
state targets, and the NN derivative is updated by backpropagation.

## JIT And TorchScript Notes

This code uses TorchScript/JIT for ODE wrappers because the same small methods
run repeatedly inside CKF/RTS propagation and RK4 rollout training. Scripting
reduces Python overhead, gives stricter tensor-only interfaces, and makes
learned ODE modules easier to save/load.

Because of that TorchScript source-inspection requirement, model files such as
`models/base.py` and `models/retina.py` are kept as normal source files even if
they are still long. They should be split later by moving whole classes or
helper functions into real importable `.py` modules, not by using the fragment
loader.

- The learned derivative MLP is built in `nn_ode_models.py` by
  `build_scripted_state_mlp(...)`, which returns `torch.jit.script(model)`.
- Hybrid ODE wrappers in `models/base.py` and `models/<model>.py` subclass `torch.jit.ScriptModule`
  and put integration methods behind `@torch.jit.script_method`.
- New learned wrappers should keep scripted methods tensor-only: no NumPy,
  SciPy, file I/O, or dynamic Python objects inside `f_full(...)`,
  `rk4_step(...)`, or rollout kernels.
- Scalar constants, masks, scales, and shifts should be registered as buffers
  when they are used inside scripted methods.
- `--torch-compile` is separate from TorchScript. It optionally compiles the
  rollout-loss path on supported CUDA setups; the model wrappers remain
  TorchScript modules either way.

## FAQ

### Where do I embed my physics?

Put the physics equations in the model file, usually by following
[`models/_template.py`](models/_template.py). The main pieces are the physics
class, `rhs_true(...)` for the full ODE right-hand side, `h(...)` for measured
outputs, optional `clip_states(...)` bounds, and the data generator. Existing
examples are in [`models/ho.py`](models/ho.py) and
[`models/yeast.py`](models/yeast.py).

Register defaults such as state labels, measured states, learned states,
covariances, NN size, and rollout settings in
[`models/__init__.py`](models/__init__.py). Wire the new model into
`experiments/ode_training/systems.py`; [`ode_experiments.py`](ode_experiments.py)
contains the universal CLI options and model configuration entry points.

### How do I add NN parts to make it hybrid?

The NN is the learned derivative that replaces one or more physics derivative
channels. The basic wrappers are in [`models/base.py`](models/base.py): they
call the physics RHS, call the learned model on selected state features, then
replace the chosen derivative before RK4 rollout.

The MLP itself is built in [`nn_ode_models.py`](nn_ode_models.py) by
`build_scripted_state_mlp(...)`. For systems that need custom masks or
multi-output learned derivatives, use the local wrappers in files such as
[`models/retina.py`](models/retina.py). The NN input-channel rules are
summarized above in the Neural Network Inputs section.

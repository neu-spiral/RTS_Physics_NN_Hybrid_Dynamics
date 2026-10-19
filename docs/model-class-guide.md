# Model Class Guide and Refactoring Map

This guide explains the responsibilities and public interfaces of the model
classes. Its purpose is to make model files easier to navigate now and to give
us a shared map before consolidating repeated code. It describes the current
architecture; it does not change training behavior or prescribe an immediate
refactor.

## The model stack at a glance

```text
models/<system>.py
  SystemConfig              model constants, bounds, and initial conditions
  Physics                   physics-only RHS used by CKF/RTS
  data functions            synthetic trajectory generation or experimental loading
  LearnedODE                physics RHS with selected derivative(s) replaced
  make_experiment()         one complete, mutable experiment preset

models/__init__.py
  EXPERIMENTS + get_experiment()   registry and fresh-copy factory

ode_experiments.py
  chooses a system, runs CKF/RTS, builds the learned ODE, and trains it
```

The important separation is between **physics-only** and **hybrid** dynamics:

- A `Physics` class is the model trusted by CKF/RTS before learning. It keeps
  every derivative mechanistic.
- A `LearnedODE` wrapper calls that physics RHS, then replaces one or more
  derivative channels with outputs from the neural network.

This separation is intentional. Do not put a learned derivative directly in a
`Physics` class unless the filter is also meant to use learned dynamics.

## Shared configuration classes

All of these are dataclasses in [`models/base.py`](../models/base.py). A
`make_experiment()` factory combines them into an `ExperimentConfig`.

| Class | Use it for | Attributes |
|---|---|---|
| `IntegratorConfig` | RK4 step policy shared by physics and hybrid wrappers. | `default_dt`, `max_dt`, `min_dt`, `n_substeps` |
| `CKFConfig` | CKF/RTS uncertainty settings. | `Q_diag` (per-state process noise), `R_diag` (per-measurement noise), `P0_diag` (per-state initial uncertainty) |
| `NNConfig` | The learned derivative MLP and its optimizer settings. | `in_dim`, `out_dim`, `hidden_dims`, `activation`, `input_scale`, `input_shift`, `lr`, `weight_decay`, `grad_clip` |
| `TrainConfig` | Data duration and rollout-loss schedule. | `T_train`, measurement-noise settings, `n_epochs`, rollout horizons, `stride`, and loss weights/toggles |
| `PlotConfig` | Diagnostic plot cadence and density. | `plot_stride`, `show_every`, `max_rollouts` |
| `DataGenConfig` | Synthetic-grid controls. | SciPy/grid options and `seed` |
| `ExperimentConfig` | A complete preset passed through the training pipeline. | `system_name`, all six configs above, `state_labels`, `title_prefix`, `meas_idx`, `pseudo_idx`, `uses_pin` |

### Model-specific `SystemConfig` classes

Each system module has one dataclass whose fields are the source of truth for
that system's physical parameters. Their names intentionally differ because
the equations differ.

| Class and file | What its attributes contain |
|---|---|
| `HOSystemConfig`, `PendulumSystemConfig` | Physical constant(s), true/prior initial states, and state bounds. |
| `BrusselatorSystemConfig`, `HodgkinHuxleySystemConfig`, `YeastSystemConfig` | Equation constants, true/prior initial states, and valid state ranges. |
| `RetinaSystemConfig` | Vascular/pressure parameters, initial states, pressure bounds, and optionally selected learned states. |

Use a `SystemConfig` for values that belong to the physical system. Use one of
the shared configs for solver, filter, network, training, plotting, or data
policy. Keeping that boundary clear is the first defense against configuration
sprawl.

## Physics classes

Every physics class is a `torch.jit.ScriptModule`, either directly or through
`BasePhysics2D`. TorchScript allows the inner ODE operations to be used in the
repeated filter and rollout loops.

### Required physics interface

The CKF/RTS code relies on this public contract:

| Attribute or method | Shape / meaning | Used by |
|---|---|---|
| `Q`, `R` | covariance matrices registered as buffers | `ckf.py` |
| `clip_states(y)` | clamps/repairs state tensor `(..., n_state)` | CKF sigma points and RK4 stages |
| `rhs_true(y)` | physics-only derivative with shape `(..., n_state)` | RK4 and hybrid wrappers |
| `rk4_step(y, dt)` | propagates one state or a batch | diagnostics and generic propagation |
| `propagate_batch(X, dt)` | propagates CKF sigma points, normally delegates to `rk4_step` | CKF/RTS |
| `h(X)` | measured-channel projection | CKF measurement update |

Models with an external drive additionally expose `rhs_true_with_drive(...)`
and/or a drive-aware RK4 method. Retina is the current example.

### Which physics base to choose

| Class | When to use it | Built-in attributes/methods |
|---|---|---|
| `BasePhysics2D` in `models/base.py` | A two-state model that measures state 0. Used by HO and Pendulum. | Registers `Q`, `R`, `dt`, `max_dt`, `min_dt`; supplies adaptive-substep RK4, `propagate_batch`, and `h`; subclass implements `clip_states` and `rhs_true`. |
| Direct `torch.jit.ScriptModule` | A model with more states, non-default measurements, drives, or special propagation. Used by Brusselator, HH, Yeast, and Retina. | The class must register its own buffers and implement the complete physics interface above. |

The different `super().__init__` signatures follow from this. A `BasePhysics2D`
subclass calls `super().__init__(integrator_cfg, ckf_cfg)`; a direct
`ScriptModule` subclass calls `super().__init__()` and registers those
quantities itself.

### Physics class reference

| File | Physics class | State size | Special responsibility |
|---|---|---:|---|
| `ho.py` | `HOPhysics` | 2 | Uses `BasePhysics2D`; observes position. |
| `pendulum.py` | `PendulumPhysics` | 2 | Uses `BasePhysics2D`; observes angle. |
| `brusselator.py` | `BrusselatorPhysics` | 3 | Custom measurement projection for `y, z`. |
| `hh.py` | `HodgkinHuxleyPhysics` | 4 | Conductance-model RHS and gating-variable bounds. |
| `yeast.py` | `YeastPhysics` | 7 | Glycolysis RHS and custom measured-state projection. |
| `retina.py` | `RetinaPhysics` | 4 | External `Pin` drive and pressure-network RHS. |

## Learned derivative classes

### The neural component

`StateODEMLP` in [`nn_ode_models.py`](../nn_ode_models.py) is the ordinary
PyTorch MLP used to predict derivative values, not next-state values.

| Class/function | Attributes | Method / result |
|---|---|---|
| `StateODEMLP` | `net`, optional `input_shift`, optional `input_scale` | `forward(features)` normalizes features and returns one or more derivatives. |
| `build_scripted_state_mlp(...)` | receives dimensions and normalization settings | returns a scripted `StateODEMLP` ready to live inside a TorchScript wrapper. |

### Shared learned-ODE wrappers

`LearnedODE2D`, `LearnedODE3D`, and `LearnedODE4D` in `models/base.py` are
the default wrappers. They own `physics`, `latent_model`, `max_dt`, `min_dt`,
and (for 3D/4D) `replaced_index`.

Their public methods are deliberately the same as the physics interface:

| Method | Responsibility |
|---|---|
| `f_full(state)` | calls `physics.rhs_true(state)` and replaces the configured derivative with `latent_model(...)`. |
| `rk4_step(y, dt)` | integrates the hybrid RHS with bounded substeps. |
| `propagate_batch(X, dt)` | provides batch propagation for a caller that needs it. |
| `h(X)`, `Q`, `R` | delegate measurement and covariance behavior to the physics module. |

The simple-model aliases are not separate implementations:

| Alias | Actual class |
|---|---|
| `LearnedODEHO`, `LearnedODEPendulum` | `LearnedODE2D` |
| `LearnedODEBrusselator` | `LearnedODE3D` |
| `LearnedODEHH` | `LearnedODE4D` |

### Custom learned wrappers

Write a custom wrapper only when the generic wrappers cannot express the
system's learned dynamics. Each one should retain the same public integration
interface so training, plotting, and checkpoint code can treat it uniformly.

| Class | Why it exists | Additional attributes/behavior |
|---|---|---|
| `LearnedODERetina4WithPin` | Retina needs a time-varying `Pin` drive. | `f_full_with_pin`, `rk4_step_with_pin`, and drive-aware propagation. |
| `LearnedODEYeast` | Yeast needs a dependency mask. | Selects the state channels passed to the derivative MLP. |

## Per-model helper functions

Each model module normally provides these functions in addition to classes:

| Function | Responsibility |
|---|---|
| `integrate_<system>_adaptive(...)` | NumPy/SciPy integration for synthetic ground truth. Its RHS must match `Physics.rhs_true`. Some systems use a differently named but equivalent data-generation path. |
| `make_data_<system>(exp, device)` | Generates synthetic truth and noisy measurements, or formats model-specific data. |
| `make_experiment()` | Returns the complete default `ExperimentConfig`. |
| `get_<system>_input_mask...` / dependency maps | Helpers that declare which state channels the MLP may use. |

`models/__init__.py` registers the output of each `make_experiment()` in
`EXPERIMENTS`. Always obtain a run configuration through `get_experiment()` or
`make_default_experiment()`; they return a deep copy so command-line overrides
do not mutate the registry's default object.

## Where repeated code currently lives

The long model files are mostly the result of these repeated, valid interface
implementations:

| Repeated concern | Current locations | Candidate shared abstraction |
|---|---|---|
| Registering `Q`, `R`, timestep buffers, and `meas_idx` | Most direct `ScriptModule` physics classes | A dimension-independent `BasePhysics` initializer. |
| RK4 substepping, batch propagation, and covariance delegation | Most direct physics and custom learned wrappers | Shared mixin/base with TorchScript-compatible hooks. |
| Measurement selection | Physics `h()` methods | A configurable index-selecting measurement helper. |
| SciPy solve/shape/noise boilerplate | `make_data_*` functions | Small data-generation utilities, while keeping each RHS local. |
| `f_full` plus hybrid RK4 implementation | Custom learned wrappers | A configurable multi-index/masked learned-ODE base. |
| Experiment construction | every `make_experiment()` | Keep per-model presets local; optionally add builder helpers only for repeated defaults. |

These are **refactoring candidates**, not evidence that all code should be
merged. Keep model-specific equations, bounds, drive logic, units, and special
numerical safeguards close to their model. A shared base should reduce only
mechanical duplication and must preserve the current CKF/RTS contract.

## Safe refactoring order

1. Add interface tests for each physics class: state shape, finite values,
   `h()` dimensions, `propagate_batch()` shape, and `Q`/`R` dimensions.
2. Extract only shared buffer registration and generic RK4 propagation into a
   dimension-independent base class.
3. Migrate one low-risk model first (for example Brusselator), compare its
   trajectories and CKF outputs against a saved baseline, then migrate others.
4. Extract learned-wrapper mechanics separately from physics mechanics.
5. Keep specialized Retina behavior as explicit overrides and document
   every deliberate exception here.

## Adding a model: class decision checklist

1. Create `<System>SystemConfig` for physical constants, initial conditions,
   bounds, and any model-specific state-selection settings.
2. If the model is two-state and observes only state 0, subclass
   `BasePhysics2D`; otherwise implement the complete physics interface as a
   direct `torch.jit.ScriptModule` subclass.
3. Implement `clip_states` and `rhs_true`, then make the synthetic NumPy RHS
   mathematically identical.
4. Use `LearnedODE2D/3D/4D` if its replacement behavior is sufficient. Add a
   custom learned wrapper only for a drive, dependency mask, residual term,
   multi-output special case, or different numerical treatment.
5. Add `make_data_<system>()`, `make_experiment()`, exports, registry entry,
   and a case in `_choose_system()` in `ode_experiments.py`.

[`models/_template.py`](../models/_template.py) is the starting point for a
simple two-state system. Update this guide whenever a new shared class or
intentional exception is introduced.

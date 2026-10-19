# Project Locations and Migration Status

This document records the project layout found on October 19, 2026. It is a
status snapshot, not a promise that the paths will exist on every machine.

## Which directory is the repository?

The canonical version-controlled checkout is:

```text
D:\RTS-HybridDynamics
```

It is a Git worktree on branch `main`, with remote
`https://github.com/neu-spiral/RTS-HybridDynamics.git`. The checked-out commit
at the time of this audit is:

```text
e1726ead431d0eeadddde8744846e7403fcc01ed
2026-08-04 01:07:45 -04:00
improve code structure
```

The local `main` and locally stored `origin/main` references point to that same
commit. A live remote check requires GitHub credentials and could not be
completed on this machine.

The newer experimental working tree is:

```text
Z:\v2
```

It contains work through October 19, 2026, but it is **not** a Git repository.
Consequently, newer timestamps in `Z:\v2` do not make it the canonical source
repository, and changes there have no commit history or remote backup.

Among the local Git repositories discovered within two directory levels of
`C:\Users\ahmet` and `D:\`, `D:\RTS-HybridDynamics` has the newest latest
commit and is the repository for this project.

## Existing uncommitted work

At the time of this audit the Git checkout already had user changes in:

- `experiments/smoother_continuation/configuration.py`
- `experiments/smoother_continuation/model_setup.py`
- `experiments/smoother_continuation/session.py`
- `experiments/smoother_continuation/training_loop.py`

These changes add continuation-run resumption controls and avoidance of
duplicate long-rollout plots. Treat them as active
work: do not overwrite or discard them during migration.

## Useful work present only in `Z:\v2`

The following later work should be migrated deliberately rather than copied
wholesale:

| Area | Principal files or directories in `Z:\v2` | Status in this repository |
|---|---|---|
| Retina one-state refinement | `one_state_retina_refinement.py` | Not migrated |
| Retina two-state replacement | `two_state_experiments.py` | Not migrated |
| Twenty-trial retina sweep | `run_retina_20_trial_sweep.py`, `summarize_retina_trials.py` | Not migrated |
| Paper figure generation | `retina_paper_plots.py`, `make_retina_paper_figures.py` | Not migrated |
| Recognition ODE/structured-NODE baseline | `structured_NODEs/` | Not migrated |
| Baseline figure overlays | `add_recognition_to_*_figure.py` | Not migrated |
| Selected retina artifacts | `PAPER_RESULTS_RETINA_20_TRIALS_20260912/` | External artifacts; do not copy wholesale |

The Git checkout already contains the cleaner modular architecture under
`models/` and `experiments/`. Newer `Z:\v2` scripts should therefore be ported
into that architecture instead of replacing the repository with the older
monolithic files.

## Recommended migration order

1. Preserve or commit the four existing continuation changes before a broad
   migration.
2. Port the oracle-boundary checks and retina experiment configuration into
   `models/retina.py` and `experiments/ode_training/`.
3. Add the one-state and two-state retina runners under a dedicated
   `experiments/paper_benchmarks/` package.
4. Add unit tests proving that removed states, removed derivatives, and the
   original removed right-hand side are inaccessible during training.
5. Port the 20-trial launcher and summarizer, using repository-relative paths.
6. Port plotting only after the result schema is stable. Plotting code should
   consume source manifests rather than contain machine-specific paths.
7. Store small manifests, summaries, captions, and exact paper metrics in Git.
   Keep checkpoints, raw trajectories, PDFs, PNGs, and large sweep folders in
   external artifact storage.

## Minimum migration checks

Before declaring `Z:\v2` retired, verify that the Git repository can:

- run one-state retina replacement with only `P4` latent;
- run two-state retina replacement with `P2` and `P4` latent;
- launch and resume seeded multi-trial sweeps;
- select runs using observed-channel validation metrics only;
- reproduce the selected metrics from a saved manifest;
- regenerate paper figures from explicitly recorded sources; and
- run the Recognition ODE baseline without hidden-state or hidden-RHS access.

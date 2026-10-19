# Oracle-Free Evaluation and Paper Provenance

This guide defines the experiment boundary used by the paper and records the
latest retina result provenance found in the external `Z:\v2` working tree.

## Oracle boundary

If a state is declared unknown or unmeasured, training must not use:

- measurements of that state;
- its ground-truth trajectory or initial condition;
- its ground-truth derivative;
- the original mechanistic equation that was removed and replaced;
- a loss term evaluated against the hidden ground truth; or
- hidden-state ground truth to select a seed, checkpoint, or hyperparameter.

Ground truth may be loaded **after training and selection** for final metrics
and reference curves. Run selection must use observed-channel validation
metrics or another criterion available from the stated measurements.

For each paper run, save explicit metadata fields equivalent to:

```text
training_supervision: measured_channels_only
oracle_hidden_state_used_for_training: false
oracle_hidden_derivative_used_for_training: false
original_hidden_rhs_evaluated_by_model: false
ground_truth_used_for_selection: false
```

A missing field is not evidence that a legacy run was oracle-free. Inspect its
code path before reusing it.

## Latest retina sweep

The latest complete retina sweep found during this audit is external to the
Git repository:

```text
Z:\v2\PAPER_RESULTS_RETINA_20_TRIALS_20260912
```

It contains 20 one-state `P4` replacement trials and 20 two-state `P2/P4`
replacement trials. Every optimization stage used at least 100 epochs. The
saved selection rule is `minimum_RMSE_against_measured_channels_only`, and the
manifest records `ground_truth_used_for_selection: false`.

### Paper-selected one-state results

The state-estimation and open-loop rows are selected independently because
they optimize different observed-data criteria.

| Result | Selected trial | Latent `P4` metric |
|---|---|---:|
| State estimate | `ONE_STATE/run01_d600_r120_w36_p5w4` | RMSE `1.6373965e-1` |
| Open loop | `ONE_STATE/run14_d1200_r300_w112_p5w14` | RMSE `5.1560992e-2` |
| Open loop | same as above | Hausdorff distance `3.5301208e-2` |

The authoritative compact record is:

```text
paper_figures/ONE_STATE/retina_one_state_paper_metrics.json
```

### Paper-selected two-state estimate

The selected state-estimation run is:

```text
TWO_STATE/run11_e1300_f015_r220_w48_lw0p75_p5w8_s11507/RETINA
```

Its latent-state RMSE values are:

| State | RMSE |
|---|---:|
| `P2` | `6.1149818e-1` |
| `P4` | `2.3541410e-1` |

Do not combine these estimate metrics with an open-loop trajectory from a
different trial without saying that the two results were selected separately.

## Recognition ODE / structured-NODE baseline

The external implementation is:

```text
Z:\v2\structured_NODEs\run_structured_nodes.py
```

For an oracle-free run, use `--fit-mode measurement` and require the metadata
flags listed above. The legacy derivative, rollout, and hybrid fit modes use
full-state supervision and must not be reported as partial-observation
Recognition ODE results.

Two retina state-estimation artifacts are relevant:

| Artifact | `P4` RMSE | Use |
|---|---:|---|
| `PAPER_RESULTS_ONE_STATE_REPLACEMENT_ORACLE_FREE/RETINA` | `1.5267394e0` | Earlier paper-table/figure source |
| `ORACLE_FREE_SMOKE/RETINA` | `9.8903829e-1` | Better verified measurement-only run used by the revised figure |

The revised `retina_time_series.pdf` therefore corresponds to
`9.8903829e-1`, not `1.5267394e0`. If the revised figure is kept, update the
Recognition ODE retina entry in the latent-state RMSE table so the table and
figure refer to the same selected run.

Do not substitute
`PAPER_RESULTS_ONE_STATE_REPLACEMENT_FINAL/RETINA`: despite its lower reported
`P4` RMSE, its legacy training path evaluates the hidden state and hidden
derivative and is not oracle-free.

## Required artifact bundle

Each result selected for a paper should have:

- the exact command, seed, epoch counts, and hyperparameters;
- the source commit hash, or an explicit note that the source was unversioned;
- measured and latent state indices;
- the training and selection oracle flags;
- state-estimation and open-loop metrics stored separately;
- raw time and trajectory arrays needed to recompute every metric;
- a figure-source manifest naming every plotted curve;
- the metric definition, state subset, and time horizon; and
- hashes for externally stored source artifacts when practical.

The September retina sweep already contains most of this information in
`trial_manifest.json`, `run_status.json`, `selection.json`, the summary CSVs,
and `paper_figures/plot_manifest.json`.

## Paper consistency checklist

Before submitting a regenerated table or figure:

1. Confirm the curve and table cell name the same run and evaluation mode.
2. Recompute the metric from the saved raw arrays.
3. Confirm the metric uses only the declared latent state(s).
4. Confirm the time horizon and units match the caption and table.
5. Verify all oracle flags and inspect legacy runs whose flags are absent.
6. Record when estimate and open-loop runs were selected independently.
7. Never alter a baseline trajectory merely to improve its appearance.
8. Preserve the untouched source figure when adding vector overlays.


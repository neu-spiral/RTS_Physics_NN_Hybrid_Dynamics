# RBSE Baseline

RBSE is the CKF-based baseline from Demirkaya et al., "Cubature Kalman Filter Based Training of Hybrid Differential Equation Recurrent Neural Network Physiological Dynamic Models" (EMBC 2021).

It augments the ODE state with neural-network weights, then uses CKF passes to jointly estimate system states and unknown learned dynamics without RTS smoothing, rollout loss, or backpropagation.

## Quick Use

```bash
python -m RBSE --models ho
python -m RBSE --models ho retina --epochs 1
python -m RBSE --list-models
```

Outputs are saved under `RBSE/results/<system>/<run_tag>/` by default.
Model-specific RBSE settings (the shallow NN and augmented-CKF noise scales)
are in `RBSE/configuration.py`.

## Slurm

```bash
bash RBSE/submit_grid.sh retina
sbatch RBSE/sbatch_job.sh ho 1e-6 1e-2 1 8
```

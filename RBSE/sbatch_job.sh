#!/usr/bin/env bash
# sbatch_job.sh -- Run a single RBSE training job on the cluster.
#
# Usage (submitted by submit_grid.sh, or manually):
#   sbatch RBSE/sbatch_job.sh ho 1e-6 1e-2 1 8
#   sbatch RBSE/sbatch_job.sh pendulum 1e-5 1e-1 3 "8,8"
#
# This script intentionally passes sweep values directly to ``run_rbse``;
# the normal RBSE command uses the model profiles in RBSE/configuration.py.
#
# Arguments:
#   $1  MODEL        System name (ho, pendulum, hh, brusselator, yeast, retina)
#   $2  Q_WEIGHT     Process noise for NN weights (e.g. 1e-6)
#   $3  P_WEIGHT     Initial covariance for NN weights (e.g. 1e-2)
#   $4  EPOCHS       Number of CKF passes (e.g. 1)
#   $5  HIDDEN_DIMS  Hidden layer sizes (e.g. "8" or "8,8")
#   $6  RUN_ID       Run identifier (seed/init variant)
#
# Environment overrides:
#   SEED=42          Set torch random seed before training

#SBATCH --job-name=rbse
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=zen2|skylake_avx512|cascadelake|broadwell|haswell|ivybridge|sapphirerapids
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/demirkaya.a/v2
#SBATCH --output=logs/rbse_%A.out
#SBATCH --error=logs/rbse_%A.err

set -euo pipefail

# If invoked without sbatch, submit itself as a batch job and exit
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[info] Not inside Slurm; submitting this script via sbatch..."
  exec sbatch "$0" "$@"
fi

MODEL="${1:-ho}"
Q_WEIGHT="${2:-1e-6}"
P_WEIGHT="${3:-1e-2}"
EPOCHS="${4:-1}"
HIDDEN_DIMS="${5:-8}"
RUN_ID="${6:-0}"
SEED="${SEED:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/cluster_python_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/ode_experiments.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/../ode_experiments.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    REPO_ROOT="/scratch/demirkaya.a/v2"
  fi
elif [[ -f "${PWD}/ode_experiments.py" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -f "/scratch/demirkaya.a/v2/ode_experiments.py" ]]; then
  REPO_ROOT="/scratch/demirkaya.a/v2"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${REPO_ROOT}/logs"

# Environment setup
cluster_setup_python_env RBSE base "${SLURM_CPUS_PER_TASK:-1}"

cd "${REPO_ROOT}"

echo "========================================"
echo "[ENV] Date: $(date)"
echo "[ENV] Node: $(hostname)"
echo "[ENV] Python: $(which python)"
echo "[ENV] Model: ${MODEL}"
echo "[ENV] Q_weight=${Q_WEIGHT} P_weight=${P_WEIGHT}"
echo "[ENV] Epochs=${EPOCHS} Hidden=${HIDDEN_DIMS} Run=${RUN_ID}"
if [[ -n "${SEED}" ]]; then
  echo "[ENV] Seed=${SEED}"
fi
echo "========================================"

python - <<PY
import torch
import numpy as np

# Set seed for reproducibility if provided
seed_str = "${SEED}"
run_id = int("${RUN_ID}")
if seed_str:
    seed = int(seed_str)
else:
    # Derive seed from run_id for reproducibility
    seed = 42 + run_id
torch.manual_seed(seed)
np.random.seed(seed)
print(f"[SEED] torch/numpy seed = {seed}")

from RBSE.train import run_rbse

hidden_dims = tuple(int(x) for x in "${HIDDEN_DIMS}".split(","))

run_rbse(
    model_name="${MODEL}",
    n_epochs=int("${EPOCHS}"),
    q_weight_scale=float("${Q_WEIGHT}"),
    p_weight_scale=float("${P_WEIGHT}"),
    hidden_dims=hidden_dims,
    run_id=run_id,
)
PY

echo "========================================"
echo "[DONE] $(date)"
echo "========================================"

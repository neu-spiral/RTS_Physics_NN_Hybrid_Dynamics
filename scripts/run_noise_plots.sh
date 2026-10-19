#!/usr/bin/env bash
# run_noise_plots.sh -- Generate noise comparison plots from completed runs.
#
# Usage:
#   bash scripts/run_noise_plots.sh              # all systems
#   bash scripts/run_noise_plots.sh ho,retina    # specific systems
#   N_RUNS=5 bash scripts/run_noise_plots.sh     # override number of runs

#SBATCH --job-name=noise_plots
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=zen2|skylake_avx512|cascadelake|broadwell|haswell|ivybridge|sapphirerapids
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/demirkaya.a/v2
#SBATCH --output=logs/noise_plots_%A.out
#SBATCH --error=logs/noise_plots_%A.err

set -euo pipefail

# If invoked outside Slurm, submit self
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec sbatch "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_python_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/noise_comparison.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/../noise_comparison.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    REPO_ROOT="/scratch/demirkaya.a/v2"
  fi
elif [[ -f "${PWD}/noise_comparison.py" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -f "/scratch/demirkaya.a/v2/noise_comparison.py" ]]; then
  REPO_ROOT="/scratch/demirkaya.a/v2"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SYSTEM="${1:-all}"
N_RUNS="${N_RUNS:-5}"
FMT="${FMT:-pdf}"
RESULTS_DIR="${RESULTS_DIR:-results}"

mkdir -p "${REPO_ROOT}/logs"

# ── Environment setup ────────────────────────────────────────────────────────
cluster_setup_python_env NOISE_PLOTS base "${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"

echo "========================================"
echo "[ENV] Date:    $(date)"
echo "[ENV] Node:    $(hostname)"
echo "[ENV] Python:  $(which python)"
echo "[ENV] Python version: $(python --version 2>&1)"
echo "[ENV] System:  ${SYSTEM}"
echo "[ENV] N_runs:  ${N_RUNS}"
echo "[ENV] Format:  ${FMT}"
echo "[ENV] Results: ${RESULTS_DIR}"
echo "========================================"

python noise_comparison.py \
    --system "${SYSTEM}" \
    --n-runs "${N_RUNS}" \
    --fmt "${FMT}" \
    --results-dir "${RESULTS_DIR}"

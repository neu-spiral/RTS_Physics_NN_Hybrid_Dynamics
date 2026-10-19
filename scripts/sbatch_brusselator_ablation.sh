#!/usr/bin/env bash
# sbatch_brusselator_ablation.sh -- Run one Brusselator ablation variant on Slurm.
#
# Usage:
#   bash scripts/sbatch_brusselator_ablation.sh full_method
#   sbatch scripts/sbatch_brusselator_ablation.sh no_rts_guidance
#
# Environment overrides (set before calling):
#   ABLATION_PARTITION    (optional Slurm partition override)
#   ABLATION_GRES         (default: gpu:1)
#   ABLATION_TIME         (optional Slurm time override)
#   ABLATION_MEM          (optional Slurm memory override)
#   ABLATION_CPUS         (optional Slurm CPU override)
#   ABLATION_CONSTRAINT   (optional Slurm constraint override)
#   ABLATION_EPOCHS       (default: 25)
#   ABLATION_T_TRAIN      (default: 40)
#   ABLATION_MIN_H        (optional)
#   ABLATION_MAX_H        (optional)
#   ABLATION_ROLL_STRIDE  (optional)
#   ABLATION_PSEUDO_WEIGHT (optional)
#   ABLATION_PLOT_EVERY   (default: 1)
#   ABLATION_PLOT_STRIDE  (default: 1)
#   ABLATION_MAX_ROLLOUTS (default: 12)
#   ABLATION_OUTPUT_ROOT  (default: results/ablations/brusselator)
#   ABLATION_SESSION_DIR  (optional exact session directory)
#   ABLATION_SEED         (default: 1234)

#SBATCH --job-name=bruss_ablation
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=zen2|skylake_avx512|cascadelake|broadwell|haswell|ivybridge|sapphirerapids
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/demirkaya.a/v2
#SBATCH --output=logs/brusselator_ablation_%x_%A.out
#SBATCH --error=logs/brusselator_ablation_%x_%A.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch_args=()
  [[ -n "${ABLATION_PARTITION:-}" ]] && sbatch_args+=(--partition="${ABLATION_PARTITION}")
  [[ -n "${ABLATION_GRES:-}" ]] && sbatch_args+=(--gres="${ABLATION_GRES}")
  [[ -n "${ABLATION_TIME:-}" ]] && sbatch_args+=(--time="${ABLATION_TIME}")
  [[ -n "${ABLATION_MEM:-}" ]] && sbatch_args+=(--mem="${ABLATION_MEM}")
  [[ -n "${ABLATION_CPUS:-}" ]] && sbatch_args+=(--cpus-per-task="${ABLATION_CPUS}")
  [[ -n "${ABLATION_CONSTRAINT:-}" ]] && sbatch_args+=(--constraint="${ABLATION_CONSTRAINT}")
  exec sbatch "${sbatch_args[@]}" "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_python_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/ablation_brusselator.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/../ablation_brusselator.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    REPO_ROOT="/scratch/demirkaya.a/v2"
  fi
elif [[ -f "${PWD}/ablation_brusselator.py" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -f "/scratch/demirkaya.a/v2/ablation_brusselator.py" ]]; then
  REPO_ROOT="/scratch/demirkaya.a/v2"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

VARIANT="${1:-full_method}"
shift || true

case "${VARIANT}" in
  full_method|no_latent_state_loss|no_covariance_weighting|no_rts_guidance)
    ;;
  *)
    echo "[error] Unknown ablation variant: ${VARIANT}"
    echo "        Expected one of: full_method, no_latent_state_loss, no_covariance_weighting, no_rts_guidance"
    exit 2
    ;;
esac

mkdir -p "${REPO_ROOT}/logs"

cluster_setup_python_env ABLATION base "${SLURM_CPUS_PER_TASK:-8}"
if command -v module >/dev/null 2>&1; then
  module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

cd "${REPO_ROOT}"

EPOCHS="${ABLATION_EPOCHS:-25}"
T_TRAIN="${ABLATION_T_TRAIN:-40}"
MIN_H="${ABLATION_MIN_H:-}"
MAX_H="${ABLATION_MAX_H:-}"
ROLL_STRIDE="${ABLATION_ROLL_STRIDE:-}"
PSEUDO_WEIGHT="${ABLATION_PSEUDO_WEIGHT:-}"
PLOT_EVERY="${ABLATION_PLOT_EVERY:-1}"
PLOT_STRIDE="${ABLATION_PLOT_STRIDE:-1}"
MAX_ROLLOUTS="${ABLATION_MAX_ROLLOUTS:-12}"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-results/ablations/brusselator}"
SESSION_DIR="${ABLATION_SESSION_DIR:-}"
SEED="${ABLATION_SEED:-1234}"

echo "========================================"
echo "[ENV] Date:         $(date)"
echo "[ENV] Node:         $(hostname)"
echo "[ENV] Python:       $(python --version 2>&1)"
echo "[ENV] Variant:      ${VARIANT}"
echo "[ENV] GRES:         ${SLURM_JOB_GRES:-gpu:1}"
echo "[ENV] Epochs:       ${EPOCHS}"
echo "[ENV] T_train:      ${T_TRAIN}"
echo "[ENV] Plot every:   ${PLOT_EVERY}"
echo "[ENV] Plot stride:  ${PLOT_STRIDE}"
echo "[ENV] Max rollouts: ${MAX_ROLLOUTS}"
echo "[ENV] Seed:         ${SEED}"
if [[ -n "${SESSION_DIR}" ]]; then
  echo "[ENV] Session dir:  ${SESSION_DIR}"
else
  echo "[ENV] Output root:  ${OUTPUT_ROOT}"
fi
echo "========================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

args=(
  --variants "${VARIANT}"
  --epochs "${EPOCHS}"
  --t-train "${T_TRAIN}"
  --plot-every "${PLOT_EVERY}"
  --plot-stride "${PLOT_STRIDE}"
  --max-rollouts "${MAX_ROLLOUTS}"
  --seed "${SEED}"
)

if [[ -n "${SESSION_DIR}" ]]; then
  args+=(--session-dir "${SESSION_DIR}")
else
  args+=(--output-dir "${OUTPUT_ROOT}")
fi

[[ -n "${MIN_H}" ]] && args+=(--min-rollout-horizon "${MIN_H}")
[[ -n "${MAX_H}" ]] && args+=(--max-rollout-horizon "${MAX_H}")
[[ -n "${ROLL_STRIDE}" ]] && args+=(--stride "${ROLL_STRIDE}")
[[ -n "${PSEUDO_WEIGHT}" ]] && args+=(--pseudo-weight "${PSEUDO_WEIGHT}")

python ablation_brusselator.py "${args[@]}" "$@"

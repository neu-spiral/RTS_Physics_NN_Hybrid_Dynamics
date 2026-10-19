#!/usr/bin/env bash
# Run a single CKF diagonal configuration for a model on CPU.

#SBATCH --job-name=hybrid_qpr
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=zen2|skylake_avx512|cascadelake|broadwell|haswell|ivybridge|sapphirerapids
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/demirkaya.a/v2
#SBATCH --output=logs/qpr_job_%A.out
#SBATCH --error=logs/qpr_job_%A.err

set -euo pipefail

# If not running under Slurm, stop and ask user to submit via sbatch
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[error] Not in a Slurm allocation. Please submit with:"
  echo "  sbatch $(basename "$0") <model> [Q_scale] [R_scale] [P_scale] [LR]"
  echo "  model may be retina2 to learn both P2 and P4 retina dynamics"
  exit 1
fi

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
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

MODEL="${1:-ho}"
Q_SCALE="${2:-1}"
R_SCALE="${3:-1}"
P_SCALE="${4:-1}"
LR="${5:-3e-3}"
if [[ -n "${SWEEP_EPOCHS:-}" ]]; then
  SWEEP_EPOCHS="${SWEEP_EPOCHS}"
else
  SWEEP_EPOCHS="100"
fi

mkdir -p "${REPO_ROOT}/logs"

# Environment setup. Keep this in the job script so it runs on the compute node.
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load anaconda3/2024.06
fi

for conda_sh in \
  /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh \
  /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
do
  if [[ -f "${conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
    break
  fi
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

source activate torchtf39

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/demirkaya.a}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/cache"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/cache/pip"
export MPLCONFIGDIR="${SCRATCH_ROOT}/cache/matplotlib"
export CONDA_NOTICES_CACHE_DIR="${SCRATCH_ROOT}/cache/conda/notices"
mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${CONDA_NOTICES_CACHE_DIR}"

# Legacy CUDA 9.0 settings, kept disabled.
# export CUDA_HOME=/home/demirkaya.a/cuda-9.0
# export PATH=$PATH:/home/demirkaya.a/cuda-9.0/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/demirkaya.a/cuda-9.0/lib64

export TF_CPP_MIN_LOG_LEVEL=0
CUDA_118_HOME="${CUDA_118_HOME:-${HOME}/cuda-11.8}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_118_HOME}"
CUDA_118_LD_PATH="${CUDA_118_HOME}/lib64:${CUDA_118_HOME}/extras/CUPTI/lib64"
export LD_LIBRARY_PATH="${CUDA_118_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Do not preload CUDA stub libraries. They are link-time placeholders, not the
# runtime driver, and cause ld.so errors when absent on compute nodes.
if [[ "${LD_PRELOAD:-}" == *"/stubs/libcuda.so"* ]]; then
  unset LD_PRELOAD
fi

export ODE_CPU_THREADS="${SLURM_CPUS_PER_TASK:-${ODE_CPU_THREADS:-1}}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

echo "========================================"
echo "[ENV] Date: $(date)"
echo "[ENV] Node: $(hostname)"
echo "[ENV] Python: $(which python)"
echo "[ENV] Python version: $(python --version 2>&1)"
echo "[ENV] Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "[ENV] CUDA_118_HOME=${CUDA_118_HOME}"
echo "[ENV] XLA_FLAGS=${XLA_FLAGS}"
echo "[ENV] Model: ${MODEL}"
echo "[ENV] Q_scale=${Q_SCALE} R_scale=${R_SCALE} P_scale=${P_SCALE} LR=${LR}"
echo "========================================"

scale_diag () {
  local which="$1"  # Q, R, or P0
  local scale="$2"
  python - <<PY
from models import make_default_experiment
exp = make_default_experiment("${MODEL}")
ck = exp.ckf_cfg
base = {
    "Q": ck.Q_diag,
    "R": ck.R_diag,
    "P0": ck.P0_diag,
}["${which}"]
scale = float("${scale}")
print(",".join(f"{x*scale:g}" for x in base))
PY
}

export ODE_CKF_Q_DIAG="$(scale_diag Q "${Q_SCALE}")"
export ODE_CKF_R_DIAG="$(scale_diag R "${R_SCALE}")"
export ODE_CKF_P0_DIAG="$(scale_diag P0 "${P_SCALE}")"
echo "[SWEEP] Q_diag=${ODE_CKF_Q_DIAG}"
echo "[SWEEP] R_diag=${ODE_CKF_R_DIAG}"
echo "[SWEEP] P0_diag=${ODE_CKF_P0_DIAG}"

echo "========================================"
echo "[RUN] model=${MODEL} lr=${LR} Q=${ODE_CKF_Q_DIAG} R=${ODE_CKF_R_DIAG} P0=${ODE_CKF_P0_DIAG}"
echo "========================================"

python - <<PY
from models import make_default_experiment
from ode_experiments import train_experiment
exp = make_default_experiment("${MODEL}")
exp.train_cfg.n_epochs = int("${SWEEP_EPOCHS}")
exp.nn_cfg.lr = float("${LR}")
exp.ckf_q_diag_arg = "${ODE_CKF_Q_DIAG}"
exp.ckf_r_diag_arg = "${ODE_CKF_R_DIAG}"
exp.ckf_p0_diag_arg = "${ODE_CKF_P0_DIAG}"
train_experiment(exp)
PY

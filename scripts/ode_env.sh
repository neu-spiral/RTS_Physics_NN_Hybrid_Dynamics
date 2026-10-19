#!/usr/bin/env bash
# Load the cluster environment needed by ode_experiments.py.
# Usage:
#   source scripts/ode_env.sh cpu
#   source scripts/ode_env.sh gpu
#   bash scripts/ode_env.sh gpu python ode_experiments.py --model retina --device cuda

sourced=0
[[ "${BASH_SOURCE[0]}" != "$0" ]] && sourced=1

mode="${1:-cpu}"
case "${mode}" in
  cpu|gpu|auto)
    if [[ "${sourced}" == "0" ]]; then
      shift || true
    fi
    ;;
  *) mode="cpu" ;;
esac

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/demirkaya.a}"
ODE_CONDA_ENV="${ODE_CONDA_ENV:-${SCRATCH_ROOT}/conda/envs/py39gpu}"

export HOME="${SCRATCH_ROOT}/home"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export CONDARC="${SCRATCH_ROOT}/conda/condarc"
export CONDA_ENVS_PATH="${SCRATCH_ROOT}/conda/envs"
export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/conda/pkgs"
export CONDA_NOTICES_CACHE_DIR="${SCRATCH_ROOT}/cache/conda/notices"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/cache"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/cache/pip"
export MPLCONFIGDIR="${SCRATCH_ROOT}/cache/matplotlib"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
if [[ "${LD_PRELOAD:-}" == *"/stubs/libcuda.so"* ]]; then
  echo "[ode_env] Removing CUDA stub from LD_PRELOAD: ${LD_PRELOAD}"
  unset LD_PRELOAD
fi

mkdir -p \
  "${HOME}" \
  "${TMPDIR}" \
  "${CONDA_ENVS_PATH}" \
  "${CONDA_PKGS_DIRS}" \
  "${CONDA_NOTICES_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${SCRATCH_ROOT}/conda"

cat > "${CONDARC}" <<EOF
envs_dirs:
  - ${CONDA_ENVS_PATH}
pkgs_dirs:
  - ${CONDA_PKGS_DIRS}
auto_activate_base: false
notify_outdated_conda: false
EOF

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load anaconda3/2024.06 2>/dev/null || true
  if [[ "${mode}" == "gpu" || "${mode}" == "auto" ]]; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
  fi
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

if ! command -v conda >/dev/null 2>&1; then
  echo "[ode_env] conda is not available after loading anaconda3/2024.06" >&2
  return 1 2>/dev/null || exit 1
fi

conda activate "${ODE_CONDA_ENV}"

case "${mode}" in
  cpu) export ODE_DEVICE=cpu ;;
  gpu) export ODE_DEVICE=cuda ;;
esac
export ODE_CPU_THREADS="${SLURM_CPUS_PER_TASK:-${ODE_CPU_THREADS:-1}}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import sys
import torch
from torch.amp import autocast, GradScaler
print(f"[ode_env] python={sys.executable}")
print(f"[ode_env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

if [[ "${sourced}" == "0" && "$#" -gt 0 ]]; then
  exec "$@"
fi

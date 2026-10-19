#!/usr/bin/env bash
# srun_qpr_grid.sh -- Run a grid of CKF/NN hyperparameter runs inside a single Slurm allocation.
#
# Usage:
#   bash scripts/srun_qpr_grid.sh
#   srun --partition=short --cpus-per-task=4 --mem=32G --time=04:00:00 bash scripts/srun_qpr_grid.sh
#   srun ... bash scripts/srun_qpr_grid.sh retina
#   Q_SCALES=1,2 P_SCALES=0.5,1 srun ... bash scripts/srun_qpr_grid.sh retina
#
# Grid defaults (can override via env):
#   Q_SCALES=0.5,1,2
#   R_SCALES=1
#   P_SCALES=0.5,1,2
#   LR_LIST=1e-3,3e-3,5e-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_python_env.sh"

# ── Repo root detection ──
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/ode_experiments.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${PWD}/ode_experiments.py" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -f "/scratch/demirkaya.a/v2/ode_experiments.py" ]]; then
  REPO_ROOT="/scratch/demirkaya.a/v2"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ── Parse arguments ──
MODEL_ARG="${1:-all}"

read_csv() {
  IFS=',' read -ra arr <<<"$1"
  echo "${arr[@]}"
}

LR_LIST=(3e-3)
Q_SCALES=(1)
R_SCALES=(1)
P_SCALES=(1)

if [[ "${MODEL_ARG}" == "all" ]]; then
  MODELS=(ho pendulum hh brusselator yeast retina retina2)
else
  IFS=',' read -ra MODELS <<<"${MODEL_ARG}"
fi

if [[ "${QPR_AUTO_SRUN:-1}" == "1" && -z "${SLURM_JOB_ID:-}" && "${QPR_IN_SRUN:-0}" != "1" ]]; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "[error] srun not found. Set QPR_AUTO_SRUN=0 to run locally."
    exit 1
  fi

  srun_args=(
    --partition="${QPR_SLURM_PARTITION:-short}"
    --ntasks=1
    --cpus-per-task="${QPR_SLURM_CPUS:-4}"
    --mem="${QPR_SLURM_MEM:-32G}"
    --time="${QPR_SLURM_TIME:-04:00:00}"
    --unbuffered
  )
  [[ -n "${QPR_SLURM_CONSTRAINT:-}" ]] && srun_args+=(--constraint="${QPR_SLURM_CONSTRAINT}")
  [[ -n "${QPR_SLURM_ACCOUNT:-}" ]] && srun_args+=(--account="${QPR_SLURM_ACCOUNT}")
  [[ -n "${QPR_SLURM_QOS:-}" ]] && srun_args+=(--qos="${QPR_SLURM_QOS}")

  echo "========================================"
  echo "Requesting Slurm allocation via srun"
  echo "srun args: ${srun_args[*]}"
  echo "========================================"
  exec env QPR_IN_SRUN=1 srun "${srun_args[@]}" bash "$0" "$@"
fi

# ── Environment setup ──
cluster_setup_python_env QPR base "${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

echo "========================================"
echo "[ENV] Date: $(date)"
echo "[ENV] Node: $(hostname)"
echo "[ENV] Python exe: $(which python)"
echo "[ENV] Python: $(python --version 2>&1)"
echo "[ENV] SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "[ENV] Systems: ${MODELS[*]}"
echo "[ENV] Grid: |Q|=${#Q_SCALES[@]}  |R|=${#R_SCALES[@]}  |P|=${#P_SCALES[@]}  |LR|=${#LR_LIST[@]}"
echo "========================================"

# ── Helper: scale diagonal ──
scale_diag () {
  local model="$1" which="$2" scale="$3"
  python - <<PY
from models import make_default_experiment
exp = make_default_experiment("${model}")
ck = exp.ckf_cfg
base = {"Q": ck.Q_diag, "R": ck.R_diag, "P0": ck.P0_diag}["${which}"]
scale = float("${scale}")
print(",".join(f"{x*scale:g}" for x in base))
PY
}

# ── Grid loop ──
TOTAL=0
DONE=0
FAILED=0

for model in "${MODELS[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for q in "${Q_SCALES[@]}"; do
      for r in "${R_SCALES[@]}"; do
        for p in "${P_SCALES[@]}"; do
          TOTAL=$((TOTAL + 1))
        done
      done
    done
  done
done

echo "Total runs: ${TOTAL}"
echo "========================================"

IDX=0
for model in "${MODELS[@]}"; do
  # Determine epochs
  if [[ -n "${SWEEP_EPOCHS:-}" ]]; then
    EPOCHS="${SWEEP_EPOCHS}"
  else
    EPOCHS="100"
  fi

  for lr in "${LR_LIST[@]}"; do
    for q in "${Q_SCALES[@]}"; do
      for r in "${R_SCALES[@]}"; do
        for p in "${P_SCALES[@]}"; do
          IDX=$((IDX + 1))

          export ODE_CKF_Q_DIAG="$(scale_diag "${model}" Q "${q}")"
          export ODE_CKF_R_DIAG="$(scale_diag "${model}" R "${r}")"
          export ODE_CKF_P0_DIAG="$(scale_diag "${model}" P0 "${p}")"

          echo ""
          echo "========================================"
          echo "[RUN ${IDX}/${TOTAL}] model=${model} lr=${lr} Q_s=${q} R_s=${r} P_s=${p}"
          echo "[DIAG] Q=${ODE_CKF_Q_DIAG}"
          echo "[DIAG] R=${ODE_CKF_R_DIAG}"
          echo "[DIAG] P0=${ODE_CKF_P0_DIAG}"
          echo "========================================"

          if python - <<PY
from models import make_default_experiment
from ode_experiments import train_experiment
exp = make_default_experiment("${model}")
exp.train_cfg.n_epochs = int("${EPOCHS}")
exp.nn_cfg.lr = float("${lr}")
train_experiment(exp)
PY
          then
            DONE=$((DONE + 1))
            echo "[OK] ${IDX}/${TOTAL} completed"
          else
            FAILED=$((FAILED + 1))
            echo "[FAIL] ${IDX}/${TOTAL} model=${model} lr=${lr} Q=${q} R=${r} P=${p}"
          fi
        done
      done
    done
  done
done

echo ""
echo "========================================"
echo "[SUMMARY] Total=${TOTAL}  OK=${DONE}  Failed=${FAILED}"
echo "========================================"

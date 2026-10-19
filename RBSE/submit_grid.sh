#!/usr/bin/env bash
# submit_grid.sh -- Submit a grid of RBSE training jobs to Slurm.
#
# Sweeps over: systems x q_weight x p_weight x epochs x hidden_dims x runs
# Each combination becomes one Slurm job.
#
# Usage (login node):
#   bash RBSE/submit_grid.sh                          # all systems, default grid
#   bash RBSE/submit_grid.sh ho                       # single system
#   bash RBSE/submit_grid.sh ho,pendulum              # multiple systems
#   bash RBSE/submit_grid.sh all                      # all systems
#
# Environment overrides (customize the grid):
#   Q_WEIGHTS="1e-7,1e-6,1e-5"  bash RBSE/submit_grid.sh ho
#   P_WEIGHTS="1e-3,1e-2,1e-1"  bash RBSE/submit_grid.sh ho
#   N_RUNS=10                   bash RBSE/submit_grid.sh ho
#   EPOCHS=3                    bash RBSE/submit_grid.sh ho
#   HIDDEN="8,8;16,16"          bash RBSE/submit_grid.sh ho
#
# Default grid (per model; override via env):
#   Base (ho/pendulum/brusselator/yeast):
#     Q_WEIGHTS = 1e-7,1e-6,1e-5   | P_WEIGHTS = 1e-3,1e-2,1e-1 | HIDDEN=8,8 | EPOCHS=1
#   HH:
#     Q_WEIGHTS = 1e-6,1e-5,1e-4   | P_WEIGHTS = 1e-3,1e-2,1e-1 | HIDDEN=8,8;16,16 | EPOCHS=3
#   Retina:
#     Q_WEIGHTS = 1e-6,1e-5,5e-5   | P_WEIGHTS = 1e-2,1e-1,1e0  | HIDDEN=8,8;16,16 | EPOCHS=3
#   N_RUNS default: 4 (per config)
# Override any list globally via env Q_WEIGHTS/P_WEIGHTS/HIDDEN; override runs via N_RUNS; override epochs via EPOCHS.

set -euo pipefail

MODEL_ARG="${1:-all}"

read_csv() {
  IFS=',' read -ra arr <<<"$1"
  echo "${arr[@]}"
}

# Hidden layer configs can contain commas (e.g., "8,8"), so ';' is used to
# separate configs: "8,8;16,16". If no ';' is present, treat the entire
# string as a single config.
read_hidden_cfgs() {
  local s="$1"
  if [[ "$s" == *";"* ]]; then
    IFS=';' read -ra arr <<<"$s"
  else
    arr=("$s")
  fi
  echo "${arr[@]}"
}

# Base defaults
BASE_Q_DEF="1e-7,1e-6,1e-5"
BASE_P_DEF="1e-3,1e-2,1e-1"
BASE_H_DEF="8,8"
BASE_EPOCHS_DEF="1"

# Model-specific defaults
HH_Q_DEF="1e-6,1e-5,1e-4"
HH_P_DEF="1e-3,1e-2,1e-1"
HH_H_DEF="8,8;16,16"
HH_EPOCHS_DEF="3"

RET_Q_DEF="1e-6,1e-5,5e-5"
RET_P_DEF="1e-2,1e-1,1e0"
RET_H_DEF="8,8;16,16"
RET_EPOCHS_DEF="3"

EPOCHS_ENV="${EPOCHS:-}"
N_RUNS="${N_RUNS:-4}"

if [[ "${MODEL_ARG}" == "all" ]]; then
  MODELS=(ho pendulum hh brusselator yeast retina)
else
  IFS=',' read -ra MODELS <<<"${MODEL_ARG}"
fi

mkdir -p logs

echo "========================================"
echo "RBSE Grid Submission"
echo "========================================"
echo "Systems:     ${MODELS[*]}"
echo "Runs/config: ${N_RUNS}"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/sbatch_job.sh"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: Cannot find ${JOB_SCRIPT}"
  exit 1
fi

count=0
for model in "${MODELS[@]}"; do
  # Pick defaults per model unless overridden by env
  if [[ -n "${Q_WEIGHTS:-}" ]]; then
    Q_LIST=($(read_csv "${Q_WEIGHTS}"))
  else
    case "${model}" in
      hh)     Q_LIST=($(read_csv "${HH_Q_DEF}")) ;;
      retina) Q_LIST=($(read_csv "${RET_Q_DEF}")) ;;
      *)      Q_LIST=($(read_csv "${BASE_Q_DEF}")) ;;
    esac
  fi

  if [[ -n "${P_WEIGHTS:-}" ]]; then
    P_LIST=($(read_csv "${P_WEIGHTS}"))
  else
    case "${model}" in
      hh)     P_LIST=($(read_csv "${HH_P_DEF}")) ;;
      retina) P_LIST=($(read_csv "${RET_P_DEF}")) ;;
      *)      P_LIST=($(read_csv "${BASE_P_DEF}")) ;;
    esac
  fi

  if [[ -n "${HIDDEN:-}" ]]; then
    H_LIST=($(read_hidden_cfgs "${HIDDEN}"))
  else
    case "${model}" in
      hh)     H_LIST=($(read_hidden_cfgs "${HH_H_DEF}")) ;;
      retina) H_LIST=($(read_hidden_cfgs "${RET_H_DEF}")) ;;
      *)      H_LIST=($(read_hidden_cfgs "${BASE_H_DEF}")) ;;
    esac
  fi

  if [[ -n "${EPOCHS_ENV}" ]]; then
    EPOCHS_MODEL="${EPOCHS_ENV}"
  else
    case "${model}" in
      hh)     EPOCHS_MODEL="${HH_EPOCHS_DEF}" ;;
      retina) EPOCHS_MODEL="${RET_EPOCHS_DEF}" ;;
      *)      EPOCHS_MODEL="${BASE_EPOCHS_DEF}" ;;
    esac
  fi

  n_q=${#Q_LIST[@]}
  n_p=${#P_LIST[@]}
  n_h=${#H_LIST[@]}
  total_per_model=$((n_q * n_p * n_h * N_RUNS))
  echo "Model ${model}: |Q|=${n_q} |P|=${n_p} |H|=${n_h} epochs=${EPOCHS_MODEL} runs=${N_RUNS} => ${total_per_model} jobs"

  for q in "${Q_LIST[@]}"; do
    for p in "${P_LIST[@]}"; do
      for h in "${H_LIST[@]}"; do
        for run in $(seq 0 $((N_RUNS - 1))); do
          jobid=$(sbatch --parsable \
            --job-name="rbse_${model}_r${run}" \
            "${JOB_SCRIPT}" "${model}" "${q}" "${p}" "${EPOCHS_MODEL}" "${h}" "${run}")
          echo "  job ${jobid}  model=${model}  q=${q} p=${p} h=${h} run=${run}"
          count=$((count + 1))
        done
      done
    done
  done
done

echo "========================================"
echo "Submitted ${count} jobs."
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      logs/rbse_*.out"
echo "========================================"

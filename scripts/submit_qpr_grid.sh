#!/usr/bin/env bash
# submit_qpr_grid.sh -- Submit a grid of CKF/NN hyperparameter runs to Slurm.
#
# Usage (login node):
#   bash submit_qpr_grid.sh                  # all systems, default grid
#   bash submit_qpr_grid.sh retina           # single system
#   bash submit_qpr_grid.sh retina2          # Retina: replace P2 and P4
#   bash submit_qpr_grid.sh ho,retina        # multiple systems
#   Q_SCALES=0.1,1,10 P_SCALES=0.1,1,10 bash submit_qpr_grid.sh retina
#
# Grid defaults (can override via env):
#   Q_SCALES=0.1,1,10
#   R_SCALES=1            (kept fixed by request)
#   P_SCALES=0.1,1,10
#   LR_LIST=5e-3,1e-2,2e-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/sbatch_qpr_job.sh"

MODEL_ARG="${1:-all}"

read_csv() {
  IFS=',' read -ra arr <<<"$1"
  echo "${arr[@]}"
}

LR_LIST=($(read_csv "${LR_LIST:-1e-3,3e-3,5e-3}"))
Q_SCALES=($(read_csv "${Q_SCALES:-0.1,1,10}"))
R_SCALES=($(read_csv "${R_SCALES:-1}"))
P_SCALES=($(read_csv "${P_SCALES:-0.1,1,10}"))

if [[ "${MODEL_ARG}" == "all" ]]; then
  MODELS=(ho pendulum hh brusselator yeast retina retina2)
else
  IFS=',' read -ra MODELS <<<"${MODEL_ARG}"
fi

mkdir -p "${SCRIPT_DIR}/../logs"

echo "Systems: ${MODELS[*]}"
echo "Grid: |Q|=${#Q_SCALES[@]}  |R|=${#R_SCALES[@]}  |P|=${#P_SCALES[@]}  |LR|=${#LR_LIST[@]}"

for model in "${MODELS[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for q in "${Q_SCALES[@]}"; do
      for r in "${R_SCALES[@]}"; do
        for p in "${P_SCALES[@]}"; do
          jobid=$(sbatch --parsable "${JOB_SCRIPT}" "${model}" "${q}" "${r}" "${p}" "${lr}")
          echo "submitted job ${jobid}  model=${model}  lr=${lr} Q=${q} R=${r} P=${p}"
        done
      done
    done
  done
done

echo "Done. Monitor with: squeue -u $USER"

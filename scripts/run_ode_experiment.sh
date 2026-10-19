#!/usr/bin/env bash
# Short runner for ode_experiments.py.
# Usage:
#   bash scripts/run_ode_experiment.sh cpu ho
#   bash scripts/run_ode_experiment.sh gpu retina

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

device="${1:-cpu}"
model="${2:-ho}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

case "${device}" in
  cpu) device_arg=(--device cpu) ;;
  gpu|cuda) device="gpu"; device_arg=(--device cuda) ;;
  auto) device_arg=(--device auto) ;;
  *)
    echo "[run_ode_experiment] device must be cpu, gpu, or auto" >&2
    exit 2
    ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  partition="${ODE_PARTITION:-short}"
  [[ "${device}" == "gpu" ]] && partition="${ODE_PARTITION:-gpu}"
  srun_args=(
    --ntasks=1
    --cpus-per-task="${ODE_CPUS:-4}"
    --mem="${ODE_MEM:-32G}"
    --time="${ODE_TIME:-04:00:00}"
    --partition="${partition}"
    --unbuffered
  )
  if [[ "${device}" == "gpu" ]]; then
    srun_args+=(--gres="${ODE_GRES:-gpu:1}")
  fi
  exec srun "${srun_args[@]}" bash "$0" "${device}" "${model}" "$@"
fi

source "${SCRIPT_DIR}/ode_env.sh" "${device}"
cd "${REPO_ROOT}"

python ode_experiments.py --model "${model}" "${device_arg[@]}" "$@"

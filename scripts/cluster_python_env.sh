#!/usr/bin/env bash

cluster_glibc_lt() {
  local target="${1:?cluster_glibc_lt requires a version}"
  local raw_version
  raw_version="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
  raw_version="${raw_version#glibc }"

  if [[ ! "${raw_version}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    return 1
  fi

  local current_major="${BASH_REMATCH[1]}"
  local current_minor="${BASH_REMATCH[2]}"
  local target_major="${target%%.*}"
  local target_minor="${target#*.}"

  if (( current_major < target_major )); then
    return 0
  fi
  if (( current_major > target_major )); then
    return 1
  fi

  (( current_minor < target_minor ))
}

cluster_pick_default_conda_sh() {
  local candidates=()

  if cluster_glibc_lt 2.28; then
    candidates+=(
      /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
      /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
    )
  else
    candidates+=(
      /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
      /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
    )
  fi

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

cluster_default_modules_for_conda() {
  local conda_sh="${1:-}"

  case "${conda_sh}" in
    /shared/centos7/*)
      printf '%s\n' "anaconda3/2024.06"
      ;;
    /shared/EL9/*)
      printf '%s\n' "anaconda3/2024.06"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

cluster_setup_python_env() {
  local prefix="${1:?cluster_setup_python_env requires a prefix}"
  local default_env="${2:-base}"
  local default_threads="${3:-${SLURM_CPUS_PER_TASK:-1}}"
  local default_modules="${4:-}"
  local default_conda_sh="${5:-}"
  local default_home="${6:-/scratch/demirkaya.a}"

  local setup_var="${prefix}_SETUP_ENV"
  local env_var="${prefix}_CONDA_ENV"
  local threads_var="${prefix}_CPU_THREADS"
  local modules_var="${prefix}_MODULES"
  local conda_sh_var="${prefix}_CONDA_SH"
  local home_var="${prefix}_HOME"

  if [[ "${!setup_var:-1}" != "1" ]]; then
    return 0
  fi

  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
  # CUDA stub libraries are link-time placeholders. If inherited via
  # LD_PRELOAD they create noisy loader errors, and can mask the real driver.
  if [[ "${LD_PRELOAD:-}" == *"/stubs/libcuda.so"* ]]; then
    echo "[env] Removing CUDA stub from LD_PRELOAD: ${LD_PRELOAD}"
    unset LD_PRELOAD
  fi

  local var_name
  for var_name in $(compgen -A variable); do
    if [[ "${var_name}" == CONDA_* ]]; then
      unset "${var_name}"
    fi
  done

  source /etc/profile.d/modules.sh 2>/dev/null || true

  local conda_sh=""
  if [[ -v ${conda_sh_var} ]]; then
    conda_sh="${!conda_sh_var}"
  else
    conda_sh="$(cluster_pick_default_conda_sh 2>/dev/null || true)"
    if [[ -z "${conda_sh}" ]]; then
      conda_sh="${default_conda_sh}"
    fi
  fi

  local module_spec=""
  if [[ -v ${modules_var} ]]; then
    module_spec="${!modules_var}"
  else
    module_spec="$(cluster_default_modules_for_conda "${conda_sh}")"
    if [[ -z "${module_spec}" ]]; then
      module_spec="${default_modules}"
    fi
  fi

  if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true

    if [[ -n "${module_spec}" ]]; then
      # shellcheck disable=SC2086
      module load ${module_spec} 2>/dev/null || true
    fi
  fi

  if [[ -f "${conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] Could not initialize conda for ${prefix}. Set ${conda_sh_var} or ${modules_var}."
    return 1
  fi

  if ! conda activate "${!env_var:-${default_env}}"; then
    echo "[error] Failed to activate conda environment '${!env_var:-${default_env}}' for ${prefix}."
    return 1
  fi

  local home_dir="${!home_var:-${default_home}}"
  if [[ -d "${home_dir}" ]]; then
    export HOME="${home_dir}"
  fi

  local cache_root="${home_dir}/cache"
  if [[ "${home_dir}" == /scratch/* ]]; then
    cache_root="${home_dir}/cache"
  elif [[ -n "${SCRATCH:-}" ]]; then
    cache_root="${SCRATCH}/cache"
  elif [[ -d "/scratch/demirkaya.a" ]]; then
    cache_root="/scratch/demirkaya.a/cache"
  fi
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
  export CONDA_NOTICES_CACHE_DIR="${CONDA_NOTICES_CACHE_DIR:-${XDG_CACHE_HOME}/conda/notices}"
  mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${CONDA_NOTICES_CACHE_DIR}" 2>/dev/null || true

  export ODE_CPU_THREADS="${!threads_var:-${default_threads}}"
  export PYTHONUNBUFFERED=1
  hash -r 2>/dev/null || true
}

#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: condor_wrapper.sh <METHOD> <MODEL_SIZE>" >&2
  exit 1
fi

METHOD="$1"
MODEL_SIZE="$2"
SCRATCH_DIR="$(pwd)"
METHOD_DIR="${SCRATCH_DIR}/${METHOD}"
ENV_TARBALL="${ENV_TARBALL:-llm_env.tar.gz}"
ENV_DIR="${SCRATCH_DIR}/llm_env"

ensure_results_tar() {
  set +e
  local tar_name="results_${MODEL_SIZE}.tar.gz"
  local tar_path="${SCRATCH_DIR}/${tar_name}"
  local results_dir="${METHOD_DIR}/results_${MODEL_SIZE}"

  if [[ -f "$tar_path" ]]; then
    set -e
    return
  fi

  if [[ ! -d "$results_dir" ]]; then
    mkdir -p "$results_dir"
    printf '%s\n' "Pipeline did not produce results; created by condor_wrapper" > "${results_dir}/NO_RESULTS.txt"
  fi

  tar -czf "$tar_path" -C "$METHOD_DIR" "results_${MODEL_SIZE}" || true
  set -e
}

trap 'ensure_results_tar' EXIT

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working Directory: $(pwd)"

if [[ ! -f "${METHOD}.tar.gz" ]]; then
  echo "ERROR: ${METHOD}.tar.gz not found in $(pwd)" >&2
  exit 1
fi

if [[ -f "${ENV_TARBALL}" ]]; then
  echo "Extracting packed conda env: ${ENV_TARBALL}"
  mkdir -p "${ENV_DIR}"
  tar -xzf "${ENV_TARBALL}" -C "${ENV_DIR}"
  if [[ -x "${ENV_DIR}/bin/conda-unpack" ]]; then
    "${ENV_DIR}/bin/conda-unpack"
  fi
  export CONDA_ENV_DIR="${ENV_DIR}"
  export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
else
  echo "WARNING: ${ENV_TARBALL} not found; relying on system/host conda path"
fi

tar -xzf "${METHOD}.tar.gz"
cd "${METHOD_DIR}"

bash scripts/run_scaling_condor.sh "${MODEL_SIZE}"

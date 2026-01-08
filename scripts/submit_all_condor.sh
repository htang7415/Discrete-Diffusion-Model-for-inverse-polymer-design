#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/submit_all_condor.sh <method|all> [sizes...]

Examples:
  bash scripts/submit_all_condor.sh Diffusion_SMILES small medium large xl
  bash scripts/submit_all_condor.sh all small medium

Notes:
  - Uses each subproject's scripts/scaling_gpu.sub
  - Creates /home/htang228/<METHOD>.tar.gz and submits jobs without waiting
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

METHOD_ARG="$1"
shift

if [[ "$METHOD_ARG" == "all" ]]; then
  METHODS=(Diffusion_SMILES Diffusion_SELFIES Diffusion_Group_SELFIES Diffusion_graph)
else
  METHODS=("$METHOD_ARG")
fi

if [[ $# -gt 0 ]]; then
  SIZES=("$@")
else
  SIZES=(small medium large xl)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"

mkdir -p /home/htang228/logs
WRAPPER_SRC="${BASE_PATH}/scripts/condor_wrapper.sh"
WRAPPER_DST="/home/htang228/condor_wrapper.sh"
CONDA_ROOT="${CONDA_ROOT:-}"
ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
if [[ -z "$ENV_PREFIX" ]]; then
  if [[ -d /home/htang228/miniconda3/envs/llm ]]; then
    CONDA_ROOT="${CONDA_ROOT:-/home/htang228/miniconda3}"
    ENV_PREFIX="/home/htang228/miniconda3/envs/llm"
  elif [[ -d /home/htang228/anaconda3/envs/llm ]]; then
    CONDA_ROOT="${CONDA_ROOT:-/home/htang228/anaconda3}"
    ENV_PREFIX="/home/htang228/anaconda3/envs/llm"
  fi
fi
if [[ -z "$CONDA_ROOT" ]]; then
  if [[ -d /home/htang228/miniconda3 ]]; then
    CONDA_ROOT="/home/htang228/miniconda3"
  elif [[ -d /home/htang228/anaconda3 ]]; then
    CONDA_ROOT="/home/htang228/anaconda3"
  else
    echo "ERROR: conda root not found at /home/htang228/miniconda3 or /home/htang228/anaconda3" >&2
    exit 1
  fi
fi
if [[ -z "$ENV_PREFIX" ]]; then
  ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_ROOT}/envs/llm}"
fi
ENV_TARBALL="${ENV_TARBALL:-/home/htang228/llm_env.tar.gz}"

if [[ ! -f "$WRAPPER_SRC" ]]; then
  echo "ERROR: missing ${WRAPPER_SRC}" >&2
  exit 1
fi

cp "$WRAPPER_SRC" "$WRAPPER_DST"
chmod +x "$WRAPPER_DST"

if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "ERROR: conda env not found at ${ENV_PREFIX}" >&2
  exit 1
fi

if [[ ! -f "$ENV_TARBALL" ]]; then
  if [[ -x "${CONDA_ROOT}/bin/conda-pack" ]]; then
    PACK_CMD="${CONDA_ROOT}/bin/conda-pack"
  elif [[ -x "${ENV_PREFIX}/bin/conda-pack" ]]; then
    PACK_CMD="${ENV_PREFIX}/bin/conda-pack"
  elif command -v conda-pack >/dev/null 2>&1; then
    PACK_CMD="conda-pack"
  else
    echo "ERROR: conda-pack not found; install it to create ${ENV_TARBALL}" >&2
    exit 1
  fi
  echo "Creating packed conda env: ${ENV_TARBALL}"
  "${PACK_CMD}" -p "$ENV_PREFIX" -o "$ENV_TARBALL"
fi

for method in "${METHODS[@]}"; do
  method_dir="${BASE_PATH}/${method}"
  sub_file="${method_dir}/scripts/scaling_gpu.sub"

  if [[ ! -d "$method_dir" ]]; then
    echo "Skipping ${method}: directory not found at ${method_dir}"
    continue
  fi

  if [[ ! -f "$sub_file" ]]; then
    echo "Skipping ${method}: missing ${sub_file}"
    continue
  fi

  echo "=========================================="
  echo "${method}: creating tarball"
  echo "=========================================="
  tar --exclude='results_*' -czf "/home/htang228/${method}.tar.gz" -C "$BASE_PATH" "${method}/"

  echo "Submitting ${method} sizes: ${SIZES[*]}"
  for size in "${SIZES[@]}"; do
    (cd /home/htang228 && condor_submit -append "MODEL_SIZE = ${size}" \
      -append "BASE_PATH = ${BASE_PATH}" \
      "$sub_file")
  done
  echo ""
done

echo "All submissions queued. Check with: condor_q"
exit 0

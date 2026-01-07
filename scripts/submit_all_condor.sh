#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/submit_all_condor.sh <method|all> [sizes...]

Examples:
  scripts/submit_all_condor.sh Diffusion_SMILES small medium large xl
  scripts/submit_all_condor.sh all small medium

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
    condor_submit -append "MODEL_SIZE = ${size}" \
      -append "BASE_PATH = ${BASE_PATH}" \
      "$sub_file"
  done
  echo ""
done

echo "All submissions queued. Check with: condor_q"
exit 0

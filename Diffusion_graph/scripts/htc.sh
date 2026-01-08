#!/bin/bash
# HTCondor Helper (Graph)
# Usage: ./scripts/htc.sh [model_size]
# Submits job, waits for completion, extracts results

set -euo pipefail

MODEL_SIZE="${1:-medium}"

# Auto-detect paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="$(basename "$(dirname "$SCRIPT_DIR")")"
BASE_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
PREFIX="gra"

echo "=========================================="
echo "${METHOD} (${MODEL_SIZE})"
echo "=========================================="

# 1. Create tarball
echo "Creating tarball..."
mkdir -p /home/htang228/logs
cd "$BASE_PATH"
tar --exclude='results_*' -czf "/home/htang228/${METHOD}.tar.gz" "${METHOD}/"
cp "${BASE_PATH}/scripts/condor_wrapper.sh" /home/htang228/condor_wrapper.sh
chmod +x /home/htang228/condor_wrapper.sh
CONDA_ROOT="${CONDA_ROOT:-}"
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

# 2. Submit job
echo "Submitting job..."
cd /home/htang228
condor_submit -append "MODEL_SIZE = ${MODEL_SIZE}" \
    -append "BASE_PATH = ${BASE_PATH}" \
    "${BASE_PATH}/${METHOD}/scripts/scaling_gpu.sub"

# 3. Wait for job to complete
echo "Waiting for job to complete..."
sleep 5
LOG=$(ls -t /home/htang228/logs/${PREFIX}_${MODEL_SIZE}_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo "Monitoring: $LOG"
    condor_wait "$LOG"
else
    echo "WARNING: Log file not found, check condor_q manually"
fi

# 4. Extract results
echo "Extracting results..."
cd "${BASE_PATH}/${METHOD}"
latest=$(ls -t results_${MODEL_SIZE}_*.tar.gz 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    tar -xzf "$latest"
    echo "=========================================="
    echo "Done! Results in: results_${MODEL_SIZE}/"
    echo "=========================================="
else
    echo "WARNING: No results tarball found"
fi

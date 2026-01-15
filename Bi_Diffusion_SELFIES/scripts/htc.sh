#!/bin/bash
# HTCondor Helper (SELFIES)
# Usage: ./scripts/htc.sh [model_size]
# Submits job, waits for completion, extracts results

set -euo pipefail

MODEL_SIZE="${1:-medium}"

# Auto-detect paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="$(basename "$(dirname "$SCRIPT_DIR")")"
BASE_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
PREFIX="sel"

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

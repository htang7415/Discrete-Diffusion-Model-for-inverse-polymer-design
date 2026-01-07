#!/bin/bash
set -euo pipefail

MODEL_SIZE="${1:-medium}"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working Directory: $(pwd)"

# === EASY CONFIG ===
METHOD="Diffusion_Group_SELFIES"

# Conda setup (HTC path - must be accessible on execute nodes)
source /home/htang228/miniconda3/etc/profile.d/conda.sh
conda activate llm

echo "Python: $(which python)"
python -V

# Fixed parameters
PROPERTY="Tg"
TARGET="300"
POLYMER_CLASS="polyimide"
NUM_SAMPLES=20000
NUM_CANDIDATES=20000

mkdir -p logs

echo "=========================================="
echo "Scaling Law Experiment (Group SELFIES)"
echo "=========================================="
echo "Model Size: ${MODEL_SIZE}"
echo "Property: ${PROPERTY}"
echo "Target: ${TARGET}"
echo "Polymer Class: ${POLYMER_CLASS}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Num Candidates: ${NUM_CANDIDATES}"
echo "Work Directory: $(pwd)"
echo "=========================================="

python scripts/run_scaling_pipeline.py \
    --model_size "${MODEL_SIZE}" \
    --property "${PROPERTY}" \
    --target "${TARGET}" \
    --polymer_class "${POLYMER_CLASS}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_candidates "${NUM_CANDIDATES}"

# Tar results for HTCondor transfer (guard against missing directory)
if [ -d "results_${MODEL_SIZE}" ]; then
    echo "Packaging results for transfer..."
    tar -czf "../results_${MODEL_SIZE}.tar.gz" "results_${MODEL_SIZE}"
    echo "Results packaged: results_${MODEL_SIZE}.tar.gz"
else
    echo "WARNING: results_${MODEL_SIZE} directory not found!"
    echo "Creating empty tarball to prevent transfer error..."
    mkdir -p "results_${MODEL_SIZE}"
    echo "Pipeline did not produce results" > "results_${MODEL_SIZE}/NO_RESULTS.txt"
    tar -czf "../results_${MODEL_SIZE}.tar.gz" "results_${MODEL_SIZE}"
fi

echo "=========================================="
echo "End Time: $(date)"
echo "Experiment Complete!"
echo "=========================================="

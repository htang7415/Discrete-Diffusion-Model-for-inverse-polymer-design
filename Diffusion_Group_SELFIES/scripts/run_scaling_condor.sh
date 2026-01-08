#!/bin/bash
set -euo pipefail

MODEL_SIZE="${1:-medium}"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working Directory: $(pwd)"

# === EASY CONFIG ===
METHOD="Diffusion_Group_SELFIES"

# Conda setup (HTC path - must be accessible on execute nodes)
DEFAULT_CONDA_ROOT="/home/htang228/miniconda3"
if [ ! -d "${DEFAULT_CONDA_ROOT}" ] && [ -d /home/htang228/anaconda3 ]; then
  DEFAULT_CONDA_ROOT="/home/htang228/anaconda3"
fi
CONDA_ENV="${CONDA_ENV_DIR:-${DEFAULT_CONDA_ROOT}/envs/llm}"
CONDA_SH="${DEFAULT_CONDA_ROOT}/etc/profile.d/conda.sh"
if [ -n "${CONDA_ENV_DIR:-}" ]; then
  if [ -x "${CONDA_ENV}/bin/python" ]; then
    export PATH="${CONDA_ENV}/bin:${PATH}"
  else
    echo "ERROR: packed conda env missing python at ${CONDA_ENV}/bin/python" >&2
    exit 1
  fi
elif [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [ -x "${CONDA_ENV}/bin/python" ]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  echo "ERROR: conda env not found at ${CONDA_ENV}" >&2
  exit 1
fi

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

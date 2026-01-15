#!/bin/bash
set -euo pipefail

MODEL_SIZES_INPUT="${1:-medium}"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working Directory: $(pwd)"

# === EASY CONFIG ===
METHOD="Diffusion_graph"

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
PROPERTY_LIST="${2:-Tg,Tm,Eg,Td}"
POLYMER_CLASS_LIST="${3:-polyimide,polyamide}"
NUM_SAMPLES="${4:-10000}"
NUM_CANDIDATES="${5:-20000}"

MODEL_SIZES_INPUT="${MODEL_SIZES_INPUT// /}"
PROPERTY_LIST="${PROPERTY_LIST// /}"
POLYMER_CLASS_LIST="${POLYMER_CLASS_LIST// /}"
if [ "${MODEL_SIZES_INPUT}" = "all" ]; then
  MODEL_SIZES_INPUT="small,medium,large,xl"
fi
if [ "${PROPERTY_LIST}" = "all" ]; then
  PROPERTY_LIST="Tg,Tm,Eg,Td"
fi
if [ "${POLYMER_CLASS_LIST}" = "all" ]; then
  POLYMER_CLASS_LIST="polyimide,polyamide"
fi

IFS=',' read -r -a MODEL_SIZES <<< "${MODEL_SIZES_INPUT}"
IFS=',' read -r -a PROPERTIES <<< "${PROPERTY_LIST}"
IFS=',' read -r -a CLASSES <<< "${POLYMER_CLASS_LIST}"

mkdir -p logs

echo "=========================================="
echo "Scaling Law Experiment (Graph)"
echo "=========================================="
echo "Model Sizes: ${MODEL_SIZES_INPUT}"
echo "Properties: ${PROPERTY_LIST}"
echo "Polymer Classes: ${POLYMER_CLASS_LIST}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Num Candidates: ${NUM_CANDIDATES}"
echo "Work Directory: $(pwd)"
echo "=========================================="

for model_size in "${MODEL_SIZES[@]}"; do
  python scripts/run_scaling_pipeline.py \
      --model_size "${model_size}" \
      --num_samples "${NUM_SAMPLES}" \
      --num_candidates "${NUM_CANDIDATES}" \
      --skip_step3 --skip_step4 --skip_step5 --skip_step6

  for prop in "${PROPERTIES[@]}"; do
    python scripts/run_scaling_pipeline.py \
        --model_size "${model_size}" \
        --property "${prop}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_candidates "${NUM_CANDIDATES}" \
        --skip_step1 --skip_step2 --skip_step5 --skip_step6
  done

  for cls in "${CLASSES[@]}"; do
    python scripts/run_scaling_pipeline.py \
        --model_size "${model_size}" \
        --polymer_class "${cls}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_candidates "${NUM_CANDIDATES}" \
        --skip_step1 --skip_step2 --skip_step3 --skip_step4 --skip_step6
  done

  for prop in "${PROPERTIES[@]}"; do
    for cls in "${CLASSES[@]}"; do
      python scripts/run_scaling_pipeline.py \
          --model_size "${model_size}" \
          --property "${prop}" \
          --polymer_class "${cls}" \
          --num_samples "${NUM_SAMPLES}" \
          --num_candidates "${NUM_CANDIDATES}" \
          --skip_step1 --skip_step2 --skip_step3 --skip_step4 --skip_step5
    done
  done

  # Tar results for HTCondor transfer (guard against missing directory)
  if [ -d "results_${model_size}" ]; then
      echo "Packaging results for transfer..."
      tar -czf "../results_${model_size}.tar.gz" "results_${model_size}"
      echo "Results packaged: results_${model_size}.tar.gz"
  else
      echo "WARNING: results_${model_size} directory not found!"
      echo "Creating empty tarball to prevent transfer error..."
      mkdir -p "results_${model_size}"
      echo "Pipeline did not produce results" > "results_${model_size}/NO_RESULTS.txt"
      tar -czf "../results_${model_size}.tar.gz" "results_${model_size}"
  fi
done

echo "=========================================="
echo "End Time: $(date)"
echo "Experiment Complete!"
echo "=========================================="

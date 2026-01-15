#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=24:00:00
#SBATCH --job-name=smi_s
#SBATCH --output=logs/smi_s_%x_%j.out
#SBATCH --error=logs/smi_s_%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# Scaling Law Experiment Runner (SMILES)
# Usage: sbatch scripts/submit_scaling_nrel.sh <model_sizes> [properties] [polymer_classes] [num_samples] [num_candidates]
# Example: sbatch scripts/submit_scaling_nrel.sh small
#          sbatch scripts/submit_scaling_nrel.sh all
#          sbatch scripts/submit_scaling_nrel.sh small,medium Tg,Tm,Eg,Td polyimide,polyamide

set -e

# Conda setup
CONDA_DIR="/home/htang/anaconda3"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate kl_active_learning

echo "Python: $(which python)"
python -V
echo "Pip: $(which pip)"
pip -V

# Work directory
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

# Parse arguments
MODEL_SIZES_INPUT=${1:-medium}
PROPERTY_LIST=${2:-Tg,Tm,Eg,Td}
POLYMER_CLASS_LIST=${3:-polyimide,polyamide}
NUM_SAMPLES=${4:-10000}
NUM_CANDIDATES=${5:-20000}

MODEL_SIZES_INPUT=${MODEL_SIZES_INPUT// /}
PROPERTY_LIST=${PROPERTY_LIST// /}
POLYMER_CLASS_LIST=${POLYMER_CLASS_LIST// /}
if [ "$MODEL_SIZES_INPUT" = "all" ]; then
  MODEL_SIZES_INPUT="small,medium,large,xl"
fi
if [ "$PROPERTY_LIST" = "all" ]; then
  PROPERTY_LIST="Tg,Tm,Eg,Td"
fi
if [ "$POLYMER_CLASS_LIST" = "all" ]; then
  POLYMER_CLASS_LIST="polyimide,polyamide"
fi

IFS=',' read -r -a MODEL_SIZES <<< "$MODEL_SIZES_INPUT"
IFS=',' read -r -a PROPERTIES <<< "$PROPERTY_LIST"
IFS=',' read -r -a CLASSES <<< "$POLYMER_CLASS_LIST"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Scaling Law Experiment (SMILES)"
echo "=========================================="
echo "Model Sizes: ${MODEL_SIZES_INPUT}"
echo "Properties: ${PROPERTY_LIST}"
echo "Polymer Classes: ${POLYMER_CLASS_LIST}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Num Candidates: ${NUM_CANDIDATES}"
echo "Work Directory: ${WORKDIR}"
echo "=========================================="
echo "Start Time: $(date)"
echo "=========================================="

# Run the scaling pipeline
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
done

echo "=========================================="
echo "End Time: $(date)"
echo "Experiment Complete!"
echo "=========================================="

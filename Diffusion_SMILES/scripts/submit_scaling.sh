#!/bin/bash
#SBATCH --job-name=smiles_scaling
#SBATCH --output=logs/scaling_%x_%j.out
#SBATCH --error=logs/scaling_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pdelab
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00

# Scaling Law Experiment Runner (SMILES)
# Usage: sbatch scripts/submit_scaling.sh <model_size>
# Example: sbatch scripts/submit_scaling.sh small
#          sbatch scripts/submit_scaling.sh medium
#          sbatch scripts/submit_scaling.sh large
#          sbatch scripts/submit_scaling.sh xl

set -e

# Conda setup
CONDA_DIR="/srv/home/htang228/anaconda3"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate euler_active_learning

echo "Python: $(which python)"
python -V
echo "Pip: $(which pip)"
pip -V

# Work directory
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

# Parse arguments
MODEL_SIZE=${1:-medium}

# Fixed parameters
PROPERTY="Tg"
TARGET="300"
POLYMER_CLASS="polyimide"
NUM_SAMPLES=50000
NUM_CANDIDATES=10000

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Scaling Law Experiment (SMILES)"
echo "=========================================="
echo "Model Size: ${MODEL_SIZE}"
echo "Property: ${PROPERTY}"
echo "Target: ${TARGET}"
echo "Polymer Class: ${POLYMER_CLASS}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Num Candidates: ${NUM_CANDIDATES}"
echo "Work Directory: ${WORKDIR}"
echo "=========================================="
echo "Start Time: $(date)"
echo "=========================================="

# Run the scaling pipeline
python scripts/run_scaling_pipeline.py \
    --model_size ${MODEL_SIZE} \
    --property ${PROPERTY} \
    --target ${TARGET} \
    --polymer_class ${POLYMER_CLASS} \
    --num_samples ${NUM_SAMPLES} \
    --num_candidates ${NUM_CANDIDATES}

echo "=========================================="
echo "End Time: $(date)"
echo "Experiment Complete!"
echo "=========================================="

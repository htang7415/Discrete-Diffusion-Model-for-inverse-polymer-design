#!/bin/bash
#SBATCH --job-name=gra_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=pdelab
#SBATCH --gres=gpu:4
#SBATCH --time=10-00:00:00

# Step 1 DDP (Euler, 1 node / 4 GPU)
# Usage: sbatch scripts/submit_step1_ddp_euler.sh <model_size>

set -e

# Conda setup
CONDA_DIR="/srv/home/htang228/anaconda3"
eval "$("$CONDA_DIR"/bin/conda shell.bash hook)"
conda activate euler_active_learning

# Work directory
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs

MODEL_SIZE=${1:-medium}

# --standalone auto-selects free port, no MASTER_ADDR/MASTER_PORT needed
torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/step1_train_backbone.py --config configs/config.yaml --model_size "$MODEL_SIZE"

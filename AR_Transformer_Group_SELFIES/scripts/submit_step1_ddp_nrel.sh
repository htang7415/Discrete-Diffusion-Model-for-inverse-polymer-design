#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ar_grp_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4

# Step 1 DDP (NREL, 1 node / 4 GPU total)
# Usage: sbatch scripts/submit_step1_ddp_nrel.sh <model_size>
# Tip: pass --partition/--qos at submit time for your GPU type.

set -e

# Conda setup
CONDA_DIR="/home/htang/anaconda3"
eval "$("$CONDA_DIR"/bin/conda shell.bash hook)"
conda activate kl_active_learning

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

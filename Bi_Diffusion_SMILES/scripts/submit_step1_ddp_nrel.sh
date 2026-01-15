#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=2-00:00:00
#SBATCH --job-name=smi_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=4

srun --cpu-bind=cores torchrun   --nnodes=$SLURM_NNODES   --nproc_per_node=$NPROC_PER_NODE   --rdzv_backend=c10d   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT   scripts/step1_train_backbone.py --config configs/config.yaml --model_size "$MODEL_SIZE"

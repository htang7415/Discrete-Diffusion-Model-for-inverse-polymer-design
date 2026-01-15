#!/bin/bash
# Option B submit wrapper (NREL): submit Step 1 DDP, then Steps 2-6 after success.
# Usage: bash scripts/submit_all_nrel.sh <model_sizes> [properties] [targets] [epsilon]
#        [num_samples] [num_candidates] [polymer_classes] [class_target] [class_epsilon] [partition] [qos]
#        bash scripts/submit_all_nrel.sh small
#        bash scripts/submit_all_nrel.sh all
#        bash scripts/submit_all_nrel.sh small,medium
#        bash scripts/submit_all_nrel.sh medium
#        bash scripts/submit_all_nrel.sh large
#        bash scripts/submit_all_nrel.sh xl

set -e

PREFIX="gsel"

MODEL_SIZES_INPUT=${1:-medium}
PROPERTY_LIST=${2:-Tg,Tm,Eg,Td}
TARGETS=${3:-}
EPSILON=${4:-}
NUM_SAMPLES=${5:-10000}
NUM_CANDIDATES=${6:-10000}
POLYMER_CLASS_LIST=${7:-polyimide,polyamide}
CLASS_TARGET=${8:-}
CLASS_EPSILON=${9:-}
MODEL_SIZES_INPUT=${MODEL_SIZES_INPUT// /}
if [ "$MODEL_SIZES_INPUT" = "all" ]; then
  MODEL_SIZES_INPUT="small,medium,large,xl"
fi
IFS=',' read -r -a MODEL_SIZES <<< "$MODEL_SIZES_INPUT"

mkdir -p logs

PARTITION=${10:-}
QOS=${11:-}

SBATCH_ARGS=()
if [ -n "$PARTITION" ]; then
  SBATCH_ARGS+=("--partition" "$PARTITION")
fi
if [ -n "$QOS" ]; then
  SBATCH_ARGS+=("--qos" "$QOS")
fi

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
  SIZE_TAG="m"
  case "$MODEL_SIZE" in
    small|s) SIZE_TAG="s" ;;
    medium|m) SIZE_TAG="m" ;;
    large|l) SIZE_TAG="l" ;;
    xl|x) SIZE_TAG="xl" ;;
    *) SIZE_TAG="$MODEL_SIZE" ;;
  esac

  JOB1="${PREFIX}_${SIZE_TAG}_1"
  JOB2="${PREFIX}_${SIZE_TAG}_2_6"

  jid=$(sbatch --parsable "${SBATCH_ARGS[@]}" --job-name "$JOB1" --output "logs/%x_%j.out" --error "logs/%x_%j.err" scripts/submit_step1_ddp_nrel.sh "$MODEL_SIZE")
  sbatch "${SBATCH_ARGS[@]}" --job-name "$JOB2" --output "logs/%x_%j.out" --error "logs/%x_%j.err" --dependency=afterok:$jid scripts/submit_steps2_6_nrel.sh   "$MODEL_SIZE" "$PROPERTY_LIST" "$TARGETS" "$EPSILON" "$NUM_SAMPLES"   "$NUM_CANDIDATES" "$POLYMER_CLASS_LIST" "$CLASS_TARGET" "$CLASS_EPSILON"
done

#!/bin/bash
#SBATCH --job-name=sel_2_6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=pdelab
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00

# Steps 2-6 (Euler, single GPU)
# Usage: sbatch scripts/submit_steps2_6_euler.sh <model_size> [properties] [targets] [epsilon]
#        [num_samples] [num_candidates] [polymer_classes] [class_target] [class_epsilon]
# Notes:
# - properties and polymer_classes are comma-separated (e.g., Tg,Tm,Eg,Td or polyimide,polyamide).
# - leave targets/epsilon/class_target/class_epsilon empty to use property-specific defaults.

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
PROPERTY_LIST=${2:-Tg,Tm,Eg,Td}
TARGETS=${3:-}
EPSILON=${4:-}
NUM_SAMPLES=${5:-10000}
NUM_CANDIDATES=${6:-10000}
POLYMER_CLASS_LIST=${7:-polyimide,polyamide}
CLASS_TARGET=${8:-}
CLASS_EPSILON=${9:-}

PROPERTY_LIST=${PROPERTY_LIST// /}
POLYMER_CLASS_LIST=${POLYMER_CLASS_LIST// /}
if [ "$PROPERTY_LIST" = "all" ]; then
  PROPERTY_LIST="Tg,Tm,Eg,Td"
fi
if [ "$POLYMER_CLASS_LIST" = "all" ]; then
  POLYMER_CLASS_LIST="polyimide,polyamide"
fi

default_target_for_property() {
  case "$1" in
    Tg) echo 350 ;;
    Tm) echo 450 ;;
    Td) echo 550 ;;
    Eg) echo 8 ;;
    *) echo 300 ;;
  esac
}

default_epsilon_for_property() {
  case "$1" in
    Eg) echo 0.5 ;;
    Tg|Tm|Td) echo 30 ;;
    *) echo 10.0 ;;
  esac
}

IFS=',' read -r -a PROPERTIES <<< "$PROPERTY_LIST"
IFS=',' read -r -a CLASSES <<< "$POLYMER_CLASS_LIST"

python scripts/step2_sample_and_evaluate.py --config configs/config.yaml --model_size "$MODEL_SIZE" --num_samples "$NUM_SAMPLES"

for prop in "${PROPERTIES[@]}"; do
  prop_targets="$TARGETS"
  if [ -z "$prop_targets" ]; then
    prop_targets="$(default_target_for_property "$prop")"
  fi
  prop_epsilon="$EPSILON"
  if [ -z "$prop_epsilon" ]; then
    prop_epsilon="$(default_epsilon_for_property "$prop")"
  fi
  python scripts/step3_train_property_head.py --config configs/config.yaml --model_size "$MODEL_SIZE" --property "$prop"
  python scripts/step4_inverse_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --property "$prop" --targets "$prop_targets" --epsilon "$prop_epsilon" --num_candidates "$NUM_CANDIDATES"
done

for cls in "${CLASSES[@]}"; do
  python scripts/step5_class_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --polymer_class "$cls" --num_candidates "$NUM_CANDIDATES"
done

for cls in "${CLASSES[@]}"; do
  for prop in "${PROPERTIES[@]}"; do
    class_target="$CLASS_TARGET"
    if [ -z "$class_target" ]; then
      class_target="$(default_target_for_property "$prop")"
    fi
    class_epsilon="$CLASS_EPSILON"
    if [ -z "$class_epsilon" ]; then
      class_epsilon="$(default_epsilon_for_property "$prop")"
    fi
    python scripts/step5_class_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --polymer_class "$cls" --property "$prop" --target_value "$class_target" --epsilon "$class_epsilon" --num_candidates "$NUM_CANDIDATES"
  done
done

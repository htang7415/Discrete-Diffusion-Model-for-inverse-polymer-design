#!/bin/bash
# Step 4: Property-guided inverse design

set -e
PROPERTY=${1:-Tg}
TARGETS=${2:-}
EPSILON=${3:-}
NUM_CANDIDATES=${4:-10000}

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

if [ -z "$TARGETS" ]; then
  TARGETS="$(default_target_for_property "$PROPERTY")"
fi
if [ -z "$EPSILON" ]; then
  EPSILON="$(default_epsilon_for_property "$PROPERTY")"
fi

echo "Step 4: Inverse design for ${PROPERTY}..."
echo "  Targets: ${TARGETS}"
echo "  Epsilon: ${EPSILON}"
echo "  Candidates: ${NUM_CANDIDATES}"

python scripts/step4_inverse_design.py \
    --config configs/config.yaml \
    --property $PROPERTY \
    --targets $TARGETS \
    --epsilon $EPSILON \
    --num_candidates $NUM_CANDIDATES

echo "Done!"

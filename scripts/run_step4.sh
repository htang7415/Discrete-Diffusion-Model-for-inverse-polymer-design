#!/bin/bash
# Step 4: Property-guided inverse design

set -e
PROPERTY=${1:-Tg}
TARGETS=${2:-"200,300,400"}
EPSILON=${3:-10.0}
NUM_CANDIDATES=${4:-10000}

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

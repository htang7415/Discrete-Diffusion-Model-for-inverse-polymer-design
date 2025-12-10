#!/bin/bash
# Step 5: Polymer class-guided design

set -e
POLYMER_CLASS=${1:-polyamide}
PROPERTY=${2:-}
TARGET_VALUE=${3:-}
EPSILON=${4:-10.0}
NUM_CANDIDATES=${5:-10000}

echo "Step 5: Class-guided design for ${POLYMER_CLASS}..."

CMD="python scripts/step5_class_design.py --config configs/config.yaml --polymer_class $POLYMER_CLASS --num_candidates $NUM_CANDIDATES"

if [ -n "$PROPERTY" ] && [ -n "$TARGET_VALUE" ]; then
    echo "  Joint design with ${PROPERTY}=${TARGET_VALUE}"
    CMD="$CMD --property $PROPERTY --target_value $TARGET_VALUE --epsilon $EPSILON"
fi

$CMD
echo "Done!"

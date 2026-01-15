#!/bin/bash
# Step 5: Polymer class-guided design

set -e
POLYMER_CLASS=${1:-polyamide}
PROPERTY=${2:-}
TARGET_VALUE=${3:-}
EPSILON=${4:-}
NUM_CANDIDATES=${5:-10000}

echo "Step 5: Class-guided design for ${POLYMER_CLASS}..."

default_epsilon_for_property() {
  case "$1" in
    Eg) echo 0.5 ;;
    Tg|Tm|Td) echo 30 ;;
    *) echo 10.0 ;;
  esac
}

CMD="python scripts/step5_class_design.py --config configs/config.yaml --polymer_class $POLYMER_CLASS --num_candidates $NUM_CANDIDATES"

if [ -n "$PROPERTY" ] && [ -n "$TARGET_VALUE" ]; then
    echo "  Joint design with ${PROPERTY}=${TARGET_VALUE}"
    if [ -z "$EPSILON" ]; then
        EPSILON="$(default_epsilon_for_property "$PROPERTY")"
    fi
    CMD="$CMD --property $PROPERTY --target_value $TARGET_VALUE --epsilon $EPSILON"
fi

$CMD
echo "Done!"

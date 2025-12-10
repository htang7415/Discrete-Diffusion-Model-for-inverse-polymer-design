#!/bin/bash
# Step 6: Hyperparameter tuning (optional)

set -e
MODE=${1:-backbone}
PROPERTY=${2:-Tg}

if [ "$MODE" == "backbone" ]; then
    echo "Step 6: Tuning backbone hyperparameters..."
    python scripts/step6_hyperparameter_tuning.py --config configs/config.yaml --mode backbone
elif [ "$MODE" == "property" ]; then
    echo "Step 6: Tuning property head for ${PROPERTY}..."
    python scripts/step6_hyperparameter_tuning.py --config configs/config.yaml --mode property --property $PROPERTY
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_step6.sh [backbone|property] [property_name]"
    exit 1
fi

echo "Done!"

#!/bin/bash
# Step 3: Train property heads

set -e
PROPERTY=${1:-Tg}
echo "Step 3: Training property head for ${PROPERTY}..."
python scripts/step3_train_property_head.py --config configs/config.yaml --property $PROPERTY
echo "Done!"

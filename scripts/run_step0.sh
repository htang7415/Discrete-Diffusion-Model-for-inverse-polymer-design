#!/bin/bash
# Step 0: Prepare data and build vocabulary

set -e
echo "Step 0: Preparing data..."
python scripts/step0_prepare_data.py --config configs/config.yaml
echo "Done!"

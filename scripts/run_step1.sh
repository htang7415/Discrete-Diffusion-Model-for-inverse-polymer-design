#!/bin/bash
# Step 1: Train diffusion backbone

set -e
echo "Step 1: Training diffusion backbone..."
python scripts/step1_train_backbone.py --config configs/config.yaml
echo "Done!"

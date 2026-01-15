#!/bin/bash
# Step 2: Sample from backbone and evaluate

set -e
NUM_SAMPLES=${1:-10000}
echo "Step 2: Sampling ${NUM_SAMPLES} polymers and evaluating..."
python scripts/step2_sample_and_evaluate.py --config configs/config.yaml --num_samples $NUM_SAMPLES
echo "Done!"

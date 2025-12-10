#!/bin/bash
# Full pipeline for inverse polymer design

set -e  # Exit on error

echo "=================================================="
echo "Inverse Polymer Design Pipeline"
echo "=================================================="

# Configuration
CONFIG="configs/config.yaml"
NUM_SAMPLES=50000
PROPERTIES="Tg Tm"  # Properties to train
INVERSE_TARGETS="200,300,400"  # Target values for inverse design
EPSILON=10.0

# Step 0: Prepare data and build vocabulary
echo ""
echo "Step 0: Preparing data..."
python scripts/step0_prepare_data.py --config $CONFIG

# Step 1: Train backbone
echo ""
echo "Step 1: Training diffusion backbone..."
python scripts/step1_train_backbone.py --config $CONFIG

# Step 2: Sample and evaluate
echo ""
echo "Step 2: Sampling and evaluating..."
python scripts/step2_sample_and_evaluate.py --config $CONFIG --num_samples $NUM_SAMPLES

# Step 3: Train property heads
echo ""
echo "Step 3: Training property heads..."
for prop in $PROPERTIES; do
    echo "Training property head for $prop..."
    python scripts/step3_train_property_head.py --config $CONFIG --property $prop
done

# Step 4: Inverse design
echo ""
echo "Step 4: Running inverse design..."
for prop in $PROPERTIES; do
    echo "Inverse design for $prop..."
    python scripts/step4_inverse_design.py --config $CONFIG --property $prop --targets $INVERSE_TARGETS --epsilon $EPSILON
done

# Step 5: Class-guided design (optional)
echo ""
echo "Step 5: Running class-guided design..."
python scripts/step5_class_design.py --config $CONFIG --polymer_class polyamide --num_candidates 10000

# Done
echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "Results saved to: results/"
echo "=================================================="

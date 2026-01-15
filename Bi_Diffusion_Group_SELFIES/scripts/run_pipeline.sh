#!/bin/bash
# Full pipeline for inverse polymer design

set -e  # Exit on error

echo "=================================================="
echo "Inverse Polymer Design Pipeline"
echo "=================================================="

# Configuration
CONFIG="configs/config.yaml"
NUM_SAMPLES=10000
PROPERTIES="Tg Tm Eg Td"  # Properties to train
POLYMER_CLASSES="polyimide polyamide"

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
    TARGET_VALUE="$(default_target_for_property "$prop")"
    EPSILON_VALUE="$(default_epsilon_for_property "$prop")"
    python scripts/step4_inverse_design.py --config $CONFIG --property $prop --targets $TARGET_VALUE --epsilon $EPSILON_VALUE
done

# Step 5: Class-guided design (optional)
echo ""
echo "Step 5: Running class-guided design..."
for cls in $POLYMER_CLASSES; do
    python scripts/step5_class_design.py --config $CONFIG --polymer_class $cls --num_candidates 10000
done

# Done
echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "Results saved to: results/"
echo "=================================================="

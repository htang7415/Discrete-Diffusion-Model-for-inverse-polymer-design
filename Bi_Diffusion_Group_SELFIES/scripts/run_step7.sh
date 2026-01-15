#!/bin/bash
# Step 7: Hyperparameter tuning (optional)

set -e
MODE=${1:-backbone}
PROPERTY=${2:-Tg}

./scripts/run_step6.sh "$MODE" "$PROPERTY"

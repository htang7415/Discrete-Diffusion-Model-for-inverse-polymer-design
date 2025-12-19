#!/bin/bash
# Run Steps 0-6 end-to-end.

set -e

PROPERTY=${1:-Tg}
TARGETS=${2:-"200,300,400"}
EPSILON=${3:-10.0}
NUM_SAMPLES=${4:-50000}
NUM_CANDIDATES=${5:-10000}
POLYMER_CLASS=${6:-polyamide}
CLASS_TARGET_VALUE=${7:-300}
CLASS_EPSILON=${8:-10.0}

echo "Running Steps 0-6..."
echo "  Property: ${PROPERTY}"
echo "  Targets: ${TARGETS}"
echo "  Epsilon: ${EPSILON}"
echo "  Num samples: ${NUM_SAMPLES}"
echo "  Num candidates: ${NUM_CANDIDATES}"
echo "  Polymer class: ${POLYMER_CLASS}"
echo "  Class target: ${CLASS_TARGET_VALUE}"
echo "  Class epsilon: ${CLASS_EPSILON}"

./scripts/run_step0.sh
./scripts/run_step1.sh
./scripts/run_step2.sh "${NUM_SAMPLES}"
./scripts/run_step3.sh "${PROPERTY}"
./scripts/run_step4.sh "${PROPERTY}" "${TARGETS}" "${EPSILON}" "${NUM_CANDIDATES}"
./scripts/run_step5.sh "${POLYMER_CLASS}" "" "" "" "${NUM_CANDIDATES}"
./scripts/run_step5.sh "${POLYMER_CLASS}" "${PROPERTY}" "${CLASS_TARGET_VALUE}" "${CLASS_EPSILON}" "${NUM_CANDIDATES}"

echo "Done!"

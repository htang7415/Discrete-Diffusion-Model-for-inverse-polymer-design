#!/bin/bash
# Run Steps 0-6 end-to-end.

set -e

PROPERTY_LIST=${1:-Tg,Tm,Eg,Td}
TARGETS=${2:-}
EPSILON=${3:-}
NUM_SAMPLES=${4:-10000}
NUM_CANDIDATES=${5:-10000}
POLYMER_CLASS_LIST=${6:-polyimide,polyamide}
CLASS_TARGET_VALUE=${7:-}
CLASS_EPSILON=${8:-}

PROPERTY_LIST=${PROPERTY_LIST// /}
POLYMER_CLASS_LIST=${POLYMER_CLASS_LIST// /}
if [ "$PROPERTY_LIST" = "all" ]; then
  PROPERTY_LIST="Tg,Tm,Eg,Td"
fi
if [ "$POLYMER_CLASS_LIST" = "all" ]; then
  POLYMER_CLASS_LIST="polyimide,polyamide"
fi

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

IFS=',' read -r -a PROPERTIES <<< "$PROPERTY_LIST"
IFS=',' read -r -a CLASSES <<< "$POLYMER_CLASS_LIST"

echo "Running Steps 0-6..."
echo "  Properties: ${PROPERTY_LIST}"
echo "  Targets: ${TARGETS}"
echo "  Epsilon: ${EPSILON}"
echo "  Num samples: ${NUM_SAMPLES}"
echo "  Num candidates: ${NUM_CANDIDATES}"
echo "  Polymer classes: ${POLYMER_CLASS_LIST}"
echo "  Class target: ${CLASS_TARGET_VALUE}"
echo "  Class epsilon: ${CLASS_EPSILON}"

./scripts/run_step1.sh
./scripts/run_step2.sh "${NUM_SAMPLES}"

for prop in "${PROPERTIES[@]}"; do
  prop_targets="$TARGETS"
  if [ -z "$prop_targets" ]; then
    prop_targets="$(default_target_for_property "$prop")"
  fi
  prop_epsilon="$EPSILON"
  if [ -z "$prop_epsilon" ]; then
    prop_epsilon="$(default_epsilon_for_property "$prop")"
  fi
  ./scripts/run_step3.sh "${prop}"
  ./scripts/run_step4.sh "${prop}" "${prop_targets}" "${prop_epsilon}" "${NUM_CANDIDATES}"
done

for cls in "${CLASSES[@]}"; do
  ./scripts/run_step5.sh "${cls}" "" "" "" "${NUM_CANDIDATES}"
done

for cls in "${CLASSES[@]}"; do
  for prop in "${PROPERTIES[@]}"; do
    class_target="$CLASS_TARGET_VALUE"
    if [ -z "$class_target" ]; then
      class_target="$(default_target_for_property "$prop")"
    fi
    class_epsilon="$CLASS_EPSILON"
    if [ -z "$class_epsilon" ]; then
      class_epsilon="$(default_epsilon_for_property "$prop")"
    fi
    ./scripts/run_step5.sh "${cls}" "${prop}" "${class_target}" "${class_epsilon}" "${NUM_CANDIDATES}"
  done
done

echo "Done!"

#!/bin/bash
#SBATCH --job-name=gsel_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=164G
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00

# Step 1 DDP (Euler, 1 node / 1 GPU)
# Usage: sbatch scripts/submit_step1_ddp_euler.sh <model_size>

set -e

TIMING_LOG=$(mktemp)
TIMING_JSON=""
JOB_START_TS=$(date +%s)
echo "Job Start Time: $(date)"

finish() {
  local status=$?
  local end_ts
  local duration
  end_ts=$(date +%s)
  duration=$((end_ts - JOB_START_TS))
  echo "Job End Time: $(date)"
  echo "Job Duration Seconds: ${duration}"
  echo "Job Exit Status: ${status}"
  write_timing_summary "$end_ts" "$status"
}

run_step() {
  local label="$1"
  shift
  local start_ts end_ts duration status
  echo "------------------------------------------"
  echo "Step: ${label}"
  echo "Start Time: $(date)"
  start_ts=$(date +%s)
  set +e
  "$@"
  status=$?
  set -e
  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))
  echo "End Time: $(date)"
  echo "Duration Seconds: ${duration}"
  echo "Exit Status: ${status}"
  echo "------------------------------------------"
  echo -e "${label}	${start_ts}	${end_ts}	${duration}	${status}" >> "$TIMING_LOG"
  return $status
}

write_timing_summary() {
  local end_ts="$1"
  local status="$2"
  if [ -z "$TIMING_JSON" ]; then
    return 0
  fi
  local script_name
  script_name=$(basename "$0")
  set +e
  TIMING_LOG="$TIMING_LOG" \
    TIMING_JSON="$TIMING_JSON" \
    JOB_START_TS="$JOB_START_TS" \
    JOB_END_TS="$end_ts" \
    JOB_EXIT_STATUS="$status" \
    MODEL_SIZE="$MODEL_SIZE" \
    SCRIPT_NAME="$script_name" \
    python - <<'PY'
import json
import os
from datetime import datetime, timezone


def iso_utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


log_path = os.environ.get("TIMING_LOG")
json_path = os.environ.get("TIMING_JSON")
job_start_ts = int(os.environ.get("JOB_START_TS", "0") or 0)
job_end_ts = int(os.environ.get("JOB_END_TS", "0") or 0)
job_exit_status = int(os.environ.get("JOB_EXIT_STATUS", "0") or 0)
model_size = os.environ.get("MODEL_SIZE")
script_name = os.environ.get("SCRIPT_NAME") or "job"
job_name = os.environ.get("SLURM_JOB_NAME") or script_name
job_id = os.environ.get("SLURM_JOB_ID")
job_key = f"{job_name}_{job_id}" if job_id else f"{job_name}_{job_start_ts}"

steps = {}
if log_path and os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("	")
            if len(parts) != 5:
                continue
            label, start_s, end_s, duration_s, status_s = parts
            start_ts = int(start_s)
            end_ts = int(end_s)
            duration_seconds = float(duration_s)
            steps[label] = {
                "start_time_utc": iso_utc(start_ts),
                "end_time_utc": iso_utc(end_ts),
                "duration_hours": round(duration_seconds / 3600.0, 6),
                "exit_status": int(status_s),
            }

payload = {}
if json_path and os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}

if not isinstance(payload, dict):
    payload = {}

jobs = payload.get("jobs")
if not isinstance(jobs, dict):
    jobs = {}

jobs[job_key] = {
    "job_name": job_name,
    "job_id": job_id,
    "script": script_name,
    "model_size": model_size,
    "job_start_time_utc": iso_utc(job_start_ts) if job_start_ts else None,
    "job_end_time_utc": iso_utc(job_end_ts) if job_end_ts else None,
    "job_duration_hours": round((job_end_ts - job_start_ts) / 3600.0, 6) if job_start_ts and job_end_ts else None,
    "job_exit_status": job_exit_status,
    "steps": steps,
}

payload["jobs"] = jobs
payload["latest_job_key"] = job_key

if json_path:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
PY
  set -e
}

trap finish EXIT

# Conda setup
CONDA_DIR="/srv/home/htang228/anaconda3"
eval "$("$CONDA_DIR"/bin/conda shell.bash hook)"
conda activate euler_active_learning

# Work directory
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs

MODEL_SIZE=${1:-medium}

RESULTS_DIR=$(MODEL_SIZE="$MODEL_SIZE" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path('.').resolve()))
from src.utils.config import load_config
from src.utils.model_scales import get_results_dir

model_size = os.environ.get("MODEL_SIZE")
config = load_config("configs/config.yaml")
base_dir = config["paths"]["results_dir"]
results_dir = Path(get_results_dir(model_size, base_dir))
results_dir.mkdir(parents=True, exist_ok=True)
print(results_dir)
PY
)
TIMING_JSON="${RESULTS_DIR}/timing_summary.json"

# --standalone auto-selects free port, no MASTER_ADDR/MASTER_PORT needed
run_step "step1_train_backbone" torchrun \
  --standalone \
  --nproc_per_node=1 \
  scripts/step1_train_backbone.py --config configs/config.yaml --model_size "$MODEL_SIZE"

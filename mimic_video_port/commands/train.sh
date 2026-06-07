#!/usr/bin/env bash
# Launch world2action training: ensure data -> resume from S3 -> torchrun -> mirror checkpoints to S3.
#
# Scratch (/opt/dlami/nvme) is EPHEMERAL, so the checkpoint dir is mirrored to S3: we pull the latest from
# S3 before training (DCP auto-resumes from the local job dir), and `aws s3 sync` the checkpoint dir to S3
# periodically + on exit, so a stopped/terminated instance never loses a run.
#
#   bash mimic_video_port/commands/train.sh                  # smoke-sized gate (yams_smoke, 1 GPU, 3 iters)
#   EXP=yams NGPU=4 bash mimic_video_port/commands/train.sh  # full run (add GPUs freely; DDP, per-rank batch)
# Override: EXP, NGPU, W2A_DATA, W2A_S3, SYNC_EVERY (seconds), SKIP_PREPARE=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

EXP="${EXP:-yams_smoke}"
NGPU="${NGPU:-1}"
W2A_DATA="${W2A_DATA:-/opt/dlami/nvme/world2action}"
W2A_S3="${W2A_S3:-s3://ethrc-ml-data-916780037007/robot-learning/world2action}"
SYNC_EVERY="${SYNC_EVERY:-600}"   # checkpoint -> S3 mirror interval (s)

# Checkpoints land on scratch under IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints.
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$W2A_DATA/runs}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # don't block on W&B login (a trainer callback); set online if you want it
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
JOB_DIR="$IMAGINAIRE_OUTPUT_ROOT/world2action/yams/$EXP"
CKPT_LOCAL="$JOB_DIR/checkpoints"
CKPT_S3="$W2A_S3/checkpoints/$EXP"

echo "==> world2action train | exp=$EXP | ngpu=$NGPU | job=$JOB_DIR"

# 1. Ensure the dataset is on scratch (re-derives zarr from raw; reuses the S3 Reason1 cache).
[ "${SKIP_PREPARE:-0}" = "1" ] || bash "$SCRIPT_DIR/prepare_data.sh"

# 2. Resume: pull the latest checkpoint from S3 so DCP auto-resumes from the local job dir.
if aws s3 ls "$CKPT_S3/" >/dev/null 2>&1; then
  echo "  resuming: syncing checkpoints $CKPT_S3 -> $CKPT_LOCAL"
  mkdir -p "$CKPT_LOCAL"
  aws s3 sync "$CKPT_S3" "$CKPT_LOCAL"
fi

# 3. Mirror checkpoints to S3 in the background (ephemeral scratch); final sync + cleanup on exit.
(
  while true; do
    sleep "$SYNC_EVERY"
    [ -d "$CKPT_LOCAL" ] && aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" >/dev/null 2>&1 || true
  done
) &
SYNC_PID=$!
final_sync() {
  kill "$SYNC_PID" 2>/dev/null || true
  if [ -d "$CKPT_LOCAL" ]; then
    echo "  final checkpoint sync -> $CKPT_S3"
    aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" || true
  fi
}
trap final_sync EXIT

# 4. Train (GPU-count-agnostic: scale via NGPU; DDP with per-rank batch).
echo "  launching torchrun --nproc_per_node=$NGPU -m scripts.train --config=mimic_video_port/config_yams.py -- experiment=$EXP"
torchrun --nproc_per_node="$NGPU" -m scripts.train \
  --config=mimic_video_port/config_yams.py -- experiment="$EXP"

echo "==> training done (exp=$EXP)."

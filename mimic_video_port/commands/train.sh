#!/usr/bin/env bash
# Launch world2action training: ensure data -> (resume from S3) -> torchrun -> mirror checkpoints to S3.
#
# Run identity: each launch is its OWN run (own W&B id + own checkpoint dir), via a timestamped RUN_NAME by
# default -- so repeated smokes don't all pile onto a single W&B run. To resume a long run after an instance
# restart, relaunch with the SAME stable RUN_NAME: it pulls that run's checkpoint + W&B id from S3 and
# continues the same W&B curve. Scratch (/opt/dlami/nvme) is EPHEMERAL, so the checkpoint dir (+ W&B id) is
# mirrored to S3 periodically + on exit, so a stopped/terminated instance never loses a run.
#
#   bash mimic_video_port/commands/train.sh                         # smoke gate (yams_smoke, 1 GPU, 3 iters, no W&B)
#   EXP=yams_medium bash mimic_video_port/commands/train.sh         # short run WITH W&B curves + held-out eval (fresh each time)
#   EXP=yams NGPU=4 RUN_NAME=yams_full_v1 bash .../train.sh         # full run; relaunch w/ same RUN_NAME to resume it
# Override: EXP, NGPU, RUN_NAME (stable name to resume), W2A_DATA, W2A_S3, SYNC_EVERY (s), SKIP_PREPARE=1,
#   WANDB_MODE=online|offline, W2A_INSTRUCTIONS="instr a|instr b" (train a task subset),
#   OVERRIDES="trainer.max_iter=20000 checkpoint.save_iter=2000" (Hydra config overrides).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

EXP="${EXP:-yams_smoke}"
NGPU="${NGPU:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"   # gradient accumulation -> bigger effective batch (batch 1 is too noisy for the 499M decoder)
W2A_DATA="${W2A_DATA:-/opt/dlami/nvme/world2action}"
W2A_S3="${W2A_S3:-s3://ethrc-ml-data-916780037007/robot-learning/world2action}"
SYNC_EVERY="${SYNC_EVERY:-600}"   # checkpoint -> S3 mirror interval (s)

# Checkpoints land on scratch under IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints.
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$W2A_DATA/runs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# W&B: yams_medium / yams log live curves + held-out eval (yams_smoke has no W&B callback). Auto-pick the
# mode unless WANDB_MODE is set: online when credentials exist (live curves), else offline so a long run
# NEVER dies on a missing login -- offline runs record locally and sync later with `wandb sync <job_dir>`.
if [ -z "${WANDB_MODE:-}" ]; then
  if [ -n "${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
    echo "  [wandb] no credentials -> WANDB_MODE=offline. Run 'wandb login' (or export WANDB_API_KEY) for live curves."
  fi
fi
echo "  [wandb] WANDB_MODE=$WANDB_MODE"
# Run identity (see header). Timestamped by default => each launch is a fresh run with its OWN W&B id and
# checkpoint dir. Pass a stable RUN_NAME to name a long run and resume it later with the same value.
RUN_NAME="${RUN_NAME:-${EXP}-$(date +%Y%m%d-%H%M%S)}"
JOB_DIR="$IMAGINAIRE_OUTPUT_ROOT/world2action/yams/$RUN_NAME"
CKPT_LOCAL="$JOB_DIR/checkpoints"
CKPT_S3="$W2A_S3/checkpoints/$RUN_NAME"
WANDB_ID_S3="$W2A_S3/checkpoints/${RUN_NAME}.wandb_id.txt"   # sibling object (not under the checkpoints dir)
mkdir -p "$JOB_DIR"

echo "==> world2action train | exp=$EXP | run=$RUN_NAME | ngpu=$NGPU | job=$JOB_DIR"

# 1. Ensure the dataset is on scratch (re-derives zarr from raw; reuses the S3 Reason1 cache).
[ "${SKIP_PREPARE:-0}" = "1" ] || bash "$SCRIPT_DIR/prepare_data.sh"

# 2. Resume ONLY if this RUN_NAME already has checkpoints in S3 (a same-name relaunch). A fresh timestamped
#    name won't match -> clean start with a brand-new W&B id. Restoring wandb_id.txt makes a resumed run
#    continue the SAME W&B curve (init_wandb reads $JOB_DIR/wandb_id.txt).
if aws s3 ls "$CKPT_S3/" >/dev/null 2>&1; then
  echo "  resuming run '$RUN_NAME': syncing checkpoints $CKPT_S3 -> $CKPT_LOCAL"
  mkdir -p "$CKPT_LOCAL"
  aws s3 sync "$CKPT_S3" "$CKPT_LOCAL"
  aws s3 cp "$WANDB_ID_S3" "$JOB_DIR/wandb_id.txt" >/dev/null 2>&1 || true
else
  echo "  fresh run '$RUN_NAME' (no S3 checkpoint for this name -> new W&B run id)"
fi

# 3. Mirror checkpoints (+ the W&B id) to S3 in the background (ephemeral scratch); final sync on exit.
sync_up() {
  [ -d "$CKPT_LOCAL" ] && aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" >/dev/null 2>&1 || true
  [ -f "$JOB_DIR/wandb_id.txt" ] && aws s3 cp "$JOB_DIR/wandb_id.txt" "$WANDB_ID_S3" >/dev/null 2>&1 || true
}
(
  while true; do
    sleep "$SYNC_EVERY"
    sync_up
  done
) &
SYNC_PID=$!
final_sync() {
  kill "$SYNC_PID" 2>/dev/null || true
  echo "  final checkpoint sync -> $CKPT_S3"
  sync_up
}
trap final_sync EXIT

# 4. Train (GPU-count-agnostic: scale via NGPU; DDP with per-rank batch).
#    OVERRIDES: extra Hydra config overrides, e.g. OVERRIDES="trainer.max_iter=20000 checkpoint.save_iter=2000".
#    W2A_INSTRUCTIONS="instr a|instr b": train only those tasks (a data subset).
# job.name=$RUN_NAME so config.job.path_local == $JOB_DIR (checkpoints + wandb_id land where we sync from)
# and the W&B run is named for this launch. trainer.grad_accum_iter=$GRAD_ACCUM raises the effective batch:
#   effective batch = NGPU x per-rank batch_size x GRAD_ACCUM  (target >= 16 for the full run; batch 1 stalls
#   at the predict-the-mean loss floor). On 1 GPU, GRAD_ACCUM trades wall-clock for a cleaner gradient.
echo "  effective batch = NGPU($NGPU) x per-rank-batch x GRAD_ACCUM($GRAD_ACCUM)  (aim >= 16 for the full run)"
echo "  launching torchrun --nproc_per_node=$NGPU ... experiment=$EXP job.name=$RUN_NAME trainer.grad_accum_iter=$GRAD_ACCUM ${OVERRIDES:-}"
# shellcheck disable=SC2086
torchrun --nproc_per_node="$NGPU" -m scripts.train \
  --config=mimic_video_port/config_yams.py -- experiment="$EXP" job.name="$RUN_NAME" \
  trainer.grad_accum_iter="$GRAD_ACCUM" ${OVERRIDES:-}

echo "==> training done (exp=$EXP, run=$RUN_NAME)."

#!/usr/bin/env bash
# world2action DATA orchestration for an ephemeral scratch disk.
#
# Scratch (/opt/dlami/nvme on this node) is FAST but EPHEMERAL (wiped on instance stop/terminate); S3 is
# durable. The processed zarr is LARGE (~295 GB, 720p decoded frames) but CHEAP to re-derive from raw
# (~3 min CPU), so we do NOT cache it to S3 -- we re-derive it. We cache to S3 only the small artifacts
# that are expensive to make: the Reason1 instruction-embedding cache (needs the 7B on GPU) + the
# normalizer stats. Flow:
#   1. ensure raw on scratch (download from raw S3 if missing)
#   2. raw -> 720p zarr  (stable names; skips already-converted episodes)
#   3. Reason1 cache: reuse the S3 copy if local is missing (skips the 7B), else compute; then upload
#   4. normalizer stats: (re)compute (cheap) and upload
# Idempotent. UPDATE=1 forces a re-scan to pick up newly-collected raw episodes (only new ones convert;
# Reason1 only encodes new instructions). Run `bash mimic_video_port/setup.sh` first (env + data-prep deps).
#
#   bash mimic_video_port/commands/prepare_data.sh
# Override via env: W2A_DATA, W2A_S3, RAW_S3, CAMERAS={workspace,all}, NUM_WORKERS, UPDATE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

W2A_DATA="${W2A_DATA:-/opt/dlami/nvme/world2action}"   # fast NVMe scratch (EPHEMERAL)
W2A_S3="${W2A_S3:-s3://ethrc-ml-data-916780037007/robot-learning/world2action}"   # durable
RAW_S3="${RAW_S3:-s3://ethrc-ml-data-916780037007/robot-learning/teleop}"
CAMERAS="${CAMERAS:-workspace}"   # training uses only camera_top; raw S3 keeps all cams (CAMERAS=all)
NUM_WORKERS="${NUM_WORKERS:-16}"
UPDATE="${UPDATE:-0}"

CONV="$W2A_DATA/teleop_converted"
RAW="$W2A_DATA/teleop_raw"
ARTIFACTS_S3="$W2A_S3/artifacts"   # small + durable: reason1_embeddings.pt + normalizer_stats.pt

have_zarr() { [ -n "$(find "$CONV" -maxdepth 1 -name '*.zarr' 2>/dev/null | head -1)" ]; }
have_processed() { have_zarr && [ -f "$CONV/reason1_embeddings.pt" ] && [ -f "$CONV/normalizer_stats.pt" ]; }
raw_present() { [ -n "$(find "$RAW" -name session_meta.json 2>/dev/null | head -1)" ]; }

ensure_artifacts() {
  mkdir -p "$CONV"
  # Reuse the S3 Reason1 cache if we don't have it locally -> precompute_reason1 then skips the 7B.
  [ -f "$CONV/reason1_embeddings.pt" ] || aws s3 cp "$ARTIFACTS_S3/reason1_embeddings.pt" "$CONV/reason1_embeddings.pt" 2>/dev/null || true
  echo "  Reason1 instruction cache (incremental; loads the 7B only if there are new instructions)"
  python mimic_video_port/data_preprocessing/precompute_reason1.py --data-dir "$CONV"
  echo "  normalizer stats"
  python mimic_video_port/data_preprocessing/precompute_stats.py --data-dir "$CONV"
  echo "  uploading small artifacts -> $ARTIFACTS_S3"
  aws s3 cp "$CONV/reason1_embeddings.pt" "$ARTIFACTS_S3/reason1_embeddings.pt"
  aws s3 cp "$CONV/normalizer_stats.pt" "$ARTIFACTS_S3/normalizer_stats.pt"
}

echo "==> world2action data prep | scratch=$W2A_DATA | cameras=$CAMERAS | update=$UPDATE"

if have_processed && [ "$UPDATE" != "1" ]; then
  echo "  processed data already on scratch ($CONV); ensuring artifacts are cached on S3."
  ensure_artifacts
  echo "==> DONE."
  exit 0
fi

if ! raw_present; then
  echo "  downloading raw teleop from $RAW_S3 -> $RAW"
  mkdir -p "$RAW"
  aws s3 sync "$RAW_S3" "$RAW"
fi
echo "  raw -> 720p zarr (cameras=$CAMERAS, $NUM_WORKERS workers; skips already-converted; re-derived, NOT cached to S3)"
python mimic_video_port/data_preprocessing/process_recordings.py \
  --input-dir "$RAW" --output-dir "$CONV" --cameras "$CAMERAS" --num-workers "$NUM_WORKERS"
ensure_artifacts
echo "==> DONE: zarr on scratch ($CONV); Reason1+stats cached at $ARTIFACTS_S3 (zarr re-derived from raw, not stored on S3)."

#!/usr/bin/env bash
# Reproduce the validated cosmos-predict2.5 backbone foundation for the mimic-video port.
# Run after cloning this fork:   bash mimic_video_port/setup.sh
# Overrides: CUDA_EXTRA, CKPT_S3, CKPT_LOCAL, SKIP_SMOKES=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd))"
cd "$REPO_ROOT"   # uv sync needs pyproject.toml / uv.lock at the repo root

CUDA_EXTRA="${CUDA_EXTRA:-cu130}"   # Blackwell + py3.13. Hopper/Ampere: CUDA_EXTRA=cu128 and `uv python pin 3.10`.
CKPT_S3="${CKPT_S3:-s3://ethrc-ml-data-916780037007/robot-learning/checkpoints/cosmos2.5-video/2b_groot_gr1_480_run1/generate_samples_smoke-1/checkpoints/iter_000005500/model_ema_bf16.pt}"
CKPT_LOCAL="${CKPT_LOCAL:-checkpoints/model_ema_bf16.pt}"
SKIP_SMOKES="${SKIP_SMOKES:-0}"

echo "==> [1/4] uv environment (--extra=$CUDA_EXTRA) at $REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi
uv python install
uv sync --locked --extra="$CUDA_EXTRA"
# shellcheck disable=SC1091
source .venv/bin/activate
# world2action data-prep deps (zarr/mcap/imageio/cv2): not in the upstream lock, so install AFTER
# `uv sync` (sync makes the venv match the lock and would otherwise drop them). cv2 is usually already
# present from the cosmos deps — only add headless opencv if it's missing (avoids an opencv conflict).
# zarr<3 on purpose: the mimic-video data pipeline (writer + MimicDataset reader) uses the zarr v2 API
# (Group.create_dataset, numcodecs.Blosc); zarr 3 removed it. numcodecs<0.16 stays v2-compatible.
uv pip install --quiet mcap "zarr<3" "numcodecs<0.16" imageio imageio-ffmpeg threadpoolctl
python -c "import cv2" 2>/dev/null || uv pip install --quiet opencv-python-headless
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_ok", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY

echo "==> [2/4] Hugging Face access (Wan2.1 VAE + gated Cosmos-Reason1-7B)"
if ! hf auth whoami >/dev/null 2>&1 && [ -z "${HF_TOKEN:-}" ]; then
  echo "  not authenticated. Run 'hf auth login' or 'export HF_TOKEN=hf_...' then re-run." >&2; exit 1
fi
if ! hf download nvidia/Cosmos-Reason1-7B config.json >/dev/null 2>&1; then
  echo "  cannot access nvidia/Cosmos-Reason1-7B -- accept its license on HF / check token." >&2; exit 1
fi

echo "==> [3/4] Backbone checkpoint"
if [ ! -f "$CKPT_LOCAL" ]; then
  command -v aws >/dev/null 2>&1 || { echo "  aws CLI not found (needed for $CKPT_S3)." >&2; exit 1; }
  mkdir -p "$(dirname "$CKPT_LOCAL")"
  echo "  downloading $CKPT_S3"; aws s3 cp "$CKPT_S3" "$CKPT_LOCAL"
else
  echo "  found $CKPT_LOCAL"
fi

echo "==> [4/4] Validation smokes"
if [ "$SKIP_SMOKES" = "1" ]; then
  echo "  SKIP_SMOKES=1 -> skipped"
else
  python "$SCRIPT_DIR/smoke/smoke_load_ckpt.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/smoke/extract_features_smoke.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/smoke/merge_lora_smoke.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/smoke/vae_encode_smoke.py"
  python "$SCRIPT_DIR/smoke/reason1_embed_smoke.py"            # first run downloads ~15GB Cosmos-Reason1-7B
  python "$SCRIPT_DIR/smoke/get_crossattn_emb_smoke.py" "$CKPT_LOCAL"   # full end-to-end get_crossattn_emb
fi
echo "==> DONE: foundation set up & validated."
echo ""
echo "Next (data + training; scratch=/opt/dlami/nvme is fast but EPHEMERAL -> S3 is durable):"
echo "  1. bash mimic_video_port/commands/prepare_data.sh   # data: scratch <- S3 <- raw+preprocess (caches to S3)"
echo "  2. bash mimic_video_port/commands/train.sh          # train: ensure data -> resume from S3 -> torchrun -> sync ckpts to S3"
echo "     (smoke gate first: python mimic_video_port/smoke/train_config_smoke.py yams_smoke)"

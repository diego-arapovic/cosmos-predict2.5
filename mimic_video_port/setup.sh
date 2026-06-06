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
  python "$SCRIPT_DIR/smoke_load_ckpt.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/extract_features_smoke.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/merge_lora_smoke.py" "$CKPT_LOCAL"
  python "$SCRIPT_DIR/vae_encode_smoke.py"
  python "$SCRIPT_DIR/reason1_embed_smoke.py"   # first run downloads ~15GB Cosmos-Reason1-7B
fi
echo "==> DONE: foundation set up & validated."

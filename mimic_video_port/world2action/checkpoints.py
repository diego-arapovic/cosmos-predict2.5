"""Redirect cosmos-predict2.5's internal `s3://bucket/...` checkpoint placeholders to public HF.

The public release ships code that points at NVIDIA-internal S3 paths which only resolve when
`INTERNAL=True` (inside NVIDIA) or after importing the heavy `cosmos_policy` config (which drags in
h5py + sim deps). This module redirects *only* the handful we need, without those imports:

  - Reason1 model:   register the S3 URI / UUID -> HF nvidia/Cosmos-Reason1-7B
  - Qwen processor:  route any ".../Qwen_tokenizer/..." path -> public HF id Qwen/Qwen2.5-VL-7B-Instruct
  - Wan2.1 VAE:      use WAN_VAE_PATH ("hf://Wan-AI/.../Wan2.1_VAE.pth") as `vae_pth`

Call `register_external_checkpoints()` once before constructing the backbone / VAE / Reason1.
"""
from cosmos_predict2._src.imaginaire.utils import checkpoint_db as _ckpt_db
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
)

# Reason1.1-7B text encoder (the finetune's text_encoder_config.ckpt_path)
REASON1_UUID = "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f"
REASON1_S3 = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model"
REASON1_HF_REPO = "nvidia/Cosmos-Reason1-7B"
REASON1_HF_REV = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

# Qwen2.5-VL processor (Reason1's tokenizer) and the Wan2.1 VAE used by this checkpoint's latent space.
QWEN_PROCESSOR_HF = "Qwen/Qwen2.5-VL-7B-Instruct"
WAN_VAE_PATH = "hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"

_REGISTERED = False


def register_external_checkpoints() -> None:
    """Idempotently register/redirect the external checkpoints to public HF."""
    global _REGISTERED
    if _REGISTERED:
        return

    # 1) Reason1 model: register so its UUID *and* internal S3 URI resolve to the public HF repo.
    try:
        CheckpointConfig(
            uuid=REASON1_UUID,
            name="nvidia/Cosmos-Reason1.1-7B",
            s3=CheckpointDirS3(uri=REASON1_S3),
            hf=CheckpointDirHf(repository=REASON1_HF_REPO, revision=REASON1_HF_REV),
        ).register()
    except ValueError:
        pass  # already registered

    # 2) Qwen processor: its S3 path isn't in the registry; route it to the public HF id so
    #    AutoProcessor.from_pretrained() fetches just the processor files (not the 16GB model).
    _orig = _ckpt_db.download_checkpoint

    def _route(uri, *args, **kwargs):
        if isinstance(uri, str) and "Qwen_tokenizer" in uri:
            return QWEN_PROCESSOR_HF
        return _orig(uri, *args, **kwargs)

    _ckpt_db.download_checkpoint = _route
    _ckpt_db.get_checkpoint_path = _route  # `from checkpoint_db import get_checkpoint_path` reads this attr

    _REGISTERED = True

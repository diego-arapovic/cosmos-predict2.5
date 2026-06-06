#!/usr/bin/env python
"""Step 1b (part 3): Reason1 text encoder loads + embeds (last backbone dependency).

    python reason1_embed_smoke.py

Resolves the internal S3 placeholders to public HF without importing the heavy cosmos_policy
config (h5py / sim deps):
  - Reason1 model:    register UUID -> HF nvidia/Cosmos-Reason1-7B
  - Qwen processor:   route its internal S3 path -> public HF id Qwen/Qwen2.5-VL-7B-Instruct
Then embeds a sample instruction and checks dim == 3584*28 = 100352 (DiT crossattn_proj_in_channels).
This is the precompute path (reason1_embedding_utils) we'll reuse to cache teleop instructions.

Requires: `hf auth login` + access to nvidia/Cosmos-Reason1-7B (~15GB).
"""
import torch

from cosmos_predict2._src.imaginaire.utils import checkpoint_db as ckpt_db
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
)

REASON1_UUID = "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    # 1) Register Reason1 model so its UUID resolves to HF nvidia/Cosmos-Reason1-7B.
    try:
        CheckpointConfig(
            uuid=REASON1_UUID,
            name="nvidia/Cosmos-Reason1.1-7B",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model"
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
            ),
        ).register()
    except ValueError:
        pass  # already registered

    # 2) Route the Qwen tokenizer's internal S3 path -> public HF id so AutoProcessor fetches
    #    just the processor files (not the 16GB model). All other paths pass through unchanged.
    _orig = ckpt_db.download_checkpoint

    def _route(uri, *a, **k):
        if isinstance(uri, str) and "Qwen_tokenizer" in uri:
            return QWEN_ID
        return _orig(uri, *a, **k)

    ckpt_db.download_checkpoint = _route
    ckpt_db.get_checkpoint_path = _route

    from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig

    enc = TextEncoder(
        TextEncoderConfig(compute_online=True, embedding_concat_strategy="full_concat", ckpt_path=REASON1_UUID),
        device="cuda",
    )
    with torch.no_grad():
        emb = enc.compute_text_embeddings_online(
            {"ai_caption": ["pick up the red block and place it on the plate"]}, "ai_caption"
        )
    exp = 3584 * 28  # 100352 == net.crossattn_proj_in_channels
    print("[reason1] emb:", tuple(emb.shape), emb.dtype, "| expect dim =", exp)
    ok = emb.ndim == 3 and emb.shape[-1] == exp
    print("\n==> " + ("PASS: Reason1 loads + embeds; dim feeds the DiT crossattn_proj (100352->1024)"
                      if ok else "CHECK shape above"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate FrozenVideoBackbone end-to-end (the world2action feature tap), with real inputs.

    python mimic_video_port/smoke/backbone_smoke.py [checkpoints/model_ema_bf16_fused.pt]

Supersedes get_crossattn_emb_smoke.py: instead of hand-building the condition, this exercises the
*faithful* path that World2ActionModel.get_crossattn_emb will use --
  get_mimic_data_and_condition  (real VideoPredictionConditioner + Wan2.1 VAE encode + FRAME_REPLACE)
  -> draw_video_sigma           (rectified-flow logitnormal sigma in [0, 1])
  -> extract_crossattn_emb      (RF interpolation + frame-replace + native intermediate_feature_ids tap)
on a dummy mimic-style data_batch (obs RGB + action RGB + cached Reason1 language embedding),
and checks the tapped crossattn_emb is (B, T*H*W, 2048) and finite.
"""
import sys

import torch

from mimic_video_port.world2action.backbone import FrozenVideoBackbone

XATTN = 19  # == mimic-video's layer-20 tap


def main(ckpt_path):
    dev = "cuda"
    backbone = FrozenVideoBackbone.from_pretrained(ckpt_path, device=dev, dtype=torch.bfloat16)
    print(f"[backbone] net params: {sum(p.numel() for p in backbone.net.parameters()) / 1e9:.2f}B")

    B, H, W = 1, 64, 64
    T_obs_px, T_act_px = 1, 8  # cat -> 9 pixel frames -> 3 latent frames (1+(9-1)//4); obs -> 1 latent
    bf = dict(device=dev, dtype=torch.bfloat16)
    data_batch = {
        "obs/workspace_rgb": torch.rand(B, 3, T_obs_px, H, W, **bf) * 2 - 1,   # [-1, 1]
        "action/workspace_rgb": torch.rand(B, 3, T_act_px, H, W, **bf) * 2 - 1,
        "obs/language_embedding": torch.randn(B, 16, 100352, device=dev, dtype=torch.float16),  # cached Reason1 (fp16, like the real cache)
    }

    _, latent, condition = backbone.get_mimic_data_and_condition(data_batch)
    print(f"[backbone] latent {tuple(latent.shape)}  "
          f"num_conditional_frames_B={condition.num_conditional_frames_B.tolist()}")

    video_sigma_B_1 = backbone.draw_video_sigma(latent.size(), condition)
    epsilon = torch.randn(latent.size(), **bf)
    crossattn_emb = backbone.extract_crossattn_emb(latent, epsilon, video_sigma_B_1, condition, feature_id=XATTN)

    _, _, T, h, w = latent.shape
    p = backbone.net.patch_spatial  # the DiT patchifies spatially by patch_spatial (==2) before the tap
    exp_tokens = T * (h // p) * (w // p)
    D = backbone.net.model_channels
    print(f"[backbone] video_sigma={video_sigma_B_1.flatten().tolist()}  ->  crossattn_emb {tuple(crossattn_emb.shape)}")
    print(f"           expected ({B}, {exp_tokens}, {D})  [T={T} x (h/{p}) x (w/{p})]; "
          f"finite={torch.isfinite(crossattn_emb).all().item()}")

    ok = crossattn_emb.shape == (B, exp_tokens, D) and torch.isfinite(crossattn_emb).all().item()
    print("\n==> " + ("PASS: FrozenVideoBackbone get-crossattn-emb path (conditioner + RF + frame-replace + tap)"
                      if ok else "CHECK shapes/finite above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_ema_bf16_fused.pt")

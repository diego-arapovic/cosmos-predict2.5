#!/usr/bin/env python
"""Step 1a: prove the frozen 2.5 backbone RUNS and returns mid-layer features.

Loads the finetuned base weights into the 2.5 MinimalV1LVGDiT 2B, runs ONE forward on
dummy latents, and checks `intermediate_feature_ids=[XATTN]` returns (B, T*Hp*Wp, 2048) --
i.e. the `crossattn_emb` world2action will consume. No VAE / Reason1 / LoRA yet (those are 1b);
this isolates "does the feature tap execute on Blackwell with these weights + shapes".

    python extract_features_smoke.py checkpoints/model_ema_bf16.pt
"""
import copy
import sys

import torch

XATTN = 20  # xattn_layer_idx mimic-video taps


def main(ckpt_path: str) -> None:
    from cosmos_predict2._src.predict2.conditioner import DataType
    from cosmos_predict2._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET

    try:
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    except Exception:
        from cosmos_predict2._src.imaginaire.lazy_config.instantiate import instantiate

    net_cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    try:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(net_cfg, False)
    except Exception:
        pass
    net_cfg.crossattn_emb_channels = 1024
    net_cfg.use_crossattn_projection = True
    net_cfg.crossattn_proj_in_channels = 100352
    net_cfg.timestep_scale = 0.001
    net_cfg.use_wan_fp32_strategy = True
    net = instantiate(net_cfg).eval()

    # --- load finetuned base weights (LoRA adapters + stat buffers ignored for this test) ---
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj
    if isinstance(obj, dict) and not any(hasattr(v, "shape") for v in obj.values()):
        for k in ("model", "state_dict", "ema", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    for pfx in ("net_ema.", "net.", "model.net.", "module.net.", "model.", "module."):
        if tensors and all(k.startswith(pfx) for k in tensors):
            tensors = {k[len(pfx):]: v for k, v in tensors.items()}
            break
    res = net.load_state_dict(tensors, strict=False)
    print(f"[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} (unexpected = LoRA/buffers, expected)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev, torch.bfloat16)

    # --- dummy inputs in LATENT space ---
    B, C, T, H, W = 1, 16, 4, 16, 16   # C=16 VAE-latent channels; H,W even (patch_spatial=2)
    N = 16                              # number of text tokens
    bf = dict(device=dev, dtype=torch.bfloat16)
    x    = torch.randn(B, C, T, H, W, **bf)                 # noisy latent
    cmask = torch.zeros(B, 1, T, H, W, **bf)                # condition_video_input_mask (1 ch)
    pad  = torch.zeros(B, 1, H, W, **bf)                    # padding_mask (resized to H,W internally)
    ts   = torch.full((B,), 0.5, **bf)                      # timesteps
    txt  = torch.randn(B, N, 100352, **bf)                  # crossattn_emb (Reason1 full_concat dim)
    fps  = torch.full((B,), 16.0, **bf)

    with torch.no_grad():
        out = net(
            x_B_C_T_H_W=x,
            timesteps_B_T=ts,
            crossattn_emb=txt,
            condition_video_input_mask_B_C_T_H_W=cmask,
            fps=fps,
            padding_mask=pad,
            data_type=DataType.VIDEO,
            intermediate_feature_ids=[XATTN],
        )

    assert isinstance(out, tuple), f"expected (output, features) tuple, got {type(out)}"
    pred, feats = out
    exp_tokens = T * (H // net.patch_spatial) * (W // net.patch_spatial)
    print(f"[fwd] denoiser output : {tuple(pred.shape)}")
    print(f"[fwd] #features={len(feats)}  feat[0]={tuple(feats[0].shape)}")
    print(f"[fwd] expected feat   : (B={B}, tokens={exp_tokens}, D={net.model_channels})")
    f = feats[0]
    ok = f.shape == (B, exp_tokens, net.model_channels)
    print("\n==> " + ("PASS: backbone runs; intermediate feature == crossattn_emb of the right shape"
                      if ok else "MISMATCH: see shapes above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_ema_bf16.pt")

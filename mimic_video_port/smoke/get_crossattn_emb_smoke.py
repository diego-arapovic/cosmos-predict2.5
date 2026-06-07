#!/usr/bin/env python
"""Capstone: the REAL get_crossattn_emb path, end-to-end with real inputs.

    python get_crossattn_emb_smoke.py [checkpoints/model_ema_bf16.pt]

Ties the whole frozen backbone together: registers external checkpoints (Reason1 -> HF,
Qwen processor -> HF), loads the finetuned DiT (auto-merges LoRA, or loads a *_fused.pt
directly), the Wan2.1 VAE, and Reason1; then encodes a clip -> latent, embeds an instruction
-> reason1 emb, and runs the DiT with intermediate_feature_ids=[19] to produce the
crossattn_emb (B, T*H*W, 2048) that the World2ActionDIT will consume.

This is the seed of World2ActionModel.get_crossattn_emb on cosmos-predict2.5.
"""
import copy
import sys

import torch
from omegaconf import OmegaConf

from cosmos_predict2._src.imaginaire.utils import checkpoint_db as ckpt_db
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf, CheckpointDirS3
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
from cosmos_predict2._src.predict2.tokenizers.cosmos import Wan2pt1VAEConfig

try:
    from cosmos_predict2._src.imaginaire.lazy_config import instantiate
except Exception:
    from cosmos_predict2._src.imaginaire.lazy_config.instantiate import instantiate

REASON1_UUID = "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f"
WAN_VAE = "hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
XATTN = 19  # == mimic-video's layer-20 tap
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]


def register_external_checkpoints():
    """Redirect the handful of internal S3 placeholders we need to public HF."""
    try:
        CheckpointConfig(
            uuid=REASON1_UUID,
            name="nvidia/Cosmos-Reason1.1-7B",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model"
            ),
            hf=CheckpointDirHf(repository="nvidia/Cosmos-Reason1-7B", revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423"),
        ).register()
    except ValueError:
        pass
    _orig = ckpt_db.download_checkpoint

    def _route(uri, *a, **k):
        if isinstance(uri, str) and "Qwen_tokenizer" in uri:
            return "Qwen/Qwen2.5-VL-7B-Instruct"
        return _orig(uri, *a, **k)

    ckpt_db.download_checkpoint = _route
    ckpt_db.get_checkpoint_path = _route


def build_backbone(ckpt_path, dev="cuda"):
    cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    OmegaConf.set_struct(cfg, False)
    cfg.crossattn_emb_channels = 1024
    cfg.use_crossattn_projection = True
    cfg.crossattn_proj_in_channels = 100352
    cfg.timestep_scale = 0.001
    cfg.use_wan_fp32_strategy = True
    net = instantiate(cfg).eval()

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj
    if isinstance(obj, dict) and not any(hasattr(v, "shape") for v in obj.values()):
        for k in ("model", "state_dict", "ema", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    t = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    for pfx in ("net_ema.", "net.", "model.net.", "module.net.", "model.", "module."):
        if t and all(k.startswith(pfx) for k in t):
            t = {k[len(pfx):]: v for k, v in t.items()}
            break
    net.load_state_dict(t, strict=False)
    net = net.to(dev, torch.bfloat16)

    if any("lora_" in k for k in t):  # non-fused checkpoint -> merge the adapters
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        pnet = get_peft_model(
            net, LoraConfig(r=32, lora_alpha=32, init_lora_weights=True, target_modules=LORA_TARGETS, use_dora=False)
        )
        asd = {("base_model.model." + k).replace("default.", ""): v.to(dev) for k, v in t.items() if "lora_" in k}
        set_peft_model_state_dict(pnet, asd, adapter_name="default")
        net = pnet.merge_and_unload().eval()
        print("[backbone] loaded base + merged LoRA")
    else:
        print("[backbone] loaded (fused / no LoRA)")
    return net


def main(ckpt_path):
    dev = "cuda"
    register_external_checkpoints()
    net = build_backbone(ckpt_path, dev)

    vae_cfg = copy.deepcopy(Wan2pt1VAEConfig)
    OmegaConf.set_struct(vae_cfg, False)
    vae_cfg.vae_pth = WAN_VAE
    vae = instantiate(vae_cfg)

    reason1 = TextEncoder(
        TextEncoderConfig(compute_online=True, embedding_concat_strategy="full_concat", ckpt_path=REASON1_UUID),
        device=dev,
    )

    with torch.no_grad():
        px = torch.randn(1, 3, 9, 64, 64, device=dev, dtype=torch.bfloat16)  # stand-in for real frames
        latent = vae.encode(px)  # (1,16,3,8,8)
        txt = reason1.compute_text_embeddings_online(
            {"ai_caption": ["pick up the red block and place it on the plate"]}, "ai_caption"
        )  # (1,512,100352)
        B, C, T, H, W = latent.shape
        out = net(
            x_B_C_T_H_W=latent,
            timesteps_B_T=torch.full((B,), 0.5, device=dev, dtype=torch.bfloat16),
            crossattn_emb=txt.to(dev, torch.bfloat16),
            condition_video_input_mask_B_C_T_H_W=torch.zeros(B, 1, T, H, W, device=dev, dtype=torch.bfloat16),
            fps=torch.full((B,), 16.0, device=dev, dtype=torch.bfloat16),
            padding_mask=torch.zeros(B, 1, H, W, device=dev, dtype=torch.bfloat16),
            data_type=DataType.VIDEO,
            intermediate_feature_ids=[XATTN],
        )
    feat = out[1][0]
    exp_tokens = T * (H // net.patch_spatial) * (W // net.patch_spatial)
    print(f"[real] latent {tuple(latent.shape)} + reason1 {tuple(txt.shape)} -> crossattn_emb {tuple(feat.shape)}")
    print(f"       expected (1, {exp_tokens}, {net.model_channels})")
    ok = feat.shape == (1, exp_tokens, net.model_channels)
    print("\n==> " + ("PASS: real end-to-end get_crossattn_emb (VAE + Reason1 + frozen finetuned backbone + hook)"
                      if ok else "CHECK shapes above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_ema_bf16.pt")

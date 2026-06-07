#!/usr/bin/env python
"""Step 1b (part 1): reconstruct the FINETUNED backbone = base + LoRA, merged.

Loads the base weights, injects PEFT LoRA adapters exactly as the repo does
(models/text2world_model_rectified_flow.py::add_lora + utils/model_loader.py),
loads YOUR adapter weights, merges them, and confirms the merge is non-trivial by
showing the tapped feature changes vs base-only. No VAE / Reason1 needed.

    python merge_lora_smoke.py checkpoints/model_ema_bf16.pt
"""
import copy
import sys

import torch

XATTN = 19  # 0-based; == mimic-video's layer-20 tap (after blocks[19])

LORA_RANK = 32
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]


def build_net():
    from cosmos_predict2._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
    try:
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    except Exception:
        from cosmos_predict2._src.imaginaire.lazy_config.instantiate import instantiate
    cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    try:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    cfg.crossattn_emb_channels = 1024
    cfg.use_crossattn_projection = True
    cfg.crossattn_proj_in_channels = 100352
    cfg.timestep_scale = 0.001
    cfg.use_wan_fp32_strategy = True
    return instantiate(cfg).eval()


def load_ckpt_tensors(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj
    if isinstance(obj, dict) and not any(hasattr(v, "shape") for v in obj.values()):
        for k in ("model", "state_dict", "ema", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]; break
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    for pfx in ("net_ema.", "net.", "model.net.", "module.net.", "model.", "module."):
        if tensors and all(k.startswith(pfx) for k in tensors):
            tensors = {k[len(pfx):]: v for k, v in tensors.items()}; break
    return tensors


def dummy_inputs(dev):
    from cosmos_predict2._src.predict2.conditioner import DataType
    torch.manual_seed(0)
    B, C, T, H, W, N = 1, 16, 4, 16, 16, 16
    bf = dict(device=dev, dtype=torch.bfloat16)
    return dict(
        x_B_C_T_H_W=torch.randn(B, C, T, H, W, **bf),
        timesteps_B_T=torch.full((B,), 0.5, **bf),
        crossattn_emb=torch.randn(B, N, 100352, **bf),
        condition_video_input_mask_B_C_T_H_W=torch.zeros(B, 1, T, H, W, **bf),
        fps=torch.full((B,), 16.0, **bf),
        padding_mask=torch.zeros(B, 1, H, W, **bf),
        data_type=DataType.VIDEO,
        intermediate_feature_ids=[XATTN],
    )


@torch.no_grad()
def tap(net, inputs):
    return net(**inputs)[1][0]


def main(ckpt_path):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = load_ckpt_tensors(ckpt_path)

    # 1) base-only finetuned net
    net = build_net()
    res = net.load_state_dict(tensors, strict=False)
    print(f"[base] loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    net = net.to(dev, torch.bfloat16)
    inputs = dummy_inputs(dev)
    feat_base = tap(net, inputs).float().clone()

    # 2) inject LoRA (PEFT) exactly as the repo does
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights=True,
                     target_modules=LORA_TARGETS, use_dora=False)
    pnet = get_peft_model(net, cfg)

    # 3) load YOUR adapter weights (remap to PEFT layout, drop the 'default.' tag)
    adapter_sd = {}
    for k, v in tensors.items():
        if "lora_" in k:
            adapter_sd[("base_model.model." + k).replace("default.", "")] = v.to(dev)
    lr = set_peft_model_state_dict(pnet, adapter_sd, adapter_name="default")
    print(f"[lora] adapter tensors={len(adapter_sd)}  set_peft missing={len(lr.missing_keys)} unexpected={len(lr.unexpected_keys)}")

    # 4) merge -> plain finetuned net, and re-tap
    merged = pnet.merge_and_unload().eval()
    feat_merged = tap(merged, inputs).float()

    diff = (feat_base - feat_merged).abs().mean().item()
    rel = diff / (feat_base.abs().mean().item() + 1e-8)
    print(f"\n[feat] base={tuple(feat_base.shape)}  mean|Δ(base,merged)|={diff:.6f}  relative={rel:.4f}")
    nontrivial = len(adapter_sd) > 0 and len(lr.unexpected_keys) == 0 and diff > 1e-4
    print("\n==> " + ("PASS: LoRA loaded + merged; finetuned backbone reconstructed (features shift after merge)"
                      if nontrivial else "CHECK: see adapter/diff numbers above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_ema_bf16.pt")

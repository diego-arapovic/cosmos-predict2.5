#!/usr/bin/env python
"""Smoke test: does the finetuned Cosmos-Predict2.5 checkpoint load into the 2.5 Video2World net?

Run on the GPU node, inside the cosmos-predict2.5 repo with the env active:
    python smoke_load_ckpt.py checkpoints/model_ema_bf16.pt

Isolated test of the DiT weights ONLY -- no VAE, no Reason1, no S3. It builds the net from the
repo's own `cosmos_v1_2B` config (+ the reason-embeddings/rectified-flow overrides from your
run's config.yaml) and checks that the checkpoint's tensors map onto it.
"""
import copy
import sys

import torch


def main(ckpt_path: str) -> None:
    # --- 1. Build the 2.5 2B Video2World net exactly as the run's config.yaml specifies ---
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
    # Overrides present in your config.yaml -> model.config.net (reason-embeddings / RF):
    net_cfg.crossattn_emb_channels = 1024
    net_cfg.use_crossattn_projection = True
    net_cfg.crossattn_proj_in_channels = 100352
    net_cfg.timestep_scale = 0.001
    net_cfg.use_wan_fp32_strategy = True

    net = instantiate(net_cfg).eval()
    model_sd = net.state_dict()
    model_keys = set(model_sd.keys())
    nparams = sum(p.numel() for p in net.parameters()) / 1e9
    print(f"[net] MinimalV1LVGDiT 2B built: {len(model_keys)} state-dict entries, {nparams:.3f}B params")

    # --- 2. Load checkpoint, locate the tensor state_dict ---
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[ckpt] top-level type: {type(obj).__name__}")
    sd = obj
    if isinstance(obj, dict) and not any(hasattr(v, "shape") for v in obj.values()):
        for k in ("model", "state_dict", "ema", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                print(f"[ckpt] descending into '{k}'")
                sd = obj[k]
                break
        else:
            print(f"[ckpt] top-level keys: {list(obj.keys())[:20]}")
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    print(f"[ckpt] {len(tensors)} tensors; sample keys: {list(tensors.keys())[:5]}")

    # --- 3. Strip a common prefix (net. / net_ema. / model.) to align with the net ---
    prefix = ""
    for pfx in ("net_ema.", "net.", "model.net.", "module.net.", "model.", "module."):
        if tensors and all(k.startswith(pfx) for k in tensors):
            prefix = pfx
            break
    if prefix:
        tensors = {k[len(prefix):]: v for k, v in tensors.items()}
        print(f"[ckpt] stripped common prefix: {prefix!r}")
    ckpt_keys = set(tensors.keys())

    # --- 4. Compare keys + shapes ---
    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys      # net expects, ckpt lacks
    unexpected = ckpt_keys - model_keys   # ckpt has, net lacks
    mism = [
        (k, tuple(tensors[k].shape), tuple(model_sd[k].shape))
        for k in matched
        if tuple(tensors[k].shape) != tuple(model_sd[k].shape)
    ]
    print(f"\n  matched        : {len(matched)}/{len(model_keys)}")
    print(f"  missing        : {len(missing)}   e.g. {sorted(missing)[:8]}")
    print(f"  unexpected     : {len(unexpected)}   e.g. {sorted(unexpected)[:8]}")
    print(f"  shape-mismatch : {len(mism)}   e.g. {mism[:8]}")

    # --- 5. Actually load ---
    res = net.load_state_dict(tensors, strict=False)
    clean = (len(missing) == 0 and len(mism) == 0)
    print(f"\n  load_state_dict(strict=False): missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    print("\n==> " + ("PASS: checkpoint maps cleanly onto the 2.5 net" if clean
                      else "PARTIAL: inspect missing / shape-mismatch above"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python smoke_load_ckpt.py <path-to-model_ema_bf16.pt>")
    main(sys.argv[1])

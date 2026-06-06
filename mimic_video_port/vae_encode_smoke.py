#!/usr/bin/env python
"""Step 1b (part 2): Wan2.1 VAE loads + encodes pixels -> latents (right shape).

    python vae_encode_smoke.py

Confirms the VAE downloads (checkpoint_db -> HF), runs on Blackwell, and the
pixel->latent contract: spatial /8, temporal /4, 16 latent channels,
get_latent_num_frames(n) = 1 + (n-1)//4  (so 93 px frames -> 24 latent = model state_t).
"""
import torch


def main():
    from cosmos_predict2._src.predict2.tokenizers.cosmos import Wan2pt1VAEConfig
    try:
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    except Exception:
        from cosmos_predict2._src.imaginaire.lazy_config.instantiate import instantiate

    import copy
    from omegaconf import OmegaConf
    cfg = copy.deepcopy(Wan2pt1VAEConfig)
    OmegaConf.set_struct(cfg, False)
    # The default vae_pth is an internal NVIDIA S3 placeholder (only resolves when INTERNAL=True).
    # nvidia/Cosmos-Predict2.5-2B ships only tokenizer.pth; the Wan2.1 VAE this checkpoint was trained
    # with is public here. cosmos WanVAE_ is built to load this exact file.
    cfg.vae_pth = "hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    vae = instantiate(cfg)  # cuda/bf16; downloads the VAE from HF
    print("[vae] loaded:", type(vae).__name__,
          "| spatial =", vae.spatial_compression_factor,
          "| temporal =", vae.temporal_compression_factor)
    print("[vae] get_latent_num_frames(93) =", vae.get_latent_num_frames(93), "(expect 24)")

    T_pix, H, W = 9, 64, 64  # small clip; H,W divisible by 8
    px = torch.randn(1, 3, T_pix, H, W, device="cuda", dtype=torch.bfloat16)  # pixels ~[-1,1]
    with torch.no_grad():
        lat = vae.encode(px)
    exp = (1, 16, vae.get_latent_num_frames(T_pix), H // 8, W // 8)
    print(f"[vae] pixels {tuple(px.shape)} -> latent {tuple(lat.shape)}  (expect {exp})")
    print("\n==> " + ("PASS: Wan2.1 VAE loads + encodes; latent shape correct"
                      if tuple(lat.shape) == exp else "MISMATCH: see shapes above"))


if __name__ == "__main__":
    main()

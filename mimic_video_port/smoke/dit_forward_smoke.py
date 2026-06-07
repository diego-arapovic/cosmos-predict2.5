#!/usr/bin/env python
"""Validate the ported World2ActionDIT actually RUNS on cu130/Blackwell (not just imports).

    python mimic_video_port/smoke/dit_forward_smoke.py

Instantiates the decoder with the real `yams` config (state/action dim 14, horizon 16,
crossattn_emb_channels=2048, flash_attn_no_cp) and runs one forward with dummy inputs shaped
like the real ones:
  state_B_HO_O          (B, H_obs, 14)
  xt_B_HA_A             (B, H_action, 14)        H_obs + H_action == max_horizon (16)
  timesteps_B_T         (B, max_horizon)         per-token action sigma
  context_timesteps_B_1 (B, 1)                   video sigma
  crossattn_emb         (B, N, 2048)             frozen-backbone features
-> output (B, max_horizon, 14). Catches flash-attn / apply_rotary_pos_emb runtime API drift.
"""
import torch

from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig
from mimic_video_port.world2action.dit import World2ActionDIT


def main():
    dev = "cuda"
    net = World2ActionDIT(
        max_horizon=16,
        in_channels=14,
        out_channels=14,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ).to(dev, torch.bfloat16).eval()
    print(f"[dit] World2ActionDIT built: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")

    B, H_obs, H_action, N = 1, 1, 15, 48  # H_obs + H_action == max_horizon (16)
    bf = dict(device=dev, dtype=torch.bfloat16)
    state = torch.randn(B, H_obs, 14, **bf)
    action = torch.randn(B, H_action, 14, **bf)
    timesteps = torch.rand(B, 16, **bf)      # per-token action sigma over the full sequence
    ctx_t = torch.rand(B, 1, **bf)           # video sigma
    cross = torch.randn(B, N, 2048, **bf)    # frozen-backbone crossattn_emb

    with torch.no_grad():
        out = net(state, action, timesteps, ctx_t, cross, obs_dropout=0.0)

    print(f"[dit] forward out: {tuple(out.shape)}  (expect (1, 16, 14))")
    ok = out.shape == (B, 16, 14)
    print("\n==> " + ("PASS: World2ActionDIT runs on cu130 (flash-attn + rope + cross-attn to 2048-d features)"
                      if ok else "CHECK shape above"))


if __name__ == "__main__":
    main()

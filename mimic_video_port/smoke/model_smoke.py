#!/usr/bin/env python
"""Validate World2ActionModel: one training_step (+ backward) on a dummy mimic-style batch.

    python mimic_video_port/smoke/model_smoke.py [checkpoints/model_ema_bf16_fused.pt]

Builds the full model (trainable World2ActionDIT decoder + frozen FrozenVideoBackbone), runs
training_step on a dummy batch (obs/action RGB + cached Reason1 emb + proprio state + action chunk),
checks the loss is finite, then backward() and checks: decoder params get finite grads while the
frozen backbone gets none. Exercises the whole train path -- backbone feature tap -> decoder denoise
-> rectified-flow MSE loss -> backward into the decoder only.
"""
import sys

import torch
from omegaconf import OmegaConf

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig
from mimic_video_port.world2action.config import EMAConfig, SchedulerConfig, World2ActionPipelineConfig
from mimic_video_port.world2action.dit import World2ActionDIT
from mimic_video_port.world2action.model import World2ActionModel, World2ActionModelConfig


def build_config(video_dit_path):
    # the `yams` action decoder: state/action dim 14, horizon 16, cross-attn to the 2048-d tap.
    net = L(World2ActionDIT)(
        max_horizon=16,
        in_channels=14,
        out_channels=14,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,  # == backbone.net.model_channels (the feature-tap width)
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=L(SACConfig)(mode="none", every_n_blocks=1),
    )
    pipe_config = World2ActionPipelineConfig(
        precision="bfloat16",
        scheduler=SchedulerConfig(alpha=1.5, beta=1.0, num_denoising_steps=10),
        net=net,
        ema=EMAConfig(enabled=False),
        xattn_layer_idx=19,  # == mimic-video's layer-20 tap (native intermediate_feature_ids)
    )
    return World2ActionModelConfig(
        train_architecture="base",  # train the full decoder (no LoRA on the decoder)
        lora_rank=0,
        lora_alpha=0,
        lora_target_modules="",
        init_lora_weights=True,
        precision="bfloat16",
        loss_reduce="mean",
        loss_scale=1.0,
        ema=EMAConfig(enabled=False),
        action_dit_path="",  # random-init decoder
        video_dit_path=video_dit_path,  # frozen 2B backbone
        pipe_config=pipe_config,
        fsdp_shard_size=0,  # no FSDP in the smoke
        data_config=OmegaConf.create({}),  # only needed by on_train_start (normalizer; not called here)
        video_state_t=24,
        video_shift=5,
        video_fps=16.0,
    )


def main(ckpt_path):
    dev = "cuda"
    model = World2ActionModel(build_config(ckpt_path))

    B, H, W = 1, 64, 64
    bf = dict(device=dev, dtype=torch.bfloat16)
    data_batch = {
        "obs/workspace_rgb": torch.rand(B, 3, 1, H, W, **bf) * 2 - 1,    # obs frames -> 1 latent (clean ctx)
        "action/workspace_rgb": torch.rand(B, 3, 8, H, W, **bf) * 2 - 1,  # future frames
        "obs/language_embedding": torch.randn(B, 1, 16, 100352, device=dev, dtype=torch.float16),  # cached Reason1 (fp16)
        "obs/lowdim_concat": torch.randn(B, 1, 14, **bf),                 # proprio state (H_obs=1)
        "action/lowdim_concat": torch.randn(B, 15, 14, **bf),            # action chunk (H_action=15; +H_obs=16)
    }

    output_batch, loss = model.training_step(data_batch, iteration=0)
    print(f"[model] training_step loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}  keys={list(output_batch)}")

    loss.backward()
    dec = [p for p in model.pipe.dit.parameters() if p.requires_grad]
    n_grad = sum(1 for p in dec if p.grad is not None)
    g_finite = all(torch.isfinite(p.grad).all() for p in dec if p.grad is not None)
    bb_grads = [p for p in model.backbone.net.parameters() if p.grad is not None]
    print(f"[model] decoder trainable params: {len(dec)}; with grad: {n_grad}; grads finite: {g_finite}")
    print(f"[model] frozen backbone params with grad: {len(bb_grads)} (expect 0)")

    ok = (
        torch.isfinite(loss).item()
        and n_grad > 0
        and g_finite
        and len(bb_grads) == 0
    )
    print("\n==> " + ("PASS: World2ActionModel training_step + backward (decoder trains, backbone frozen)"
                      if ok else "CHECK above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_ema_bf16_fused.pt")

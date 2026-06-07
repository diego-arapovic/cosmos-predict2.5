# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FrozenVideoBackbone — the Cosmos-Predict2.5 replacement for mimic-video's frozen Video2WorldPipeline.

world2action trains a small action decoder that cross-attends to the *frozen* video world-model's
intermediate hidden states. In mimic-video (Cosmos-Predict2) that frozen backbone was a
`Video2WorldPipeline` (a BasePipeline). Cosmos-Predict2.5 has no such pipeline class — the backbone
logic lives in `Video2WorldModelRectifiedFlow` (an ImaginaireModel, heavyweight: EMA/FSDP/text-encoder
construction + Hydra config). This module re-creates *only the slice world2action needs*, built
piecemeal exactly like the validated capstone smoke (`get_crossattn_emb_smoke.py`), so it is
self-contained (no Hydra, no h5py/sim deps) and testable on its own.

It mirrors the three things `World2ActionModel` calls on the old `video2world_pipe`:
  - `get_mimic_data_and_condition(data_batch)` -> (raw_state, latent_B_C_T_H_W, condition)
  - `draw_video_sigma(x0_size, condition)`      -> per-sample video noise level sigma in [0, 1]
  - `extract_crossattn_emb(...)`                -> the (B, T*H*W, D) features (the old
                                                   `denoise(..., return_only_hidden_states_up_to=idx)`)

Key adaptation vs mimic-video (necessary because the backbone changed):
  - Noising is **pure rectified flow** (2.5 convention): xt = sigma*eps + (1-sigma)*x0, and the net
    receives a *discrete* timestep in [0, 1000] (= sigma * 1000), NOT the additive `x + eps*sigma`
    EDM-style noising mimic-video used. This matches `Video2WorldModelRectifiedFlow.forward/denoise`.
  - The hidden-state tap is **native**: `net(..., intermediate_feature_ids=[idx])` returns
    `(output, [feat])` with `feat` already shaped `(B, T*H*W, D)` — no reshape needed. Tap index
    `idx == 19` corresponds to mimic-video's layer-20 (`hidden_states[20]`, after `blocks[19]`).
  - FRAME_REPLACE conditioning (first `num_conditional_frames` latent frames are the clean obs) is
    replicated from `Video2WorldModelRectifiedFlow.denoise`.

Net / VAE / conditioner config values are pinned to the finetuned backbone's experiment config
`T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_26_SIZE_2B_RES_720_FPS16_RECTIFIED_FLOW`.
"""

import copy

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import VideoPredictionConditioner
from cosmos_predict2._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_predict2._src.predict2.tokenizers.cosmos import Wan2pt1VAEConfig

from .checkpoints import WAN_VAE_PATH, register_external_checkpoints

# LoRA adapter targets, used only when loading a non-fused (base + LoRA) checkpoint.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]

# Net overrides matching the finetuned backbone's config (see module docstring). These do not change
# the parameter shapes (the checkpoint loads 689/689, 0 mismatch) but the rope ratios + projection
# affect the forward numerically, so we pin them for faithful feature extraction.
_NET_OVERRIDES = dict(
    crossattn_emb_channels=1024,
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    timestep_scale=0.001,
    use_wan_fp32_strategy=True,
    rope_enable_fps_modulation=False,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,  # 24 / 24
)


def _strip_prefix(sd: dict) -> dict:
    """Pull the bare DiT tensors out of a consolidated checkpoint (drop net./net_ema./model. ...)."""
    t = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    for pfx in ("net_ema.", "net.", "model.net.", "module.net.", "model.", "module."):
        if t and all(k.startswith(pfx) for k in t):
            t = {k[len(pfx):]: v for k, v in t.items()}
            break
    return t


def _load_dit_weights(net: nn.Module, dit_path: str, device: str) -> nn.Module:
    """Load a consolidated finetuned checkpoint into `net`; merge LoRA if the ckpt is non-fused.

    Mirrors the validated capstone (`get_crossattn_emb_smoke.build_backbone`): a `*_fused.pt`
    (EMA + LoRA pre-merged) loads directly; a base+LoRA ckpt is loaded then merged via PEFT.
    """
    obj = torch.load(dit_path, map_location="cpu", weights_only=False)
    sd = obj
    if isinstance(obj, dict) and not any(hasattr(v, "shape") for v in obj.values()):
        for k in ("model", "state_dict", "ema", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    t = _strip_prefix(sd)
    net.load_state_dict(t, strict=False)
    net = net.to(device, torch.bfloat16)

    if any("lora_" in k for k in t):  # non-fused checkpoint -> merge the adapters
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        pnet = get_peft_model(
            net, LoraConfig(r=32, lora_alpha=32, init_lora_weights=True, target_modules=LORA_TARGETS, use_dora=False)
        )
        asd = {("base_model.model." + k).replace("default.", ""): v.to(device) for k, v in t.items() if "lora_" in k}
        set_peft_model_state_dict(pnet, asd, adapter_name="default")
        net = pnet.merge_and_unload().eval()
        log.info("[backbone] loaded base DiT + merged LoRA adapters")
    else:
        log.info("[backbone] loaded DiT (fused / no LoRA)")
    return net


class FrozenVideoBackbone(nn.Module):
    """Frozen Cosmos-Predict2.5 2B Video2World backbone, exposing the world2action feature tap."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        self.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
        self.net: nn.Module = None
        self.tokenizer = None
        self.conditioner: nn.Module = None
        self.rectified_flow: RectifiedFlow = None
        self.state_t: int = 24
        self.fps: float = 16.0

    @staticmethod
    def from_pretrained(
        dit_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        state_t: int = 24,
        fps: float = 16.0,
        shift: int = 5,
        train_time_distribution: str = "logitnormal",
        temporal_window: int = 16,
        vae_pth: str = WAN_VAE_PATH,
    ) -> "FrozenVideoBackbone":
        """Build the frozen backbone: DiT (+ckpt) + Wan2.1 VAE + conditioner + rectified-flow helper."""
        register_external_checkpoints()
        self = FrozenVideoBackbone(device=device, dtype=dtype)
        self.state_t = state_t
        self.fps = fps

        # 1. DiT net (cosmos_v1_2B + finetune overrides), load consolidated finetuned weights.
        net_cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
        OmegaConf.set_struct(net_cfg, False)
        for k, v in _NET_OVERRIDES.items():
            setattr(net_cfg, k, v)
        net = instantiate(net_cfg).eval()
        if dit_path:
            net = _load_dit_weights(net, dit_path, device)
        else:
            net = net.to(device, dtype)
            log.warning("[backbone] no dit_path given; using randomly-initialised DiT (smoke only)")
        self.net = net.eval().requires_grad_(False)

        # 2. Wan2.1 VAE (16-ch latent, x8 spatial / x4 temporal).
        vae_cfg = copy.deepcopy(Wan2pt1VAEConfig)
        OmegaConf.set_struct(vae_cfg, False)
        vae_cfg.vae_pth = vae_pth
        vae_cfg.temporal_window = temporal_window
        self.tokenizer = instantiate(vae_cfg)

        # 3. Conditioner (text -> crossattn_emb, fps, padding_mask, use_video_condition flag).
        self.conditioner = instantiate(VideoPredictionConditioner).eval().requires_grad_(False)

        # 4. Rectified-flow helper for train-time sampling + interpolation (velocity_field unused here).
        self.rectified_flow = RectifiedFlow(
            velocity_field=self.net,
            train_time_distribution=train_time_distribution,
            shift=shift,
            device=torch.device(device),
            dtype=torch.float32,
        )
        return self

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, raw_state: torch.Tensor) -> torch.Tensor:
        """Pixel video (B, C, T, H, W) in [-1, 1] -> latent (B, 16, T', H/8, W/8). sigma_data == 1."""
        return self.tokenizer.encode(raw_state)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return self.tokenizer.get_latent_num_frames(num_pixel_frames)

    # ----------------------------------------------------- data + conditioning
    @torch.no_grad()
    def get_mimic_data_and_condition(self, data_batch: dict[str, torch.Tensor]):
        """Mimic-video's obs+action -> (raw_state, latent, Video2WorldCondition), routed through 2.5.

        The obs RGB frames become the clean FRAME_REPLACE condition; the action RGB frames are the
        future to be (notionally) predicted. Language is fed via `obs/language_embedding` (cached
        Reason1 full_concat, 100352-d), mapped to the conditioner's `t5_text_embeddings` key.
        """
        raw_state = torch.cat(
            (data_batch["obs/workspace_rgb"], data_batch["action/workspace_rgb"]), dim=2
        )
        latent_state = self.encode(raw_state).contiguous().float()
        B, _C, _T, H, W = latent_state.shape

        cond_batch = dict(data_batch)  # shallow copy; don't mutate the caller's batch
        # The cached Reason1 embedding is stored fp16 (to save space); the net computes in bf16, so cast
        # to the backbone dtype here or crossattn_proj hits "mat1 and mat2 must have the same dtype".
        cond_batch["t5_text_embeddings"] = data_batch["obs/language_embedding"].to(**self.tensor_kwargs)
        cond_batch["fps"] = torch.full((B,), self.fps, **self.tensor_kwargs)
        cond_batch["padding_mask"] = torch.zeros(B, 1, H, W, **self.tensor_kwargs)

        # Deterministic feature extraction: no text / video-condition dropout (frozen extractor).
        condition = self.conditioner(
            cond_batch, override_dropout_rate={"text": 0.0, "use_video_condition": 0.0}
        )
        condition = condition.edit_data_type(DataType.VIDEO)

        num_obs_latent = self.tokenizer.get_latent_num_frames(data_batch["obs/workspace_rgb"].shape[2])
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_obs_latent,
        )
        return raw_state, latent_state, condition

    # ------------------------------------------------------------- noise level
    def draw_video_sigma(self, x0_size: torch.Size, condition=None) -> torch.Tensor:
        """Sample a per-sample video noise level sigma in [0, 1] (rectified-flow, shift-warped).

        Returns sigma_B_1 of shape (B, 1). The decoder is conditioned on this value
        (`context_timesteps_B_1`) so it knows how noisy the tapped features are.
        """
        del condition
        B = x0_size[0]
        t_B = self.rectified_flow.sample_train_time(B)  # [0, 1], logitnormal
        timesteps_B = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)  # [0, 1000]
        sigma_B = self.rectified_flow.get_sigmas(timesteps_B, self.tensor_kwargs_fp32)  # [0, 1]
        return rearrange(sigma_B, "b -> b 1")

    # ----------------------------------------------------- the feature tap
    @torch.no_grad()
    def extract_crossattn_emb(
        self,
        video_B_C_T_H_W: torch.Tensor,
        video_epsilon_B_C_T_H_W: torch.Tensor,
        video_sigma_B_1: torch.Tensor,
        condition,
        feature_id: int,
    ) -> torch.Tensor:
        """Noise the video at `video_sigma`, run the frozen DiT, return the tapped features.

        Replaces mimic-video's
            `video2world_pipe.denoise(video + eps*sigma, sigma, condition,
                                      return_only_hidden_states_up_to=idx).hidden_states[idx]`.
        Returns crossattn_emb of shape (B, T*H*W, model_channels) (already flattened by the tap).
        """
        sigma_B = video_sigma_B_1.squeeze(-1).to(**self.tensor_kwargs_fp32)
        # Rectified-flow interpolation: xt = sigma*noise + (1-sigma)*clean   (x_0=noise, x_1=clean).
        xt_B_C_T_H_W, _ = self.rectified_flow.get_interpolation(
            video_epsilon_B_C_T_H_W.to(**self.tensor_kwargs_fp32),
            video_B_C_T_H_W.to(**self.tensor_kwargs_fp32),
            sigma_B,
        )

        # FRAME_REPLACE: overwrite the conditional (obs) latent frames with the clean ground truth.
        if condition.is_video and condition.condition_video_input_mask_B_C_T_H_W is not None:
            gt = condition.gt_frames.type_as(xt_B_C_T_H_W)
            mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, xt_B_C_T_H_W.shape[1], 1, 1, 1
            ).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = gt * mask + xt_B_C_T_H_W * (1 - mask)

        # The net consumes a discrete timestep in [0, 1000] (== sigma * num_train_timesteps).
        timesteps_B_T = (video_sigma_B_1.to(**self.tensor_kwargs_fp32) * self.rectified_flow.num_train_timesteps)

        out = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T.squeeze(-1).to(**self.tensor_kwargs),
            **condition.to_dict(),
            intermediate_feature_ids=[feature_id],
        )
        crossattn_emb = out[1][0]  # (B, T*H*W, model_channels)
        return crossattn_emb

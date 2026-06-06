# mimic-video → Cosmos-Predict2.5 port (world2action)

**Fresh agent/session: read this first. It is the self-contained handoff for continuing the port.**

## Goal
Port the **mimic-video "world2action"** method onto **Cosmos-Predict2.5**. world2action extracts a
language-conditioned **visuomotor action decoder** from a *frozen* video world-model: it taps the
video DiT's intermediate hidden states and trains a small, separate `World2ActionDIT` that
cross-attends to those features to predict action chunks.

**Our setup:** a custom bimanual **"YAMS" arm** rig. The Cosmos-Predict2.5 2B Video2World **backbone
was finetuned on video we collected on the YAMS rig**. The goal now is to **train the world2action
action decoder on our own YAMS teleop data**, on top of that frozen finetuned backbone.

**Approach decision (locked):** keep mimic-video's method — *frozen backbone + separate
cross-attention decoder*. We are **NOT** using NVIDIA's "Cosmos Policy" (a different method:
full fine-tune + latent-frame injection).

## The exact model (our finetuned backbone)
From the finetune's `config.yaml`:
- **Cosmos-Predict2.5 2B Video2World, rectified-flow** (`_src/predict2/models/video2world_model_rectified_flow.py`).
- **DiT** `MinimalV1LVGDiT` (`_src/predict2/networks/minimal_v1_lvg_dit.py`), net config `cosmos_v1_2B`
  (`_src/predict2/configs/video2world/defaults/net.py`) + overrides:
  `crossattn_emb_channels=1024, use_crossattn_projection=True, crossattn_proj_in_channels=100352,
  timestep_scale=0.001, use_wan_fp32_strategy=True`. (2048 ch, 28 blocks, 16 heads, 16-ch latent, patch_spatial 2.)
- **VAE** Wan2.1 (`Wan2pt1VAEInterface`) — 16-ch latent, ×8 spatial / ×4 temporal (`get_latent_num_frames(93)=24`).
- **Text encoder** Cosmos-Reason1.1-7B (Qwen2.5-VL-7B); `full_concat` 28×3584 = **100352**, projected in-net to 1024.
- **Checkpoint** is a **LoRA finetune** (rank 32 on attn q/k/v/output_proj + mlp.layer1/2). Prefer the
  **fused** `model_ema_bf16_fused.pt` (LoRA pre-merged); else load base `model_ema_bf16.pt` + merge (PEFT).
- 720p/16fps; `state_t=24`.

## Setup
```
bash mimic_video_port/setup.sh
```
Prereqs: `hf auth login` + access to gated `nvidia/Cosmos-Reason1-7B`; AWS creds for the backbone ckpt.
Hardware: 4× RTX PRO 6000 (Blackwell, 96 GB) → `cu130` + Python 3.13 (the repo default). Hopper/Ampere:
`CUDA_EXTRA=cu128` + `uv python pin 3.10`. setup.sh runs `uv sync` (at repo root), checks HF access,
downloads the backbone checkpoint, and runs the validation smokes.

## External checkpoint redirects (important)
The repo's internal `s3://bucket/...` paths don't resolve externally (they'd register by importing
`cosmos_policy.config`, which drags in `h5py` + sim deps). Redirect just the few we need to public HF
in a small `register_external_checkpoints()` shim (see `reason1_embed_smoke.py` / `get_crossattn_emb_smoke.py`):
- Reason1 model: `CheckpointConfig(uuid="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f", hf="nvidia/Cosmos-Reason1-7B"@3210bec…).register()`
- Qwen processor: route any `…/Qwen_tokenizer/…` path → HF id `Qwen/Qwen2.5-VL-7B-Instruct`
- Wan2.1 VAE: `vae_pth="hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"`

## Validated so far (the smokes in this dir — all PASS on the node)
- `smoke_load_ckpt.py` — checkpoint loads into the 2.5 net (689/689, 0 mismatch; 560 extra = LoRA).
- `extract_features_smoke.py` — DiT forward + **native `intermediate_feature_ids=[19]`** → `(B, T·H·W, 2048)`.
- `merge_lora_smoke.py` — base + LoRA merges into the finetuned backbone (features shift ~11%).
- `vae_encode_smoke.py` — Wan2.1 VAE encodes `(1,3,9,64,64) → (1,16,3,8,8)`.
- `reason1_embed_smoke.py` — Reason1 embeds text → `(1, 512, 100352)`.
- `get_crossattn_emb_smoke.py` — **WIP capstone** (gitignored): VAE + Reason1 + frozen backbone + hook →
  the real `get_crossattn_emb`. Seed for the overlay; validate then commit.

## Key integration facts
- **Feature tap (core coupling, native in 2.5):** `out, feats = net(x, timesteps, **cond, intermediate_feature_ids=[19]); crossattn_emb = feats[0]` → `(B, T·H·W, 2048)`. Tap index **19** == mimic-video's layer-20.
- Call **`self.net(...)` directly** for extraction — the model's `denoise()` does `.float()` and can't return the `(output, features)` tuple.
- `MinimalV1LVGDiT` adds +1 condition-mask channel; `concat_padding_mask` adds another (PatchEmbed in=18). `padding_mask` is required.
- Text: conditioner `TextAttr` just renames `t5_text_embeddings` → `crossattn_emb`; feed Reason1 100352-d, net projects to 1024.
- **Precompute & cache** Reason1 embeddings per instruction (don't run the 7B every step). Reuse
  `_src/predict2/cosmos_policy/datasets/reason1_embedding_utils.py` (`generate_reason1_embeddings` / `save_reason1_embeddings`).

## What's next (the build)
1. **`get_crossattn_emb`** as a module (seed: `get_crossattn_emb_smoke.py`) + factor the redirect shim into `register_external_checkpoints()`.
2. **Port world2action** onto 2.5: `World2ActionDIT`, `World2ActionModel`, `World2ActionPipeline`.
   Reference = the original **mimic-video (cosmos-predict2) repo**:
   `model/cosmos_predict2/models/world2action_{model,dit}.py`, `pipelines/world2action.py`,
   `data/action/*` (`dataset_action.py`=MimicDataset, `chunk_reader.py`, `data_transforms.py`, `types.py`),
   `module/normalizer.py`, `configs/defaults/world2action_*`, `configs/experiment/world2action.py`, `eval/`.
3. **Data:** port the zarr `MimicDataset`; our YAMS teleop → zarr via `data_preprocessing/action/process_recordings.py`;
   swap the T5 language field for **cached Reason1 embeddings**.
4. **Trainer:** imaginaire4 `ImaginaireTrainer` (`_src/imaginaire/trainer.py`) + **DCP** checkpointer; only the
   small decoder trains (backbone frozen). Convert DCP→`.pt` with `convert_distcp_to_pt.py`.
5. **Eval:** LIBERO / Bridge (or a YAMS eval).

## Notes
- Only the small action decoder trains; the 2B backbone is frozen (forward-only; features are cacheable).
- Don't commit `checkpoints/` (large; downloaded by setup.sh).
- Hidden-state tap layer hyperparam = `intermediate_feature_ids=[19]` (≈70% depth, == mimic-video layer 20).

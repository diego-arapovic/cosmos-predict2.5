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

## How we work — collaboration model + rigour (a fresh agent MUST follow this)
- **Roles & loop.** The agent authors/edits files in the LOCAL clone (`/Users/diego/Desktop/ETHRC/mimic-video/cosmos-predict2.5/`);
  the **user** copies them to the AWS GPU node (VSCode ctrl+c/ctrl+v) and runs everything there. The agent has **no node access** —
  it relies on the user pasting output. **Every turn, state the exact files you created/edited** so the user copies the right ones.
  **No commit/push** from the agent. If you need node info, **give the user a shell command** and wait for the paste.
- **Smoke-test every step before building on it.** Validate *runtime*, not imports — each `smoke/*.py` exercises the real path on
  real/realistic inputs and prints `PASS`. Be **critical**: add a smoke only where there's genuinely new risk; don't pile on
  redundant tests. Smokes are the gates between steps.
- **Port by minimal diff.** Copy the original mimic-video file, re-point imports to 2.5 `_src`/imaginaire4 paths, change nothing
  else unless an API truly moved. Document every necessary deviation (e.g. the rectified-flow noising; see `DESIGN_RISKS.md`).
- **Overlay discipline — ZERO stock-2.5 edits.** Everything lives in `mimic_video_port/`. The training experiment plugs into
  `scripts.train` without touching stock files (`config_yams.py` + a registered experiment). External weights only via
  `register_external_checkpoints()` (never import `cosmos_policy.config` → drags in h5py/sim).
- **Fix what the outputs reveal.** When a run surfaces a real issue (dropped episodes, a dtype mismatch, …), fix the root cause
  AND make the relevant smoke representative so it can't regress. Don't paper over.
- **Keep the docs current every turn.** Update this README (state + how-we-work) and `DESIGN_RISKS.md` (graded risks) so a fresh
  session resumes with full context. Cross-session memory: `mimic-video-port-goal` records the end goal + collaboration model.
- **Storage/compute reality.** Scratch `/opt/dlami/nvme` (1.7 TB, fast, **EPHEMERAL**) is the working copy; **S3 is durable**. We
  re-derive the zarr from raw and cache only the small Reason1 + stats artifacts to S3 (`commands/prepare_data.sh`). Training is
  GPU-count-agnostic (DDP, per-rank batch; scale via `torchrun --nproc_per_node=N`).
- **Decisions.** Surface genuine trade-offs and let the user choose (cache strategy, data curation, …); otherwise pick a sensible
  default and say so. Validate shapes against the real schema: crossattn_emb **2048**, latent **16-ch**, Reason1 **100352**,
  decoder horizon/dim **16 / 14**.

## Usage (the 3-step flow)
```
bash mimic_video_port/setup.sh                       # 1. env (uv) + foundation smokes + data-prep deps
bash mimic_video_port/commands/prepare_data.sh       # 2. data: scratch zarr re-derived from raw; Reason1+stats cached to S3
bash mimic_video_port/commands/train.sh              # 3. train: resume from S3 -> torchrun -> checkpoints to S3   [being built]
```

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
**Dev/training node (confirmed 2026-06):** currently **1× RTX PRO 6000 Blackwell, 96 GB**, 32 CPU, 249 GB RAM;
driver 595.64 → `cu130` + Python 3.13 (repo default), torch 2.9.1+cu130. **Disk/GPUs are NOT fixed** — user can
size scratch to 1 TB+ and add GPUs on demand, so the training config is GPU-count-agnostic (DDP, per-rank batch;
scale = `torchrun --nproc_per_node=N`) and data/checkpoints live on a big/fast **scratch** volume (`$W2A_DATA`).
Hopper/Ampere: `CUDA_EXTRA=cu128` + `uv python pin 3.10`. setup.sh runs `uv sync` (at repo root), checks HF
access, downloads the backbone checkpoint, and runs the validation smokes.

## External checkpoint redirects (important)
The repo's internal `s3://bucket/...` paths don't resolve externally (they'd register by importing
`cosmos_policy.config`, which drags in `h5py` + sim deps). Redirect just the few we need to public HF
via `world2action/checkpoints.py::register_external_checkpoints()` (call once before building backbone/VAE/Reason1):
- Reason1 model: `CheckpointConfig(uuid="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f", hf="nvidia/Cosmos-Reason1-7B"@3210bec…).register()`
- Qwen processor: route any `…/Qwen_tokenizer/…` path → HF id `Qwen/Qwen2.5-VL-7B-Instruct`
- Wan2.1 VAE: `vae_pth="hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"`

## Validated so far (`smoke/` — all PASS on the node; rerun via `setup.sh`)
- `smoke/smoke_load_ckpt.py` — checkpoint loads into the 2.5 net (689/689, 0 mismatch; 560 extra = LoRA).
- `smoke/extract_features_smoke.py` — DiT forward + **native `intermediate_feature_ids=[19]`** → `(B, T·H·W, 2048)`.
- `smoke/merge_lora_smoke.py` — base + LoRA merges into the finetuned backbone (features shift ~11%).
- `smoke/vae_encode_smoke.py` — Wan2.1 VAE encodes `(1,3,9,64,64) → (1,16,3,8,8)`.
- `smoke/reason1_embed_smoke.py` — Reason1 embeds text → `(1, 512, 100352)`.
- `smoke/get_crossattn_emb_smoke.py` — **capstone (PASS)**: VAE + Reason1 + frozen fused backbone + hook →
  the real `get_crossattn_emb` `(B, T·H·W, 2048)`. Seed for the overlay's `get_crossattn_emb`.
- `smoke/dit_forward_smoke.py` — **ported `World2ActionDIT` runs (PASS)**: 499M params on cu130, flash-attn + rope + cross-attn to the 2048-d features → `(1, 16, 14)` (the `yams` config).
- `smoke/backbone_smoke.py` — **`FrozenVideoBackbone` path ran end-to-end on the node (finite features).**
  Real `VideoPredictionConditioner` + Wan2.1 VAE + rectified-flow interpolation + FRAME_REPLACE
  (`num_conditional_frames=1`) + native `intermediate_feature_ids=[19]` tap on a mimic-style
  `data_batch` → `crossattn_emb (1, 48, 2048)`, finite, σ=0.93. (48 = T·(H/2)·(W/2) — the tap returns
  **patchified** tokens, `patch_spatial=2`; the first run's "CHECK" was a wrong expected-token formula in
  the smoke, now fixed to match the capstone.) Re-run for a green PASS:
  `python mimic_video_port/smoke/backbone_smoke.py checkpoints/model_ema_bf16_fused.pt`
  Supersedes the capstone (which hand-built the condition). **Design risks logged in `DESIGN_RISKS.md`.**
- `smoke/model_smoke.py` — **full `World2ActionModel` train path PASS on the node.** 2.56B total = **2.06B
  frozen backbone + 499M trainable decoder**; `training_step` loss finite (3.19), `backward()` → all 509
  decoder param-groups get finite grads, **frozen backbone gets 0 grads**. The whole single-GPU path
  (backbone tap → decoder denoise → RF MSE → backward) is proven. Re-run:
  `python mimic_video_port/smoke/model_smoke.py checkpoints/model_ema_bf16_fused.pt`
- `smoke/preprocess_smoke.py` — **PASS on the node.** `process_recordings.py` on the example episode
  (`dummy_ep/episode_183420_f531fab8`) → world2action zarr: `T=211 @ 1280×720`, 0 dropped, 17.7 ms sync;
  `workspace_rgb (211,720,1280,3) uint8`, `joint_state_lowdim (211,14) float32`, instruction, timestamps ok.
  **Native cameras ≈30 fps** (211/7.01 s) → the data config will resample workspace_rgb toward ~16 fps so 93
  frames fit. Data-prep deps via `setup.sh` (uv project → **`uv pip install`**, never `pip`; **pin `zarr<3`** —
  pipeline uses the zarr v2 API): `uv pip install mcap "zarr<3" "numcodecs<0.16" imageio imageio-ffmpeg threadpoolctl`.
- `smoke/dataset_smoke.py` — **PASS on the node** (`len=211`). Ported `MimicDataset` on the converted zarr →
  `obs/workspace_rgb (3,5,720,1280)`, `action/workspace_rgb (3,88,720,1280)` (=93 px → 24 latent), normalized
  [-1,1]; `obs/lowdim_concat (1,14)`, `action/lowdim_concat (15,14)`. Validates chunk_reader + transforms on real data.
- `smoke/language_smoke.py` — **PASS on the node.** Reason1 cache built (`(512,100352)` per instruction) +
  `Reason1EmbeddingLookup` → `obs/language_embedding (1,512,100352) float16` in the dataset (mmap'd cache, not
  per-episode zarr). **Data path now complete: every model input key validated on real data.**
  Run: `python mimic_video_port/smoke/language_smoke.py /tmp/w2a_preprocess_smoke`
- `smoke/stats_smoke.py` — **PASS on the node.** `precompute_stats.build_stats` → `normalizer_stats.pt` (stats for
  `{obs,action}/joint_state_lowdim`), then `build_from_stats` → normalizer with `{obs,action}/lowdim_concat`,
  normalizes a dummy chunk. Exactly what `on_train_start` loads. **The full data + normalizer path is now validated.**
  Run: `python mimic_video_port/smoke/stats_smoke.py /tmp/w2a_preprocess_smoke`
- `smoke/train_config_smoke.py` — **PASS on the node (the training-wiring gate).** Mirrors `scripts.train` (no DDP/loop):
  `make_config` → `override(experiment=yams_smoke)` → `instantiate(config.model)` → `on_train_start` →
  `instantiate(dataloader_train)` → one `training_step` + backward, on the **real** converted data: 2.56B model + normalizer
  built, loss finite (22.3), decoder grads 509/509, backbone 0 grads. (Surfaced + fixed a fp16(language)/bf16(net)
  `crossattn_proj` mismatch — backbone casts to the model dtype; Reason1 cache now stored bf16; smokes use fp16 dummies.)
  Run: `python mimic_video_port/smoke/train_config_smoke.py yams_smoke`

## Key integration facts
- **Feature tap (core coupling, native in 2.5):** `out, feats = net(x, timesteps, **cond, intermediate_feature_ids=[19]); crossattn_emb = feats[0]` → `(B, T·H·W, 2048)` (already flattened — no reshape). Tap index **19** == mimic-video's layer-20.
- Call **`self.net(...)` directly** for extraction — the model's `denoise()` does `.float()` and can't return the `(output, features)` tuple. The DiT `forward` has `**kwargs`, so passing the *full* `condition.to_dict()` (incl. `gt_frames`, `use_video_condition`, `num_conditional_frames_B`) + `intermediate_feature_ids` is exactly what the real `Video2WorldModelRectifiedFlow.denoise` does — safe.
- **The 2.5 backbone is rectified-flow, so the noising convention CHANGED vs mimic-video** (this is the one necessary, non-minimal-diff deviation). Pure RF: `xt = sigma·eps + (1-sigma)·x0` (sigma∈[0,1]) and the net receives a *discrete* timestep `= sigma·1000` (∈[0,1000]) — **not** mimic-video's additive `x + eps·sigma` (EDM-style) + `RectifiedFlowScaling`. `FrozenVideoBackbone.extract_crossattn_emb` uses `RectifiedFlow.get_interpolation`; sigma is sampled `logitnormal` + shift-warped (`shift=5`) via `draw_video_sigma`. The decoder is conditioned on `context_timesteps_B_1 = sigma` (∈[0,1]).
- **FRAME_REPLACE conditioning** (from `Video2WorldModelRectifiedFlow.denoise`): the first `num_conditional_frames` latent frames of `xt` are overwritten with the clean obs latent (`gt_frames * mask`). world2action sets `num_conditional_frames = get_latent_num_frames(obs_pixel_frames)` so obs is the clean context, the action-future frames are noised.
- **Backbone build is piecemeal (capstone-style), NOT the heavy `Video2WorldModelRectifiedFlow`/Hydra** — avoids EMA/FSDP/h5py/sim. Net overrides pinned to the finetune (`..._RECTIFIED_FLOW`): `crossattn_emb_channels=1024, use_crossattn_projection, crossattn_proj_in_channels=100352, timestep_scale=0.001, use_wan_fp32_strategy, rope_enable_fps_modulation=False, rope_{h,w}_extrapolation_ratio=3.0, rope_t=1.0`; VAE `temporal_window=16`; conditioner dropout forced to 0 for deterministic extraction.
- `MinimalV1LVGDiT` adds +1 condition-mask channel; `concat_padding_mask` adds another (PatchEmbed in=18). `padding_mask` is required.
- Text: conditioner `TextAttr` just renames `t5_text_embeddings` → `crossattn_emb`; feed Reason1 100352-d, net projects to 1024.
- **Precompute & cache** Reason1 embeddings per instruction (don't run the 7B every step). Reuse
  `_src/predict2/cosmos_policy/datasets/reason1_embedding_utils.py` (`generate_reason1_embeddings` / `save_reason1_embeddings`).
- **`EMAConfig` gotcha:** the decoder EMA needs `enabled`/`rate`/`iteration_shift` (power-EMA, like the
  *predict2* `EMAConfig` at `text2world_model.py:73`). `_src/imaginaire/config.py::EMAConfig` is a *different*
  class (`enabled`/`beta` only) — do **not** use it. `config.py` defines our own 3-field `EMAConfig`; `model.py` imports it from `.config`.
- **Grad clipping:** `model.clip_grad_norm_` uses stdlib `torch.nn.utils.clip_grad.clip_grad_norm_` (what 2.5's
  own `Text2WorldModelRectifiedFlow` uses under FSDP), not mimic-video's custom `utils.torch_future` wrapper.
- **Trainer hooks (2.5 `ImaginaireTrainer`, `_src/imaginaire/trainer.py:219`):** calls `on_train_start(memory_format)`
  — **1 arg, no dataset stats** (mimic-video passed `dataset_stats`/`stats_id` via introspection) — then
  `init_optimizer_scheduler`, `training_step`, `on_before_zero_grad`, and `state_dict/load_state_dict` via the **DCP**
  checkpointer. So world2action loads normalizer stats from a **precomputed file** (`precompute_stats.py` →
  `config.normalizer_stats_path`), and grad-clip is a `callbacks.grad_clip` entry (the trainer never calls
  `model.clip_grad_norm_`). Entrypoint: `python -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=<name>`.
- **Callbacks assume a video2world model.** The `basic` callbacks (group package `trainer.callbacks`) include
  `compile_tokenizer` which does `model.tokenizer` → `World2ActionModel` exposes a `.tokenizer` property (the frozen
  backbone VAE) so it works (and `compile`s the VAE encode = perf win). `.tokenizer`/EMA-sampling callbacks that aren't
  in `basic` (`val_loss_computation`, `every_n_draw_sample`) don't run (validation off). The trainer loop + callbacks +
  DDP + DCP are first exercised by `commands/train.sh` — the config smoke bypasses them (`DESIGN_RISKS.md` #21).

## What's next (the build)
**DONE (runtime-validated on the node unless noted):** `checkpoints.py`, `dit.py` (fwd PASS),
`normalizer.py` (imports the canonical `data/action/types.py`), `config.py` (+ own `EMAConfig`), `base.py`, `beta_scheduler.py`, `pipeline.py` (`World2ActionPipeline`),
`backbone.py` (`FrozenVideoBackbone` — fused + base+LoRA PASS), `model.py` (`World2ActionModel` train step+backward PASS),
`data/action/{types,utils,interpolate,convert_pose_repr,chunk_reader,data_transforms,dataset_action}.py` (literal copy +
re-point), `data/yams_config.py`, `data_preprocessing/{process_recordings,precompute_reason1,precompute_stats}.py`.
**All data smokes PASS:** `dataset_smoke`, `language_smoke`, `stats_smoke` (+ `preprocess_smoke`).
Remaining, in dependency order:
0. **`smoke/model_smoke.py` PASS on the node** — `training_step` + backward proven (decoder trains, backbone
   frozen). `model.py` was ported by minimal diff with one structural swap: `Video2WorldPipeline` →
   **`FrozenVideoBackbone`** (calls map ~1:1, see the file header). `validation_step` keeps the gt-video MSE
   sweep but **defers the `generate_video`/genvid sweep to eval (step 3)**.
1. **Data → `world2action/data/` — DONE (every model input key validated on real YAMS data).** The full path:
   `data_preprocessing/process_recordings.py` (raw bimanual mcap+mp4 → 720p zarr, **camera_top only by default**
   — world2action uses only camera_top; stable names by source id; `--cameras all` keeps wrist), the
   `data/action/` library (literal copy + relative-import re-point), the
   **Reason1 language cache** (`precompute_reason1.py` + `Reason1EmbeddingLookup`, mmap'd, keyed by instruction), and
   the offline **normalizer stats** (`precompute_stats.py`). **Single source of truth for the data spec:**
   `world2action/data/yams_config.py` (policy_io: obs ws=5 / action ws=88 → 93 px → 24 latent; 16 fps; joints 1/15
   → 14-d; concat_groups; transforms). **Confirmed:** camera_top→`workspace_rgb`, 720p, action = future YAM joint states.
   Smokes PASS: `preprocess`, `dataset`, `language`, `stats`.
   **Storage model — scratch is EPHEMERAL, S3 is durable.** Node scratch = **`/opt/dlami/nvme`** (1.7 TB NVMe, fast, but
   **wiped on instance stop/terminate**). The processed zarr is large (**~295 GB** workspace-only — decoded 720p frames)
   but cheap to re-derive from raw (~3 min CPU), so we do **NOT** cache the zarr to S3 — we re-derive it. We cache to S3
   only the small, expensive-to-make artifacts: the **Reason1 cache** (needs the 7B GPU) + **stats**.
   **One idempotent script:** `bash mimic_video_port/commands/prepare_data.sh` — (1) ensure raw on scratch (download from
   raw S3 if missing), (2) raw → zarr (stable names; skips already-converted), (3) Reason1 cache (reuse the S3 copy if
   local is missing → skips the 7B), (4) stats; uploads Reason1+stats to `s3://…/world2action/artifacts/`. `UPDATE=1`
   forces a re-scan for newly-collected episodes (only new ones convert; Reason1 only encodes new instructions). Defaults:
   `W2A_DATA=/opt/dlami/nvme/world2action`, `CAMERAS=workspace` (raw S3 keeps all cams → `CAMERAS=all` for future multi-view).
   **Incremental — add new episodes later with `UPDATE=1 bash mimic_video_port/commands/prepare_data.sh`:** zarrs are named
   by source episode id (stable), so `process_recordings` skips already-converted episodes and converts only new ones;
   `precompute_reason1` only encodes new instructions (won't even load the 7B if none); stats are recomputed (cheap);
   S3 sync uploads only deltas. So adding data ≈ "fetch + process only the new", never redo. **Camera requirement is
   per stored camera** — an episode missing an *unused* wrist cam is NOT skipped (earlier bug dropped ~11% of episodes).
2. **Configs + trainer:** **IN PROGRESS.**
   - **DONE:** `world2action/data/yams_config.py` (single source of truth for the data spec — policy_io, transforms,
     concat_groups, 93-px/24-latent + 16 fps numbers); `data_preprocessing/precompute_stats.py` + `model.on_train_start`
     1-arg loading `config.normalizer_stats_path` (the 2.5-trainer divergence, risk #17); `stats_smoke.py`.
   - **BUILT (no stock-file edits; gate = `smoke/train_config_smoke.py`):** the base video2world `make_config` tolerates a
     custom model — the trainer only `instantiate`s `config.model` + the dataloaders; `net`/`conditioner`/`tokenizer`
     defaults are ignored (we override them to None). (i) `mimic_video_port/world2action/configs/yams_experiment.py` registers
     `yams_smoke` + `yams` via `cs.store(group="experiment", package="_global_", ...)`; (ii) `mimic_video_port/config_yams.py`
     re-exports stock `make_config` and imports the experiment module (so `cs.store` runs before compose). The experiment
     overrides `/model` = `L(World2ActionModel)(config=L(World2ActionModelConfig)(...))` (`normalizer_stats_path`,
     `video_dit_path`, `data_config`=yams policy_io, the `yams` decoder net as a **raw LazyDict — keep `_recursive_=False`
     so the net/pipe sub-configs aren't pre-instantiated**, since `pipeline.from_config` does `instantiate(net)`),
     `dataloader_train/val` = `L(DataLoader)(dataset=L(MimicDataset)(**yams_config.dataset_kwargs(reason1_cache)))`
     (per-rank `batch_size`, `sampler=None` → DDP-handled), `defaults: [{override /ckpt_type: dcp}, ...]`, `trainer`
     (`distributed_parallelism="ddp"`, lr 1e-4, bf16, grad_clip callback), `job.path_local=$W2A_DATA/checkpoints/world2action`.
     **Launch (GPU-agnostic):** `torchrun --nproc_per_node=N -m scripts.train --config=mimic_video_port/config_yams.py -- experiment=yams_smoke`.
     **Config gate PASS:** `train_config_smoke.py yams_smoke` (compose+model+dataloader+one step). **`commands/train.sh` BUILT:**
     prepare_data → resume latest checkpoint from S3 → `torchrun` → mirror checkpoints to S3 (periodic + on exit;
     `IMAGINAIRE_OUTPUT_ROOT=$W2A_DATA/runs` so they land on scratch). **NEXT GATE — the real smoke-sized run (DDP+DCP+loop):**
     `bash mimic_video_port/commands/train.sh` (EXP=yams_smoke, 1 GPU, 3 iters). If green → `EXP=yams NGPU=N bash …/train.sh`.
     ⚠️ **Throughput (risk #8):** the per-step frozen-backbone forward at 720p (≈86k tokens) is the bottleneck; scales ~linearly
     with GPUs (data-parallel). Measure per-step time on the smoke; a long single-GPU run likely wants feature caching or more GPUs.
3. **Eval + real-world deployment (an explicit end goal):** a **live YAMS rollout loop** — read camera_top
   (720p) + YAM proprio + the task instruction, run the frozen backbone tap + `World2ActionPipeline.__call__`
   (the decoder sampler) to predict an action chunk, and command the bimanual arms; loop. The mimic-video
   LIBERO/Bridge sim eval is only a reference for the inference wiring. Inference reuses the **same** Reason1
   instruction cache + 720p prep as training so features stay in-distribution.

## Notes
- Only the small action decoder trains; the 2B backbone is frozen (forward-only; features are cacheable).
- Don't commit `checkpoints/` (large; downloaded by setup.sh).
- Hidden-state tap layer hyperparam = `intermediate_feature_ids=[19]` (≈70% depth, == mimic-video layer 20).
- **One `types.py`:** the canonical enums (`NormalizationType`, `ObsType`, `LieRepr`, ...) live ONLY at
  `world2action/data/action/types.py`; `normalizer.py` imports from there. Do NOT recreate `world2action/types.py` —
  a second copy is a *distinct* `NormalizationType` enum, so `mode is NormalizationType.NONE` silently fails
  (the `obs/workspace_rgb` KeyError in `build_from_stats`). (Deleted on 2026-06; `rm` it on the node if present.)

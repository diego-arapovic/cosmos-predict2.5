"""Registered YAMS world2action training experiments for `scripts.train` (no stock-2.5 edits).

Imported by `mimic_video_port/config_yams.py` so `cs.store(...)` runs before Hydra composes. Launch:
  torchrun --nproc_per_node=N -m scripts.train --config=mimic_video_port/config_yams.py -- experiment=yams_smoke

Experiments:
  - yams_smoke : 3-iter gate, no validation, no W&B. The fast "does the whole pipeline run" check.
  - yams_medium: short run (default 1500 iters) WITH W&B curves + periodic held-out eval (val loss +
                 sampled-action MSE vs a hold-last baseline). The "watch it learn a good policy" run.
  - yams       : the full run (W&B + eval on, longer validation cadence).

Design notes:
  - The base video2world `make_config` tolerates a custom model: the trainer only `instantiate`s
    `config.model` + the dataloaders. We override `/model` and the video2world-only groups
    (`/net /conditioner /tokenizer /ema`) to None so their `model.config.*` injections don't pollute
    our `World2ActionModelConfig`, then set `model` directly in `_self_`.
  - `_recursive_=False` on the model: `instantiate(config.model)` builds `World2ActionModel(config=<DictConfig>)`
    WITHOUT pre-instantiating the nested decoder net -- `World2ActionPipeline.from_config` does
    `instantiate(net)` itself. The model reads config via attribute access (works on the DictConfig).
  - The dataset is built by a factory (`yams_config.make_dataset`) so the enum-valued data spec stays in
    Python and never enters the Hydra config (avoids enum->string coercion).
  - `checkpoint.load_path=""` so the entrypoint does `instantiate(config.model)` (a `.pt` load_path would
    trigger a video2world-specific consolidated loader). DCP auto-resumes from the job dir.
  - W&B: experiments that log curves override the `/callbacks` group to None and set `trainer.callbacks`
    explicitly to BASIC_CALLBACKS + our World2ActionWandb (the stock WandbCallback assumes a diffusion
    `edm_loss` and can't log the action eval). Mode is `job.wandb_mode` (WANDB_MODE env; commands/train.sh
    auto-picks online/offline from credentials). yams_smoke keeps the default `basic` callbacks (no W&B).
  - Paths are env-overridable so the same config runs on scratch ($W2A_DATA) or /tmp (the smoke).
"""
import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig

from mimic_video_port.world2action.callbacks import World2ActionWandb
from mimic_video_port.world2action.config import EMAConfig, SchedulerConfig, World2ActionPipelineConfig
from mimic_video_port.world2action.data import yams_config
from mimic_video_port.world2action.dit import World2ActionDIT
from mimic_video_port.world2action.model import World2ActionModel, World2ActionModelConfig

W2A_DATA = os.environ.get("W2A_DATA", "/opt/dlami/nvme/world2action")
CONV = os.environ.get("W2A_CONVERTED", f"{W2A_DATA}/teleop_converted")
REASON1_CACHE = os.environ.get("W2A_REASON1_CACHE", f"{CONV}/reason1_embeddings.pt")
STATS = os.environ.get("W2A_STATS", f"{CONV}/normalizer_stats.pt")
VIDEO_DIT = os.environ.get("W2A_VIDEO_DIT", "checkpoints/model_ema_bf16_fused.pt")  # frozen backbone (fused EMA)

# Optional task subset: W2A_INSTRUCTIONS="instr a|instr b" trains only those episodes (exact match). Empty = all.
INSTRUCTIONS = [s for s in os.environ.get("W2A_INSTRUCTIONS", "").split("|") if s.strip()] or None
# W&B mode: offline by default (never blocks on login); commands/train.sh sets WANDB_MODE=online when
# credentials exist so you get live curves. Offline runs sync later with `wandb sync <job_dir>`.
WANDB_MODE = os.environ.get("WANDB_MODE", "offline")


def _model(val_action_sigmas=()) -> LazyDict:
    net = L(World2ActionDIT)(  # the action decoder (instantiated by pipeline.from_config, not here)
        max_horizon=16,
        in_channels=14,
        out_channels=14,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,  # == frozen backbone model_channels (the feature-tap width)
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=L(SACConfig)(mode="none", every_n_blocks=1),
    )
    return L(World2ActionModel)(
        config=L(World2ActionModelConfig)(
            train_architecture="base",
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            precision="bfloat16",
            loss_reduce="mean",
            loss_scale=10.0,
            ema=L(EMAConfig)(enabled=False),
            action_dit_path="",  # decoder trained from scratch
            video_dit_path=VIDEO_DIT,  # frozen backbone
            pipe_config=L(World2ActionPipelineConfig)(
                precision="bfloat16",
                scheduler=L(SchedulerConfig)(alpha=1.0, beta=1.0, num_denoising_steps=10),  # =mimic-video (uniform decoder t; was 1.5)
                net=net,
                ema=L(EMAConfig)(enabled=False),
                xattn_layer_idx=19,  # == mimic-video's layer-20 tap
            ),
            fsdp_shard_size=0,  # single-GPU DDP; only the decoder trains, backbone replicated + frozen
            video_state_t=24,
            video_shift=5,
            video_fps=16.0,
            normalizer_stats_path=STATS,
            val_action_sigmas=list(val_action_sigmas),  # [] -> only cheap val loss; else the action-MSE eval
        ),
        _recursive_=False,
    )


def _dataloader(train: bool, batch_size: int, num_workers: int, num_val_episodes: int = 1, shuffle: bool = False) -> LazyDict:
    return L(DataLoader)(
        dataset=L(yams_config.make_dataset)(
            data_dir=CONV, reason1_cache_path=REASON1_CACHE, train=train, num_val_episodes=num_val_episodes,
            instruction_filter=INSTRUCTIONS,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,  # train: mix chunks across episodes (else SGD sees one episode at a time);
        sampler=None,     # val: spread the capped max_val_iter sample across episodes (representative baseline)
    )


def _experiment(
    name,
    *,
    max_iter,
    batch_size,
    save_iter,
    logging_iter,
    num_workers,
    run_validation,
    validation_iter=0,
    run_validation_on_start=False,
    max_val_iter=0,
    num_val_episodes=1,
    val_action_sigmas=(),
    warm_up_steps=1_000,
    use_wandb=False,
    wandb_mode="offline",
) -> LazyDict:
    defaults = [
        {"override /model": None},
        {"override /net": None},
        {"override /conditioner": None},
        {"override /tokenizer": None},
        {"override /ema": None},
        {"override /data_train": None},
        {"override /data_val": None},
        {"override /ckpt_type": "dcp"},
    ]
    if use_wandb:  # merge the stock `basic` callbacks with our W&B/eval logger group (registered below)
        defaults.append({"override /callbacks": ["basic", "w2a_wandb"]})
    defaults.append("_self_")

    cfg = dict(
        defaults=defaults,
        model=_model(val_action_sigmas=val_action_sigmas),
        # Both splits MUST use the same num_val_episodes (same seed) so they're complementary/disjoint:
        # train = ~val_mask, val = val_mask. (If train used the default 1, val's other 7 episodes would leak
        # into training -> contaminated val metrics once shuffle is on.)
        dataloader_train=_dataloader(True, batch_size, num_workers, num_val_episodes=num_val_episodes, shuffle=True),
        dataloader_val=_dataloader(False, 1, min(2, num_workers), num_val_episodes=num_val_episodes, shuffle=True),
        optimizer=dict(lr=1.0e-4),
        scheduler=dict(f_max=[1.0], f_min=[0.2], warm_up_steps=[warm_up_steps], cycle_lengths=[500_000]),
        trainer=dict(
            distributed_parallelism="ddp",
            grad_accum_iter=1,
            max_iter=max_iter,
            logging_iter=logging_iter,
            validation_iter=max(1, validation_iter or max_iter),
            run_validation=run_validation,
            run_validation_on_start=run_validation_on_start,
            max_val_iter=(max_val_iter or None),  # None = whole val set; set small (action eval is costly)
        ),
        checkpoint=dict(save_iter=save_iter, load_path="", load_training_state=False, strict_resume=False),
        job=dict(project="world2action", group="yams", name=name, wandb_mode=wandb_mode),
        model_parallel=dict(context_parallel_size=1),
        upload_reproducible_setup=False,
    )
    return LazyDict(cfg, flags={"allow_objects": True})


YAMS_SMOKE = _experiment(
    "yams_smoke", max_iter=3, batch_size=1, save_iter=3, logging_iter=1, num_workers=2, run_validation=False
)
# Short run to WATCH the decoder learn: W&B curves + periodic held-out eval (val loss, sampled-action MSE
# vs a hold-last baseline + echo/normalized metrics). ~1500 iters; on 1 GPU at ~10 s/iter that's ~4 h.
# Eval is cheap: every 500 iters, 8 val chunks, 1 sigma => ~2-3 min (~3% of wall-clock). Override via OVERRIDES.
YAMS_MEDIUM = _experiment(
    "yams_medium",
    max_iter=1500,
    batch_size=1,
    save_iter=500,
    logging_iter=10,
    num_workers=4,
    run_validation=True,
    validation_iter=500,
    run_validation_on_start=True,  # iter-0 point = baseline-level action MSE (untrained decoder)
    max_val_iter=8,                # 8 val chunks (shuffled => spread across the val episodes)
    num_val_episodes=8,            # enough held-out episodes that the baseline isn't a static-sample fluke
    val_action_sigmas=(0.9, 1.0),  # eval where it's TRAINED+DEPLOYED (high noise / no GT-future leakage); 0.5 leaks GT
    warm_up_steps=200,             # short run -> short LR warmup so the loss-decrease is visible early
    use_wandb=True,
    wandb_mode=WANDB_MODE,
)
YAMS = _experiment(
    "yams",
    max_iter=500_000,
    batch_size=2,
    save_iter=1_000,
    logging_iter=50,
    num_workers=8,
    run_validation=True,
    validation_iter=2_000,         # eval << 1% of wall-clock at this cadence
    max_val_iter=8,
    num_val_episodes=8,
    val_action_sigmas=(0.9, 1.0),  # eval where it's TRAINED+DEPLOYED (high noise / no GT-future leakage)
    use_wandb=True,
    wandb_mode=WANDB_MODE,
)

cs = ConfigStore.instance()
# Our W&B + eval logger as a `callbacks` group, so experiments select it with the canonical merge syntax
# `override /callbacks: ["basic", "w2a_wandb"]` (same mechanism stock experiments use to add "wandb").
cs.store(group="callbacks", package="trainer.callbacks", name="w2a_wandb", node=dict(w2a_wandb=L(World2ActionWandb)()))
cs.store(group="experiment", package="_global_", name="yams_smoke", node=YAMS_SMOKE)
cs.store(group="experiment", package="_global_", name="yams_medium", node=YAMS_MEDIUM)
cs.store(group="experiment", package="_global_", name="yams", node=YAMS)

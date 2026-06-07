"""Registered YAMS world2action training experiments for `scripts.train` (no stock-2.5 edits).

Imported by `mimic_video_port/config_yams.py` so `cs.store(...)` runs before Hydra composes. Launch:
  torchrun --nproc_per_node=N -m scripts.train --config=mimic_video_port/config_yams.py -- experiment=yams_smoke

Design notes:
  - The base video2world `make_config` tolerates a custom model: the trainer only `instantiate`s
    `config.model` + the dataloaders. We override `/model` and the video2world-only groups
    (`/net /conditioner /tokenizer /ema`) to None so their `model.config.*` injections don't pollute
    our `World2ActionModelConfig`, then set `model` directly in `_self_`.
  - `_recursive_=False` on the model: `instantiate(config.model)` builds `World2ActionModel(config=<DictConfig>)`
    WITHOUT pre-instantiating the nested decoder net — `World2ActionPipeline.from_config` does
    `instantiate(net)` itself. The model reads config via attribute access (works on the DictConfig).
  - The dataset is built by a factory (`yams_config.make_dataset`) so the enum-valued data spec stays in
    Python and never enters the Hydra config (avoids enum->string coercion).
  - `checkpoint.load_path=""` so the entrypoint does `instantiate(config.model)` (a `.pt` load_path would
    trigger a video2world-specific consolidated loader). DCP auto-resumes from the job dir.
  - Paths are env-overridable so the same config runs on scratch ($W2A_DATA) or /tmp (the smoke).
"""
import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig

from mimic_video_port.world2action.config import EMAConfig, SchedulerConfig, World2ActionPipelineConfig
from mimic_video_port.world2action.data import yams_config
from mimic_video_port.world2action.dit import World2ActionDIT
from mimic_video_port.world2action.model import World2ActionModel, World2ActionModelConfig

W2A_DATA = os.environ.get("W2A_DATA", "/opt/dlami/nvme/world2action")
CONV = os.environ.get("W2A_CONVERTED", f"{W2A_DATA}/teleop_converted")
REASON1_CACHE = os.environ.get("W2A_REASON1_CACHE", f"{CONV}/reason1_embeddings.pt")
STATS = os.environ.get("W2A_STATS", f"{CONV}/normalizer_stats.pt")
VIDEO_DIT = os.environ.get("W2A_VIDEO_DIT", "checkpoints/model_ema_bf16_fused.pt")  # frozen backbone (fused EMA)


def _model() -> LazyDict:
    net = L(World2ActionDIT)(  # the `yams` action decoder (instantiated by pipeline.from_config, not here)
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
                scheduler=L(SchedulerConfig)(alpha=1.5, beta=1.0, num_denoising_steps=10),
                net=net,
                ema=L(EMAConfig)(enabled=False),
                xattn_layer_idx=19,  # == mimic-video's layer-20 tap
            ),
            fsdp_shard_size=0,  # single-GPU DDP; only the decoder trains, backbone replicated + frozen
            video_state_t=24,
            video_shift=5,
            video_fps=16.0,
            normalizer_stats_path=STATS,
        ),
        _recursive_=False,
    )


def _dataloader(train: bool, batch_size: int, num_workers: int) -> LazyDict:
    return L(DataLoader)(
        dataset=L(yams_config.make_dataset)(
            data_dir=CONV, reason1_cache_path=REASON1_CACHE, train=train, num_val_episodes=1
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        sampler=None,  # DDP-aware sampling handled by the trainer
    )


def _experiment(name, *, max_iter, batch_size, save_iter, logging_iter, num_workers, run_validation) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                {"override /model": None},
                {"override /net": None},
                {"override /conditioner": None},
                {"override /tokenizer": None},
                {"override /ema": None},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            model=_model(),
            dataloader_train=_dataloader(True, batch_size, num_workers),
            dataloader_val=_dataloader(False, 1, min(2, num_workers)),
            optimizer=dict(lr=1.0e-4),
            scheduler=dict(f_max=[1.0], f_min=[0.2], warm_up_steps=[1_000], cycle_lengths=[500_000]),
            trainer=dict(
                distributed_parallelism="ddp",
                grad_accum_iter=1,
                max_iter=max_iter,
                logging_iter=logging_iter,
                validation_iter=max(1, max_iter),
                run_validation=run_validation,
            ),
            checkpoint=dict(save_iter=save_iter, load_path="", load_training_state=False, strict_resume=False),
            job=dict(project="world2action", group="yams", name=name),
            model_parallel=dict(context_parallel_size=1),
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )


YAMS_SMOKE = _experiment(
    "yams_smoke", max_iter=3, batch_size=1, save_iter=3, logging_iter=1, num_workers=2, run_validation=False
)
YAMS = _experiment(
    "yams", max_iter=500_000, batch_size=2, save_iter=1_000, logging_iter=50, num_workers=8, run_validation=False
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="yams_smoke", node=YAMS_SMOKE)
cs.store(group="experiment", package="_global_", name="yams", node=YAMS)

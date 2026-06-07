"""Canonical world2action data spec for the YAMS rig — single source of truth.

Used by both `data_preprocessing/precompute_stats.py` (offline normalizer stats) and the training
experiment config (the dataloader). Encodes the 2.5-adapted numbers:

  - workspace_rgb (camera_top, 720p): obs 5 + action 88 = **93 px frames -> 24 latent** for the backbone;
    obs 5 px -> get_latent_num_frames(5)=2 conditional latent frames (matches the finetune's
    conditional_frames_probs {0,1,2}).
  - joint_state_lowdim (14-d = 7 left + 7 right, future YAM): obs 1 + action 15, VARIANCE-normalized.
  - language: read the `language_instruction` string, swapped to `obs/language_embedding (1,512,100352)`
    by Reason1EmbeddingLookup (cache built by precompute_reason1.py).
  - ~16 fps resample (cameras are ~30 fps); action shifted +0.2 s (latency compensation, as mimic-video).
"""
from .action.types import LieRepr, NormalizationType, ObsType

FPS = 16
ACTION_SHIFT = 0.2  # seconds; predict actions starting slightly ahead (matches mimic-video)
RESIZE_HW = [720, 1280]  # [H, W] — 720p; passthrough since process_recordings already wrote 720p
TIMESTEP_ANCHOR = "workspace_rgb"

OBS_WS_HORIZON = 5
ACTION_WS_HORIZON = 88  # 5 + 88 = 93 px frames -> tokenizer.get_latent_num_frames(93) = 24 (= backbone state_t)
OBS_JOINT_HORIZON = 1
ACTION_JOINT_HORIZON = 15  # decoder max_horizon 16 = obs 1 + action 15

DATA_COMPONENTS = {
    "workspace_rgb": {"obs_type": ObsType.RGB, "repr": None},
    "joint_state_lowdim": {"obs_type": ObsType.JOINT_POS, "repr": LieRepr.ABSOLUTE},
    "language_instruction": {"obs_type": ObsType.LANGUAGE, "repr": None},
}

CONCAT_GROUPS = {
    "action/lowdim_concat": ["action/joint_state_lowdim"],
    "obs/lowdim_concat": ["obs/joint_state_lowdim"],
}


def _spec(horizon, norm, *, shift, target_repr=None, freq=FPS):
    return {
        "horizon": horizon,
        "target_frequency": freq,
        "shift_right_by": shift,
        "normalization_type": norm,
        "target_repr": target_repr,
    }


def policy_io() -> dict:
    return {
        "obs": {
            "workspace_rgb": _spec(OBS_WS_HORIZON, NormalizationType.NONE, shift=0.0),
            "joint_state_lowdim": _spec(OBS_JOINT_HORIZON, NormalizationType.VARIANCE, shift=0.0, target_repr=LieRepr.ABSOLUTE),
            "language_instruction": {
                "horizon": 1,
                "target_frequency": None,
                "shift_right_by": 0.0,
                "normalization_type": NormalizationType.NONE,
                "target_repr": None,
            },
        },
        "action": {
            "workspace_rgb": _spec(ACTION_WS_HORIZON, NormalizationType.NONE, shift=ACTION_SHIFT),
            "joint_state_lowdim": _spec(ACTION_JOINT_HORIZON, NormalizationType.VARIANCE, shift=ACTION_SHIFT, target_repr=LieRepr.ABSOLUTE),
        },
    }


def data_transforms(reason1_cache_path: str) -> list[dict]:
    return [
        {"name": "CosmosProcessImage", "targets": ["workspace_rgb"], "resize_sizes": RESIZE_HW},
        {"name": "Flatten", "targets": ["lowdim"]},
        {"name": "Concat", "targets": ["action/joint_state_lowdim"], "out_key": "action/lowdim_concat"},
        {"name": "Concat", "targets": ["obs/joint_state_lowdim"], "out_key": "obs/lowdim_concat"},
        {
            "name": "Reason1EmbeddingLookup",
            "targets": ["language_instruction"],
            "cache_path": reason1_cache_path,
            "out_key": "obs/language_embedding",
        },
    ]


def dataset_kwargs(reason1_cache_path: str) -> dict:
    """MimicDataset kwargs (minus data_dir / train / num_val_episodes / seed)."""
    return dict(
        timestep_anchor=TIMESTEP_ANCHOR,
        data_components=DATA_COMPONENTS,
        data_transforms=data_transforms(reason1_cache_path),
        policy_io=policy_io(),
        source_component_names={},
        should_include_padded_tails=True,
    )


def normalization_types() -> dict:
    from .action.utils import extract_normalization_types

    return extract_normalization_types(policy_io())


def make_dataset(
    data_dir: str,
    reason1_cache_path: str,
    *,
    train: bool = True,
    num_val_episodes: int = 1,
    seed: int = 0,
    instruction_filter: list[str] | None = None,
):
    """Build the YAMS world2action MimicDataset. Used as the training config's dataset factory so the
    enum-valued data spec (ObsType/NormalizationType/LieRepr) stays in Python and never enters the Hydra
    config (which would risk enum->string coercion). The config passes only paths/bools/ints/strings.

    instruction_filter: if given, train on only the episodes whose `instruction` is in this list (e.g.
    ["push the box to the right with the right arm"]) -- lets us train per-task on a data subset.
    """
    from .action.dataset_action import MimicDataset

    return MimicDataset(
        data_dir=data_dir,
        **dataset_kwargs(reason1_cache_path),
        seed=seed,
        num_val_episodes=num_val_episodes,
        train=train,
        instruction_filter=instruction_filter,
    )

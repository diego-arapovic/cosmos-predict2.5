#!/usr/bin/env python
"""Validate the ported world2action MimicDataset on a real converted episode (video + lowdim path).

    python mimic_video_port/smoke/dataset_smoke.py [/tmp/w2a_preprocess_smoke]

Run preprocess_smoke.py first (it writes the zarr this reads). Builds MimicDataset with a faithful
lerobot_bi_yams-style config (workspace_rgb obs 5 + action 88 = 93 px frames -> 24 latent for the
backbone; joint_state_lowdim obs 1 + action 15 -> 14-dim) and pulls one sample, checking the keys +
shapes the model consumes:
  obs/workspace_rgb     (3, 5, 720, 1280)  float32 in [-1, 1]
  action/workspace_rgb  (3, 88, 720, 1280) float32 in [-1, 1]
  obs/lowdim_concat     (1, 14)            float32
  action/lowdim_concat  (15, 14)           float32
Language (obs/language_embedding) is added next via the Reason1 instruction cache (DESIGN_RISKS #16).
This exercises chunk_reader + data_transforms (CosmosProcessImage/Flatten/Concat) + dataset_action.
"""
import copy
import sys

import numpy as np

from mimic_video_port.world2action.data.action.dataset_action import MimicDataset
from mimic_video_port.world2action.data.action.types import LieRepr, NormalizationType, ObsType

FPS = 16  # resample the ~30 fps cameras toward the backbone's 16 fps


def _spec(horizon, norm, *, shift=0.0, target_repr=None):
    return {
        "horizon": horizon,
        "target_frequency": FPS,
        "shift_right_by": shift,
        "normalization_type": norm,
        "target_repr": target_repr,
    }


def main(data_dir):
    data_components = {
        "workspace_rgb": {"obs_type": ObsType.RGB, "repr": None},
        "joint_state_lowdim": {"obs_type": ObsType.JOINT_POS, "repr": LieRepr.ABSOLUTE},
    }
    policy_io = {
        "obs": {
            "workspace_rgb": _spec(5, NormalizationType.NONE),
            "joint_state_lowdim": _spec(1, NormalizationType.VARIANCE, target_repr=LieRepr.ABSOLUTE),
        },
        "action": {
            "workspace_rgb": _spec(88, NormalizationType.NONE, shift=0.0),
            "joint_state_lowdim": _spec(15, NormalizationType.VARIANCE, shift=0.0, target_repr=LieRepr.ABSOLUTE),
        },
    }
    data_transforms = [
        {"name": "CosmosProcessImage", "targets": ["workspace_rgb"], "resize_sizes": [720, 1280]},  # [H, W]
        {"name": "Flatten", "targets": ["lowdim"]},
        {"name": "Concat", "targets": ["action/joint_state_lowdim"], "out_key": "action/lowdim_concat"},
        {"name": "Concat", "targets": ["obs/joint_state_lowdim"], "out_key": "obs/lowdim_concat"},
    ]

    ds = MimicDataset(
        data_dir=data_dir,
        timestep_anchor="workspace_rgb",
        data_components=data_components,
        data_transforms=copy.deepcopy(data_transforms),  # make_data_transforms pops "name"
        policy_io=policy_io,
        source_component_names={},
        should_include_padded_tails=True,
        seed=42,
        num_val_episodes=0,  # only 1 episode in the smoke -> keep it in the train split
        train=True,
        verbose=True,
    )
    print(f"[data] dataset len={len(ds)}  stats_id={ds.stats_id[:12]}...")

    sample = ds[0]
    for k in sorted(sample):
        v = sample[k]
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    ws_o, ws_a = sample["obs/workspace_rgb"], sample["action/workspace_rgb"]
    lo_o, lo_a = sample["obs/lowdim_concat"], sample["action/lowdim_concat"]
    checks = {
        "obs/workspace_rgb (3,5,720,1280) f32": ws_o.shape == (3, 5, 720, 1280) and ws_o.dtype == np.float32,
        "action/workspace_rgb (3,88,720,1280) f32": ws_a.shape == (3, 88, 720, 1280) and ws_a.dtype == np.float32,
        "rgb normalized to [-1,1]": float(ws_o.min()) >= -1.01 and float(ws_o.max()) <= 1.01,
        "obs/lowdim_concat (1,14) f32": lo_o.shape == (1, 14) and lo_o.dtype == np.float32,
        "action/lowdim_concat (15,14) f32": lo_a.shape == (15, 14) and lo_a.dtype == np.float32,
        "obs+action workspace = 93 px frames (->24 latent)": ws_o.shape[1] + ws_a.shape[1] == 93,
    }
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'XX'}] {name}")

    ok = all(checks.values())
    print("\n==> " + ("PASS: MimicDataset video+lowdim path (chunk_reader + transforms) on real 720p data"
                      if ok else "CHECK failures above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w2a_preprocess_smoke")

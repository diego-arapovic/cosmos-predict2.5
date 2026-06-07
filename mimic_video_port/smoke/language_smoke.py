#!/usr/bin/env python
"""Validate the Reason1 language cache + lookup transform inside the dataset (the T5 -> Reason1 swap).

    python mimic_video_port/smoke/language_smoke.py [/tmp/w2a_preprocess_smoke]

Run preprocess_smoke.py first. Builds the Reason1 instruction cache (runs the 7B encoder once if the
cache is missing -- needs gated nvidia/Cosmos-Reason1-7B), then builds a MimicDataset whose obs reads
`language_instruction` and whose `Reason1EmbeddingLookup` transform turns it into the cached embedding,
and checks the sample's `obs/language_embedding` is (1, L, 100352) -- exactly what the model feeds the
frozen backbone's crossattn_proj (100352 -> 1024). Lowdim is included so the chunk reader has a
non-persistent action component for its window math.
"""
import copy
import pathlib
import sys

import numpy as np

from mimic_video_port.data_preprocessing.precompute_reason1 import DEFAULT_CACHE_NAME, build_cache
from mimic_video_port.world2action.data.action.dataset_action import MimicDataset
from mimic_video_port.world2action.data.action.types import LieRepr, NormalizationType, ObsType


def _jspec(horizon):
    return {
        "horizon": horizon,
        "target_frequency": 16,
        "shift_right_by": 0.0,
        "normalization_type": NormalizationType.VARIANCE,
        "target_repr": LieRepr.ABSOLUTE,
    }


def main(data_dir):
    data_dir = pathlib.Path(data_dir)
    cache_path = data_dir / DEFAULT_CACHE_NAME
    if cache_path.exists():
        print(f"[lang] using existing cache {cache_path}")
    else:
        print("[lang] cache missing -> building it (runs the 7B Reason1 encoder once)")
        build_cache(data_dir, cache_path)

    data_components = {
        "joint_state_lowdim": {"obs_type": ObsType.JOINT_POS, "repr": LieRepr.ABSOLUTE},
        "language_instruction": {"obs_type": ObsType.LANGUAGE, "repr": None},
    }
    policy_io = {
        "obs": {
            "joint_state_lowdim": _jspec(1),
            "language_instruction": {
                "horizon": 1,
                "target_frequency": None,
                "shift_right_by": 0.0,
                "normalization_type": NormalizationType.NONE,
                "target_repr": None,
            },
        },
        "action": {"joint_state_lowdim": _jspec(15)},
    }
    data_transforms = [
        {"name": "Flatten", "targets": ["lowdim"]},
        {"name": "Concat", "targets": ["action/joint_state_lowdim"], "out_key": "action/lowdim_concat"},
        {"name": "Concat", "targets": ["obs/joint_state_lowdim"], "out_key": "obs/lowdim_concat"},
        {
            "name": "Reason1EmbeddingLookup",
            "targets": ["language_instruction"],
            "cache_path": str(cache_path),
            "out_key": "obs/language_embedding",
        },
    ]

    ds = MimicDataset(
        data_dir=str(data_dir),
        timestep_anchor="workspace_rgb",
        data_components=data_components,
        data_transforms=copy.deepcopy(data_transforms),
        policy_io=policy_io,
        source_component_names={},
        should_include_padded_tails=True,
        seed=42,
        num_val_episodes=0,
        train=True,
        verbose=False,
    )
    print(f"[lang] dataset len={len(ds)}")
    sample = ds[0]
    for k in sorted(sample):
        v = np.asarray(sample[k])
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    le = np.asarray(sample["obs/language_embedding"])
    ok = le.ndim == 3 and le.shape[0] == 1 and le.shape[-1] == 100352
    print(f"\n[lang] obs/language_embedding {tuple(le.shape)} {le.dtype} (expect (1, L, 100352))")
    print("==> " + ("PASS: Reason1 cache + lookup transform -> obs/language_embedding (100352-d)"
                    if ok else "CHECK above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w2a_preprocess_smoke")

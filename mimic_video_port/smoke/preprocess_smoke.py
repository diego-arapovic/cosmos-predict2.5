#!/usr/bin/env python
"""Validate process_recordings.py on the example episode (raw mcap+mp4 -> world2action zarr).

    python mimic_video_port/smoke/preprocess_smoke.py [dummy_ep] [/tmp/w2a_preprocess_smoke]

Converts the first discovered episode under <input_dir> and checks the output zarr has the schema the
world2action MimicDataset expects: workspace_rgb (T,720,1280,3) uint8 (720p, camera_top), wrist_rgb_*,
joint_state_lowdim (T,14) float32 (yam_left++yam_right), language_instruction, and matching timestamps.

Data-prep deps (uv project -> use `uv pip install`, not pip; setup.sh also installs them after sync).
Pin zarr<3 -- the pipeline uses the zarr v2 API (Group.create_dataset / numcodecs.Blosc):
    uv pip install mcap "zarr<3" "numcodecs<0.16" imageio imageio-ffmpeg
    python -c "import cv2" || uv pip install opencv-python-headless
"""
import pathlib
import sys

import numpy as np
import zarr

from mimic_video_port.data_preprocessing.process_recordings import (
    DEFAULT_RESOLUTION,
    ConversionTask,
    convert_episode,
    discover_episodes,
)


def main(input_dir, out_dir):
    W, H = DEFAULT_RESOLUTION
    episodes = discover_episodes(pathlib.Path(input_dir))
    assert episodes, f"no episode_*/session_meta.json found under {input_dir!r}"
    print(f"[prep] {len(episodes)} episode(s) found; converting {episodes[0].name} @ {W}x{H}")

    out = pathlib.Path(out_dir)
    task = ConversionTask(
        episode_dir=episodes[0],
        episode_index=0,
        output_dir=out,
        max_sync_ms=50.0,
        resolution=DEFAULT_RESOLUTION,
        cameras=("workspace_rgb",),  # default: camera_top only (world2action uses only this)
        overwrite=True,
        dry_run=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    print("[prep] " + convert_episode(task))

    z = zarr.open(str(out / f"{episodes[0].name}.zarr"), mode="r")  # stable name by source episode id
    print(f"[prep] instruction: {z.attrs.get('instruction')!r}  res={z.attrs.get('resolution_wh')}")
    for k in sorted(z.array_keys()):
        print(f"  {k}: {z[k].shape} {z[k].dtype}")

    ws = z["workspace_rgb"]
    js = z["joint_state_lowdim"]
    T = ws.shape[0]
    ts = z["workspace_rgb_timestamps"][:]
    keys = set(z.array_keys())
    checks = {
        "workspace_rgb is (T,720,1280,3) uint8": ws.shape[1:] == (H, W, 3) and ws.dtype == np.uint8,
        "joint_state_lowdim is (T,14) float32": js.shape == (T, 14) and js.dtype == np.float32,
        "wrist cams dropped (workspace-only default)": "wrist_rgb_left" not in keys and "wrist_rgb_right" not in keys,
        "instruction non-empty": bool(z.attrs.get("instruction")),
        "timestamps strictly increasing": T == 1 or bool(np.all(np.diff(ts) > 0)),
        "joints finite": bool(np.all(np.isfinite(js[:]))),
    }
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'XX'}] {name}")

    ok = all(checks.values())
    print(f"\n[prep] episode T={T} frames @ {W}x{H}")
    print("\n==> " + ("PASS: process_recordings.py produces the expected world2action zarr (720p)"
                      if ok else "CHECK failures above"))


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "dummy_ep",
        sys.argv[2] if len(sys.argv) > 2 else "/tmp/w2a_preprocess_smoke",
    )

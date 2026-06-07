#!/usr/bin/env python
"""Validate the offline normalizer-stats path (precompute_stats + StaticBatchNormalizer.build_from_stats).

    python mimic_video_port/smoke/stats_smoke.py [/tmp/w2a_preprocess_smoke]

Run preprocess_smoke.py first. Computes the action-normalizer statistics from the converted zarr
(`MimicDataset.get_statistics()` — reads only the VARIANCE-normalized lowdim, not the 720p video),
saves them, reloads, builds the StaticBatchNormalizer with the YAMS normalization_types + concat_groups,
and normalizes a dummy action/obs chunk. This is exactly what World2ActionModel.on_train_start loads
(2.5's trainer doesn't pass dataset stats). Lightweight: no 2B backbone.
"""
import pathlib
import sys

import torch

from mimic_video_port.data_preprocessing.precompute_stats import build_stats
from mimic_video_port.world2action.data import yams_config
from mimic_video_port.world2action.normalizer import StaticBatchNormalizer


def main(data_dir):
    data_dir = pathlib.Path(data_dir)
    out = data_dir / "normalizer_stats.pt"
    build_stats(data_dir, out)

    blob = torch.load(out, map_location="cpu", weights_only=False)
    print(f"[stats] stats_id={blob['stats_id'][:12]}...  fields={sorted(blob['stats'])}")

    norm = StaticBatchNormalizer().build_from_stats(
        blob["stats"],
        normalization_types=yams_config.normalization_types(),
        concat_groups=yams_config.CONCAT_GROUPS,
        dtype=torch.float32,
    )
    have = set(norm.norms.keys())
    print(f"[stats] normalizer keys: {sorted(have)}")

    batch = {"action/lowdim_concat": torch.randn(2, 15, 14), "obs/lowdim_concat": torch.randn(2, 1, 14)}
    out_b = norm(batch, strict=False)
    shapes_ok = out_b["action/lowdim_concat"].shape == (2, 15, 14) and out_b["obs/lowdim_concat"].shape == (2, 1, 14)
    finite = bool(torch.isfinite(out_b["action/lowdim_concat"]).all() and torch.isfinite(out_b["obs/lowdim_concat"]).all())

    checks = {
        "stats has obs/action joint_state_lowdim": {"action/joint_state_lowdim", "obs/joint_state_lowdim"} <= set(blob["stats"]),
        "normalizer built lowdim_concat keys": {"action/lowdim_concat", "obs/lowdim_concat"} <= have,
        "normalize preserves shape": shapes_ok,
        "normalized output finite": finite,
    }
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'XX'}] {name}")

    ok = all(checks.values())
    print("\n==> " + ("PASS: precompute_stats + build_from_stats -> action normalizer (what on_train_start loads)"
                      if ok else "CHECK failures above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w2a_preprocess_smoke")

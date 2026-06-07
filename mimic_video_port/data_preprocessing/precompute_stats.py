"""Precompute action-normalizer statistics for world2action (offline; loaded at train start).

2.5's ImaginaireTrainer calls `model.on_train_start(memory_format)` with NO dataset stats (mimic-video's
trainer passed them via introspection). To keep the overlay self-contained (no stock-2.5 edits) we
compute the per-field normalization statistics offline once and `World2ActionModel.on_train_start`
loads them from `config.normalizer_stats_path`.

`MimicDataset.get_statistics()` is cheap here: it restricts the chunk reader to the VARIANCE-normalized
field (`joint_state_lowdim`) and ignores the image/concat/language transforms, so it never reads the
720p video. Stats are also cached by the dataset under `<data-dir>/.statistics_cache/<stats_id>`.

    python mimic_video_port/data_preprocessing/precompute_stats.py --data-dir data/teleop_converted \
        --out data/teleop_converted/normalizer_stats.pt

Output: a `.pt` with {"stats_id": str, "stats": {field: {mean,std,percentiles,...}}}.
"""
import argparse
import pathlib

import torch

from mimic_video_port.world2action.data import yams_config
from mimic_video_port.world2action.data.action.dataset_action import MimicDataset

PLACEHOLDER_CACHE = "<reason1-cache-unused-for-stats>"  # language is skipped during get_statistics()


def build_stats(data_dir: pathlib.Path, out_path: pathlib.Path, num_val_episodes: int = 0) -> dict:
    ds = MimicDataset(
        data_dir=str(data_dir),
        **yams_config.dataset_kwargs(PLACEHOLDER_CACHE),
        seed=42,
        num_val_episodes=num_val_episodes,
        train=True,
        verbose=True,
    )
    print(f"[stats] dataset len={len(ds)}  stats_id={ds.stats_id[:12]}...")
    stats = ds.get_statistics()  # full pass over the (lowdim-only) normalized fields
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats_id": ds.stats_id, "stats": stats}, out_path)
    fields = sorted(stats.keys())
    print(f"[stats] saved stats for {fields} -> {out_path}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute world2action normalizer statistics.")
    p.add_argument("--data-dir", type=pathlib.Path, required=True, help="Dir of converted episode_*.zarr files.")
    p.add_argument("--out", type=pathlib.Path, default=None, help="Output .pt (default: <data-dir>/normalizer_stats.pt).")
    p.add_argument("--num-val-episodes", type=int, default=0)
    args = p.parse_args()
    out = args.out or (args.data_dir / "normalizer_stats.pt")
    build_stats(args.data_dir, out, num_val_episodes=args.num_val_episodes)


if __name__ == "__main__":
    main()

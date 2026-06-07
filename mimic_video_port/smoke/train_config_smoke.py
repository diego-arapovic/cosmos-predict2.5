#!/usr/bin/env python
"""Validate the YAMS training experiment resolves + trains one step (mirrors scripts.train, no DDP/loop).

    python mimic_video_port/smoke/train_config_smoke.py [yams_smoke]

Does what cosmos_oss/scripts/train.py does up to the trainer loop:
  make_config() -> override(experiment=...) -> instantiate(config.model) -> on_train_start
  -> instantiate(config.dataloader_train) -> one training_step (+ backward).
This is the gate for the whole training wiring: Hydra compose of a custom (non-video2world) model, the
World2ActionModel built from config, the MimicDataset dataloader (real 720p data), the normalizer load,
and the frozen-backbone-tap -> decoder -> RF-loss -> backward path. Reads the real data at
$W2A_DATA/teleop_converted (run prepare_data.sh first); paths are env-overridable (see yams_experiment.py).
"""
import sys

import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import override

import mimic_video_port.config_yams as cfg


def main(experiment="yams_smoke"):
    config = cfg.make_config()
    config = override(config, ["--", f"experiment={experiment}"])
    print(f"[cfg] resolved experiment={experiment} | job={config.job.project}/{config.job.group}/{config.job.name}")
    print(f"[cfg] trainer: dist={config.trainer.distributed_parallelism} max_iter={config.trainer.max_iter} "
          f"ckpt_type={getattr(config.checkpoint, 'type', None)}")

    print("[cfg] instantiating model (loads frozen 2B backbone + builds decoder)...")
    model = instantiate(config.model).to("cuda")
    model.on_train_start(torch.preserve_format)
    n_total = sum(p.numel() for p in model.parameters()) / 1e9
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[cfg] model: {n_total:.2f}B params ({n_train:.0f}M trainable) | "
          f"normalizer keys={sorted(model.pipe.normalizer.norms.keys())}")

    print("[cfg] instantiating dataloader + fetching one real batch...")
    dl = instantiate(config.dataloader_train)
    batch = next(iter(dl))
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    print("[cfg] batch keys/shapes: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in batch.items() if torch.is_tensor(v)))

    output_batch, loss = model.training_step(batch, iteration=0)
    loss.backward()
    dec = [p for p in model.pipe.dit.parameters() if p.requires_grad]
    n_grad = sum(1 for p in dec if p.grad is not None)
    bb_grads = sum(1 for p in model.backbone.net.parameters() if p.grad is not None)

    checks = {
        "normalizer built ({obs,action}/lowdim_concat)": {"obs/lowdim_concat", "action/lowdim_concat"} <= set(model.pipe.normalizer.norms.keys()),
        "training_step loss finite": bool(torch.isfinite(loss).item()),
        "decoder gets grads": n_grad > 0,
        "frozen backbone gets 0 grads": bb_grads == 0,
    }
    print(f"[cfg] loss={loss.item():.4f} | decoder grads {n_grad}/{len(dec)} | backbone grads {bb_grads}")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'XX'}] {name}")
    print("\n==> " + ("PASS: yams experiment resolves and trains one step end-to-end (ready for scripts.train)"
                      if all(checks.values()) else "CHECK failures above"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "yams_smoke")

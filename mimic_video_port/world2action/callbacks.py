# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weights & Biases logging for world2action training + in-training eval.

We can't reuse the stock `predict2.callbacks.wandb_log.WandbCallback`: it assumes a diffusion
`output_batch["edm_loss"]` (KeyError for us) and an image/video batch split, and it only logs a scalar
val loss -- it has no notion of our action-prediction eval. This callback is the minimal world2action
analogue. It performs NO collective ops itself: the model's training_step/validation_step already
DP-reduce every scalar it returns, so here we only accumulate + log on rank 0.

Logged to W&B (curves you can watch to confirm the decoder is learning a good policy):
  train/loss, train/Var_inst[x_0]          -- every trainer.logging_iter (the RF velocity loss)
  optim/lr_*                               -- learning rate(s)
  val/loss                                 -- held-out RF velocity loss (unbiased analogue of train/loss)
  val/action_mse/sigma<σ>, val/action_mse_mean
                                           -- the policy's ACTUAL sampled action chunk vs the demo
                                              actions, in raw joint space (the closest in-training
                                              proxy for rollout quality). Only when the experiment sets
                                              model.config.val_action_sigmas.
  val/action_nmse_mean                     -- same, in NORMALIZED (per-dim unit-variance) space, so the
                                              few joints that move in a task aren't drowned by static ones.
  val/baseline_hold_last_mse               -- "hold the last observed joint state" error = how much the
                                              demo actually moves; the policy should drive action_mse below it.
  val/echo_mse                             -- distance from the do-nothing solution. Read with the two
                                              above it forms the copy<->perfect axis: echo_mse->0 with
                                              action_mse~=baseline means the policy collapsed to proprio echo.
  val/skill_ratio                          -- action_mse_mean / baseline (<1 = beats do-nothing); only
                                              logged when the demo moves (baseline > 1e-5).

Mode (online vs offline) is `config.job.wandb_mode` (driven by the WANDB_MODE env in the experiment
config / commands/train.sh). Offline runs record to the local job dir and sync later with `wandb sync`.
"""
from __future__ import annotations

import torch
import wandb

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, wandb_util
from cosmos_predict2._src.imaginaire.utils.callback import Callback


class World2ActionWandb(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._tr_loss = 0.0
        self._tr_var = 0.0
        self._tr_n = 0
        self._v: dict | None = None

    # ---- init / teardown ----
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)  # rank0-guarded internally; reads config.job.wandb_mode

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if distributed.is_rank0() and wandb.run is not None:
            wandb.finish()

    # ---- train ----
    def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0() and wandb.run is not None:
            wandb.log({f"optim/lr_{i}": g["lr"] for i, g in enumerate(optimizer.param_groups)}, step=iteration)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        # output_batch carries the already-DP-reduced scalars only on data-parallel rank 0.
        if "loss" in output_batch:
            self._tr_loss += float(output_batch["loss"])
            self._tr_var += float(output_batch.get("Var_inst[x_0]", 0.0))
            self._tr_n += 1
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0() and wandb.run is not None:
            if self._tr_n:
                wandb.log(
                    {
                        "train/loss": self._tr_loss / self._tr_n,
                        "train/Var_inst[x_0]": self._tr_var / self._tr_n,
                        "iteration": iteration,
                    },
                    step=iteration,
                )
            self._tr_loss = self._tr_var = 0.0
            self._tr_n = 0

    # ---- validation (aggregate across the val batches the trainer feeds us, then log once) ----
    def on_validation_start(self, model, dataloader_val, iteration: int = 0) -> None:
        self._v = {"loss": [], "base": [], "echo": [], "nmse": [], "mse": {}}

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if self._v is None:
            return
        if "loss" in output_batch:
            self._v["loss"].append(float(output_batch["loss"]))
        if "action_baseline_mse" in output_batch:
            self._v["base"].append(float(output_batch["action_baseline_mse"]))
        if "action_echo_mse" in output_batch:
            self._v["echo"].append(float(output_batch["action_echo_mse"]))
        if "action_nmse" in output_batch:
            self._v["nmse"].append(float(output_batch["action_nmse"]))
        for sigma_key, mse in output_batch.get("action_mse", {}).items():
            self._v["mse"].setdefault(sigma_key, []).append(float(mse))

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def on_validation_end(self, model, iteration: int = 0) -> None:
        if self._v is None or not distributed.is_rank0() or wandb.run is None:
            return
        info: dict[str, float] = {}
        for key, name in (("loss", "val/loss"), ("base", "val/baseline_hold_last_mse"),
                          ("echo", "val/echo_mse"), ("nmse", "val/action_nmse_mean")):
            m = self._mean(self._v[key])
            if m is not None:
                info[name] = m
        per_sigma = {k: self._mean(v) for k, v in self._v["mse"].items() if v}
        for sigma_key, mse in per_sigma.items():
            info[f"val/action_mse/sigma{sigma_key}"] = mse
        if per_sigma:
            info["val/action_mse_mean"] = sum(per_sigma.values()) / len(per_sigma)
            # skill_ratio < 1 means the policy beats "do nothing"; only meaningful when the demo moves.
            base = info.get("val/baseline_hold_last_mse")
            if base is not None and base > 1e-5:
                info["val/skill_ratio"] = info["val/action_mse_mean"] / base
        if info:
            log.info(f"[w2a val iter {iteration}] " + ", ".join(f"{k}={v:.5f}" for k, v in info.items()))
            wandb.log(info, step=iteration)
        self._v = None

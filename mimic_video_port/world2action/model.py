# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""World2ActionModel ported to Cosmos-Predict2.5.

Minimal diff over mimic-video's `models/world2action_model.py`, with ONE structural change: the
frozen video backbone is no longer a `Video2WorldPipeline` (absent in 2.5) but a `FrozenVideoBackbone`
(`backbone.py`). The method calls map ~1:1:

    video2world_pipe.get_mimic_data_and_condition  -> backbone.get_mimic_data_and_condition
    video2world_pipe.denoise(..., return_only_hidden_states_up_to=idx).hidden_states[idx]
                                                   -> backbone.extract_crossattn_emb(..., feature_id=idx)
    model.draw_video_sigma (EDM high-sigma)        -> backbone.draw_video_sigma (rectified-flow)

The trainable action decoder (`World2ActionPipeline`, `pipe`) is unchanged. Only the decoder trains;
the 2B backbone is frozen.
"""
import collections
import gc
from collections.abc import Mapping
from typing import Any

import attrs
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.utils.clip_grad import clip_grad_norm_

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict, instantiate
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_scheduler

from .backbone import FrozenVideoBackbone
from .config import EMAConfig, World2ActionPipelineConfig
from .pipeline import World2ActionPipeline


@attrs.define(slots=False)
class World2ActionModelConfig:
    train_architecture: str  # base or lora
    lora_rank: int
    lora_alpha: int
    lora_target_modules: str
    init_lora_weights: bool

    precision: str
    loss_reduce: str
    loss_scale: float
    ema: EMAConfig

    # This is used for the original way to load models
    action_dit_path: str  # the trainable action decoder
    video_dit_path: str  # the frozen 2B video backbone (consolidated finetuned ckpt; fused preferred)
    pipe_config: World2ActionPipelineConfig

    fsdp_shard_size: int  # 0 means not using fsdp, -1 means set to world size

    # Frozen-backbone build params (replace the old `video_pipe_config: Video2WorldPipelineConfig`,
    # which is absent in 2.5 -- the backbone is built piecemeal by FrozenVideoBackbone.from_pretrained).
    video_state_t: int = 24
    video_shift: int = 5
    video_fps: float = 16.0

    # Precomputed action-normalizer stats (data_preprocessing/precompute_stats.py). 2.5's trainer calls
    # on_train_start(memory_format) with no dataset stats, so we load them from here instead.
    normalizer_stats_path: str = ""

    # Unused (normalization spec now comes from yams_config, not here); kept for config compatibility.
    data_config: DictConfig | None = None


def _dp_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        group = parallel_state.get_data_parallel_group()
        world = parallel_state.get_data_parallel_world_size()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        x /= world
    return x


def _dp_mean_dict(d: dict[str, object], device: torch.device) -> dict[str, float]:
    keys = list(d.keys())
    t = torch.stack([torch.as_tensor(d[k], device=device, dtype=torch.float32) for k in keys], dim=0)
    t = _dp_mean(t)
    return {k: t[i].item() for i, k in enumerate(keys)}


class World2ActionModel(ImaginaireModel):
    def __init__(self, config: World2ActionModelConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # 1. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")

        # 2. The trainable action decoder.
        self.pipe: World2ActionPipeline = World2ActionPipeline.from_config(
            config.pipe_config,
            dit_path=config.action_dit_path,
            **self.tensor_kwargs,
        )

        # 3. The frozen 2B video backbone (replaces the old Video2WorldPipeline). Built piecemeal and
        #    frozen inside from_pretrained (net.eval().requires_grad_(False)).
        self.backbone: FrozenVideoBackbone = FrozenVideoBackbone.from_pretrained(
            config.video_dit_path,
            device=self.tensor_kwargs["device"],
            dtype=self.precision,
            state_t=config.video_state_t,
            fps=config.video_fps,
            shift=config.video_shift,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda",
                (replica_group_size, fsdp_shard_size),
                mesh_dim_names=("replicate", "shard"),
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            # Only the trainable decoder is sharded; the frozen backbone is forward-only and fits per-GPU.
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    # New function, added for i4 adaption
    @property
    def net(self) -> torch.nn.Module:
        return self.pipe.dit

    # New function, added for i4 adaption
    @property
    def net_ema(self) -> torch.nn.Module:
        return self.pipe.dit_ema

    @property
    def tokenizer(self):
        # Exposed so 2.5's video2world callbacks work on our custom model (e.g. `compile_tokenizer` does
        # `model.tokenizer.encode = torch.compile(...)`). It's the frozen backbone's Wan2.1 VAE — compiling
        # its encode is a perf win, since get_mimic_data_and_condition calls it every step.
        return self.backbone.tokenizer

    def is_image_batch(self, batch: dict) -> bool:
        return False

    # New function, added for i4 adaption
    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer: torch.optim.Optimizer = instantiate(optimizer_config, model=self.net)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.pipe_config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.pipe.dit_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # New function, added for i4 adaption.
    # NOTE: 2.5's ImaginaireTrainer calls on_train_start(memory_format) with NO dataset stats (mimic-video's
    # trainer passed dataset_stats/stats_id via introspection). To avoid patching the stock trainer, we load
    # precomputed normalizer stats from config.normalizer_stats_path (data_preprocessing/precompute_stats.py).
    # See DESIGN_RISKS.md #17.
    def on_train_start(self, memory_format: torch.memory_format) -> None:
        from .data import yams_config

        if self.config.pipe_config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        stats_path = getattr(self.config, "normalizer_stats_path", "")
        if not stats_path:
            log.warning("normalizer_stats_path not set; skipping normalizer build (actions will NOT be normalized)")
            return
        blob = torch.load(stats_path, map_location="cpu", weights_only=False)
        dataset_stats = blob["stats"] if isinstance(blob, dict) and "stats" in blob else blob
        self.stats_id = blob.get("stats_id") if isinstance(blob, dict) else None
        # The normalization spec (types + concat_groups) comes from the single source of truth, yams_config;
        # only the per-field stats numbers come from the file. This keeps enums out of the Hydra config.
        self.pipe.normalizer.build_from_stats(
            dataset_stats,
            normalization_types=yams_config.normalization_types(),
            concat_groups=yams_config.CONCAT_GROUPS,
            **self.tensor_kwargs,
        )
        self.pipe.normalizer.requires_grad_(False)

    def freeze_parameters(self) -> None:
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
    ):
        from peft import LoraConfig, inject_adapter_in_model

        # Add LoRA to UNet
        self.lora_alpha = lora_alpha

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def draw_training_t_and_epsilon(
        self,
        x0_size: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = torch.randn(x0_size, dtype=torch.float32, device=self.tensor_kwargs["device"])
        t_B = self.pipe.scheduler.sample_t(x0_size[0])

        return t_B.unsqueeze(1).repeat(1, x0_size[1]).unsqueeze(2), epsilon

    def compute_loss_with_epsilon_and_t(
        self,
        x0_B_HA_A: torch.Tensor,
        epsilon_B_HA_A: torch.Tensor,
        t_B_HA_1: torch.Tensor,
        crossattn_emb: torch.Tensor,
        video_sigma_B_1: torch.Tensor,
        state_B_HO_O: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """
        Compute loss given epsilon and t

        It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            crossattn_emb: video condition
            epsilon: noise
            t: noise level
        """
        # scale to have unit variance. don't know if this helps.
        xt_B_HA_A = (1 - t_B_HA_1) * x0_B_HA_A + t_B_HA_1 * epsilon_B_HA_A
        ut_B_HA_A = epsilon_B_HA_A - x0_B_HA_A

        vt_B_HA_A = self.pipe.denoise(
            xt_B_HA_A,
            t_B_HA_1,
            state_B_HO_O,
            crossattn_emb,
            video_sigma_B_1,
            obs_dropout=0.2,
            return_hidden_states=False,
        ).float()
        loss = F.mse_loss(vt_B_HA_A, ut_B_HA_A, reduction=self.loss_reduce) * self.loss_scale

        with torch.no_grad():
            var_inst_x0 = x0_B_HA_A.float().var(dim=(1, 2)).mean()

            metrics = torch.stack(
                [
                    loss.float(),
                    var_inst_x0,
                ],
                dim=0,
            ).to(x0_B_HA_A.device)
            metrics = _dp_mean(metrics)

            if not dist.is_available() or not dist.is_initialized() or parallel_state.get_data_parallel_rank() == 0:
                output_batch = {
                    "loss": metrics[0].item(),
                    "Var_inst[x_0]": metrics[1].item(),
                }
            else:
                output_batch = {}

        del var_inst_x0  # , var_batch_x0, var_eps, var_xt, var_ut, var_vt
        gc.collect(0)

        return output_batch, loss

    def get_crossattn_emb(
        self,
        data_batch: dict,
        video_sigma_B_1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Frozen-backbone feature tap. Replaces the old
        #   video2world_pipe.denoise(video + eps*sigma, sigma, condition,
        #                            return_only_hidden_states_up_to=idx).hidden_states[idx]
        # with FrozenVideoBackbone (rectified-flow noising + native intermediate_feature_ids tap).
        _, video_B_C_T_H_W, condition = self.backbone.get_mimic_data_and_condition(data_batch)

        video_epsilon_B_C_T_H_W = torch.randn(video_B_C_T_H_W.size(), **self.tensor_kwargs)

        if video_sigma_B_1 is None:
            video_sigma_B_1 = self.draw_video_sigma(video_B_C_T_H_W.size(), condition)

        crossattn_emb = self.backbone.extract_crossattn_emb(
            video_B_C_T_H_W,
            video_epsilon_B_C_T_H_W,
            video_sigma_B_1,
            condition,
            feature_id=self.pipe.config.xattn_layer_idx,
        )  # already (B, T*H*W, D)

        gc.collect(0)

        return crossattn_emb, video_sigma_B_1

    def predict(self, data_batch: dict, video_sigma_B_1: torch.Tensor) -> torch.Tensor:
        crossattn_emb, video_sigma_B_1 = self.get_crossattn_emb(data_batch, video_sigma_B_1)
        state_B_HO_O = data_batch["obs/lowdim_concat"]

        return self.pipe(state_B_HO_O, crossattn_emb, video_sigma_B_1)

    def draw_video_sigma(self, x0_size: torch.Size, condition: Any = None) -> torch.Tensor:
        # The rectified-flow backbone owns sigma sampling (logitnormal + shift). This delegates so the
        # decoder is conditioned on the same sigma the features were tapped at (context_timesteps_B_1).
        return self.backbone.draw_video_sigma(x0_size, condition)

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        data_batch["obs/language_embedding"] = data_batch["obs/language_embedding"].squeeze(1)
        B, _HA, A = data_batch["action/lowdim_concat"].shape
        if "obs/lowdim_concat" not in data_batch:
            data_batch["obs/lowdim_concat"] = torch.empty((B, 0, A), **self.tensor_kwargs)

        crossattn_emb, video_sigma_B_1 = self.get_crossattn_emb(data_batch)

        normalised_data_batch: dict = self.pipe.normalizer(data_batch, strict=False)

        x0_B_HA_A = normalised_data_batch["action/lowdim_concat"]

        state_B_HO_O = normalised_data_batch["obs/lowdim_concat"]

        t_B_HA_1, epsilon_B_HA_A = self.draw_training_t_and_epsilon(x0_B_HA_A.size())

        output_batch, loss = self.compute_loss_with_epsilon_and_t(
            x0_B_HA_A,
            epsilon_B_HA_A,
            t_B_HA_1,
            crossattn_emb,
            video_sigma_B_1,
            state_B_HO_O,
        )

        return output_batch, loss

    @torch.inference_mode()
    def validation_step(self, data_batch: dict, iteration: int):
        # Loss + a "ground-truth-video" MSE sweep: predict the action chunk from features tapped at a
        # range of video noise levels and compare to the demonstrated actions. The "generated-video"
        # sweep (mimic-video's genvid via video2world_pipe.generate_video) needs a backbone sampler and
        # is deferred to the eval port (step 4).
        output_batch, loss = self.training_step(data_batch, iteration)
        unnormed_x0_B_HA_A = data_batch["action/lowdim_concat"]

        output_batch["mses"] = collections.defaultdict(list)

        for video_sigma in torch.linspace(0.1, 0.9, 9, device=self.tensor_kwargs["device"]):
            video_sigma_B_1 = video_sigma.repeat(unnormed_x0_B_HA_A.shape[0]).unsqueeze(1)
            unnormed_x0_pred_B_HA_A = self.predict(data_batch, video_sigma_B_1).float()

            mses_gtvid = {
                "gtvid/full": F.mse_loss(unnormed_x0_pred_B_HA_A, unnormed_x0_B_HA_A.float()),
            }
            mses_gtvid = _dp_mean_dict(mses_gtvid, device=unnormed_x0_pred_B_HA_A.device)

            if dist.is_available() and dist.is_initialized() and parallel_state.get_data_parallel_rank() != 0:
                continue

            for name, mse in mses_gtvid.items():
                output_batch["mses"][name].append((video_sigma.item(), mse))

        return output_batch, loss

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        # convert DTensor to Tensor
        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()

        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.pipe.dit.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )

            if self.config.pipe_config.ema.enabled:
                ema_results: _IncompatibleKeys = self.pipe.dit_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.pipe_config.ema.enabled else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.pipe_config.ema.enabled else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.pipe.dit, _reg_state_dict), rank0_only=False)
            if self.config.pipe_config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(
                    non_strict_load_model(self.pipe.dit_ema, _ema_state_dict),
                    rank0_only=False,
                )

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.pipe_config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.pipe.ema_exp_coefficient + 1)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

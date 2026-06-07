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

import attrs

from cosmos_predict2._src.imaginaire.config import make_freezable
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    """Decoder EMA (power-EMA), matching mimic-video's `config_video2world.EMAConfig` fields.

    Deliberately our own, NOT `_src.imaginaire.config.EMAConfig` (which has `enabled`/`beta` only):
    `pipeline.py` reads `ema.rate` and `model.py` reads `ema.iteration_shift`. These three fields
    match the predict2 `EMAConfig` (text2world_model.py:73) without importing that heavy module.
    """

    enabled: bool = False
    rate: float = 0.1
    iteration_shift: int = 0


@make_freezable
@attrs.define(slots=False)
class SchedulerConfig:
    alpha: float
    beta: float
    num_denoising_steps: int


@make_freezable
@attrs.define(slots=False)
class World2ActionPipelineConfig:
    precision: str
    scheduler: SchedulerConfig
    net: LazyDict  # holds the lazy World2ActionDIT config (2.5 LazyDict isn't subscriptable)
    ema: EMAConfig
    xattn_layer_idx: int

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_franka_sm_leader")
@dataclass
class BiFrankaSmLeaderConfig(TeleoperatorConfig):
    """Shared-memory based leader that mirrors dual Franka teleop commands."""

    left_namespace: str = "L"
    right_namespace: str = "R"
    vector_length: int = 8  # 7 joints + gripper flag by default
    update_rate_hz: float = 50.0

    left_joint_names: list[str] | None = None
    right_joint_names: list[str] | None = None

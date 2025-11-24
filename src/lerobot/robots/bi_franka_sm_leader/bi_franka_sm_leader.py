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

import logging
from functools import cached_property
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .config_bi_franka_sm_leader import BiFrankaSmLeaderConfig
from ...robots.bi_franka_sm_follower.bi_franka_sm_follower import _SharedArmStream

logger = logging.getLogger(__name__)


class BiFrankaSmLeader(Teleoperator):
    """SpaceMouse leader proxy that reads previously-published joint targets from shared memory."""

    config_class = BiFrankaSmLeaderConfig
    name = "bi_franka_sm_leader"

    def __init__(self, config: BiFrankaSmLeaderConfig):
        super().__init__(config)
        self.config = config
        self._left_reader = _SharedArmStream(config.left_namespace, config.vector_length)
        self._right_reader = _SharedArmStream(config.right_namespace, config.vector_length)
        self._left_features = self._build_feature_names(config.left_joint_names)
        self._right_features = self._build_feature_names(config.right_joint_names)

    def _build_feature_names(self, override: list[str] | None) -> list[str]:
        if override:
            return override
        names = [f"joint_{i}" for i in range(self.config.vector_length - 1)]
        names.append("gripper_open")
        return names

    @cached_property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}
        for prefix, names in (("left", self._left_features), ("right", self._right_features)):
            for name in names:
                feats[f"{prefix}_{name}"] = float
        return feats

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._left_reader.is_connected and self._right_reader.is_connected

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self._left_reader.connect()
        self._right_reader.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s: calibration not required.", self)

    def configure(self) -> None:
        logger.info("%s: configuration not required.", self)

    def setup_motors(self) -> None:
        raise NotImplementedError("bi_franka_sm_leader exposes no motors to setup.")

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")

    def _vec_to_dict(self, vec: np.ndarray, names: list[str], prefix: str) -> dict[str, float]:
        return {f"{prefix}_{name}": float(vec[i]) for i, name in enumerate(names)}

    def get_action(self) -> dict[str, float]:
        self._ensure_connected()
        left_vec = self._left_reader.read_vector()
        right_vec = self._right_reader.read_vector()
        action: dict[str, float] = {}
        action.update(self._vec_to_dict(left_vec, self._left_features, "left"))
        action.update(self._vec_to_dict(right_vec, self._right_features, "right"))
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:  # noqa: ARG002
        logger.info("%s: feedback channel is not implemented.", self)
        raise NotImplementedError("bi_franka_sm_leader does not support feedback.")

    def disconnect(self) -> None:
        self._left_reader.disconnect()
        self._right_reader.disconnect()

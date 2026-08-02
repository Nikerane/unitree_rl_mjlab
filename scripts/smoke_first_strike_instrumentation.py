"""Frozen record schema and pure population gate for the first-strike smoke."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import torch
import warp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import scripts.eval_impulse as eval_impulse
import src.tasks  # noqa: F401  (register the four hammer tasks)
from src.assets.robots.unitree_z1.z1_constants import (
    ARM_JOINT_NAMES,
    HAMMER_HEAD_SITE_NAME,
)
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.first_strike import FirstStrikeEventTracker
from src.tasks.hammer.mdp.guideline import (
    GUIDELINE_NUM_GATES,
    WaypointProgressTracker,
    _ENV_GUIDELINE_ATTR,
    ordered_waypoint_progress_reward,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH, NAIL_SUCCESS_THRESHOLD


SCHEMA_VERSION: str = "four-task-cuda-smoke-v1"

ARM_TASKS = {
    "C": "Unitree-Z1-Hammer-CaT-Impulse",
    "D-prime": "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
    "F": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "E": "Unitree-Z1-Hammer-CaT-Impulse-Event",
    # fq4x8 quality-conditioned campaign. F8 is deliberately absent: its
    # registered task is byte-identical to ARM_TASKS["F"] above, so the "F"
    # contract already covers every F8 predicate.
    "F0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    "D0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    "FQ-min": "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    "B8": "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
}
GUIDELINE_ARM_TASKS = {
    "C0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    "C-Gate": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
    "P": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress",
}


@dataclass(frozen=True)
class ArmContract:
    arm: str
    task: str
    impact_reader: str
    delivered_reader: str
    tracker_required: bool
    event_i_ref_n_s: float
    delivered_saturate: bool | None
    # fq4x8 payout shape: True zeroes that side's RewardManager weight to
    # exactly 0.0 (mjlab then never invokes the raw reader for it -- see
    # RewardManager.compute). Both default False (every pre-fq4x8 arm pays
    # both sides raw).
    speed_zero: bool = False
    delivered_zero: bool = False
    # True only for the quality-conditioned arm (FQ-min): requires the
    # passive hammer_nail_quality onset snapshot to be valid and non-overflowing.
    quality_required: bool = False
    # FQ-min's speed reader is bounded by a different normalizer than every
    # other arm's default 1.0; checked against config drift when not None.
    impact_v_expected_n_s: float | None = None
    # Cartesian-guideline arms require device-latched tracker geometry/gates.
    guideline_required: bool = False
    # C-Gate alone must produce a finite positive manager r_gate payout.
    gate_reward_required: bool = False
    # P alone must produce a finite positive manager dense-progress payout.
    progress_reward_required: bool = False


@dataclass(frozen=True)
class PredicateCount:
    passed: int
    total: int

    @property
    def failed(self) -> int:
        return self.total - self.passed


ARM_CONTRACTS: dict[str, ArmContract] = {
    "C": ArmContract(
        "C", ARM_TASKS["C"], "ImpactProgressTerm", "DeliveredImpulseTerm",
        False, 0.6094, None,
    ),
    "D-prime": ArmContract(
        "D-prime", ARM_TASKS["D-prime"],
        "FirstStrikeLegacyImpactRewardTerm",
        "FirstStrikeLegacyDeliveredRewardTerm", True, 0.6094, None,
    ),
    "F": ArmContract(
        "F", ARM_TASKS["F"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
    ),
    "E": ArmContract(
        "E", ARM_TASKS["E"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, True,
    ),
    "F0": ArmContract(
        "F0", ARM_TASKS["F0"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        speed_zero=True,
    ),
    "D0": ArmContract(
        "D0", ARM_TASKS["D0"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        delivered_zero=True,
    ),
    "FQ-min": ArmContract(
        "FQ-min", ARM_TASKS["FQ-min"], "FirstStrikeQualityImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        delivered_zero=True, quality_required=True,
        impact_v_expected_n_s=1.4598331451416016,
    ),
    "B8": ArmContract(
        "B8", ARM_TASKS["B8"], "FirstStrikeBoundedImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        delivered_zero=True, quality_required=True,
        impact_v_expected_n_s=1.4598331451416016,
    ),
    "C0": ArmContract(
        "C0", GUIDELINE_ARM_TASKS["C0"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        guideline_required=True,
    ),
    "C-Gate": ArmContract(
        "C-Gate", GUIDELINE_ARM_TASKS["C-Gate"],
        "FirstStrikeImpactRewardTerm", "FirstStrikeDeliveredRewardTerm",
        True, 0.3088, False, guideline_required=True,
        gate_reward_required=True,
    ),
    "P": ArmContract(
        "P", GUIDELINE_ARM_TASKS["P"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
        guideline_required=True, progress_reward_required=True,
    ),
}
TASK_CONTRACTS = {contract.task: contract for contract in ARM_CONTRACTS.values()}
LITERAL_IMPULSE_LIMITS_N_M_S = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
_FULL_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


COMMON_PREDICATES = (
    "finite_signals",
    "hammer_nail_contact",
    "normal_success",
    "lambda_live",
    "delivered_live",
    "hardware_qvel",
)
INSTRUMENTATION_PREDICATES = tuple(
    name for name in COMMON_PREDICATES if name != "hardware_qvel"
)
# Reward-manager payout predicates depend on the arm's payout shape (see
# ArmContract.speed_zero/delivered_zero): evaluate_gate selects the right
# name per side instead of requiring one fixed set for every arm.
TRACKER_COMMON_PREDICATES = (
    "tracker_exists",
    "tracker_started",
    "productive_success",
    "terminal_reason_success",
    "delivered_event_impulse",
)
TRACKER_SPEED_PULSE_PREDICATES = (
    "impact_no_early_pulse",
    "impact_exactly_one_terminal_pulse",
)
TRACKER_DELIVERED_PULSE_PREDICATES = (
    "delivered_no_early_pulse",
    "delivered_exactly_one_terminal_pulse",
)
QUALITY_PREDICATES = ("quality_snapshot_valid", "quality_no_overflow")
GUIDELINE_PREDICATES = (
    "guideline_tracker_exists",
    "guideline_tracker_initialized",
    "guideline_geometry_finite",
    "guideline_geometry_nondegenerate",
    "guideline_all_gates_crossed",
    "guideline_progress_state_finite",
)


class RawRewardTap:
    """Delegate one reward evaluation and retain its pre-sanitization device state."""

    def __init__(
        self,
        original: Any,
        num_envs: int,
        device: torch.device | str,
    ):
        self.original = original
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.last_raw = torch.zeros(num_envs, device=self.device)
        self.all_finite = torch.ones(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.positive_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.call_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.first_positive_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_positive_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device
        )

    def __call__(self, env: Any, **params: Any) -> torch.Tensor:
        raw = self.original(env, **params)
        self.last_raw.copy_(raw.detach())
        self.all_finite &= torch.isfinite(raw)
        positive = raw > 0.0
        first = positive & (self.first_positive_step < 0)
        self.first_positive_step.copy_(
            torch.where(first, self.call_count, self.first_positive_step)
        )
        self.last_positive_step.copy_(
            torch.where(positive, self.call_count, self.last_positive_step)
        )
        self.positive_count += positive.to(dtype=torch.long)
        self.call_count += 1
        return raw

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        reset = getattr(self.original, "reset", None)
        if reset is not None:
            reset(env_ids=env_ids)
        idx = slice(None) if env_ids is None else env_ids
        self.last_raw[idx] = 0.0
        self.all_finite[idx] = True
        self.positive_count[idx] = 0
        self.call_count[idx] = 0
        self.first_positive_step[idx] = -1
        self.last_positive_step[idx] = -1


def install_reward_taps(env: Any) -> tuple[RawRewardTap, RawRewardTap]:
    """Replace the two live RewardManager configs with exactly-once raw taps."""
    impact_cfg = env.reward_manager.get_term_cfg("impact_progress")
    delivered_cfg = env.reward_manager.get_term_cfg("delivered_impulse")
    if isinstance(impact_cfg.func, RawRewardTap) or isinstance(
        delivered_cfg.func, RawRewardTap
    ):
        raise RuntimeError("raw reward taps are already installed")
    impact_tap = RawRewardTap(impact_cfg.func, env.num_envs, env.device)
    delivered_tap = RawRewardTap(delivered_cfg.func, env.num_envs, env.device)
    impact_cfg.func = impact_tap
    delivered_cfg.func = delivered_tap
    return impact_tap, delivered_tap


class DeviceSmokeRecorder:
    """Accumulate and latch the first terminal episode without leaving the device."""

    def __init__(
        self,
        impact_tap: RawRewardTap,
        delivered_tap: RawRewardTap,
    ):
        if impact_tap.num_envs != delivered_tap.num_envs:
            raise ValueError("reward taps must have the same environment count")
        if impact_tap.device != delivered_tap.device:
            raise ValueError("reward taps must use the same device")
        self.impact_tap = impact_tap
        self.delivered_tap = delivered_tap
        self.num_envs = impact_tap.num_envs
        self.device = impact_tap.device
        self._env: Any | None = None
        self._original_sim_step: Any | None = None
        self._original_compute_substep: Any | None = None
        self._original_compute: Any | None = None

        shape = (self.num_envs,)
        self.contact_seen = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.force_max = torch.zeros(shape, device=self.device)
        self.pre_qvel_max = torch.zeros(shape, device=self.device)
        self.post_qvel_max = torch.zeros(shape, device=self.device)
        self.pre_depth_max = torch.zeros(shape, device=self.device)
        self.post_depth_max = torch.zeros(shape, device=self.device)
        self.manager_impact = torch.zeros(shape, device=self.device)
        self.manager_delivered = torch.zeros(shape, device=self.device)
        self.manager_gate = torch.zeros(shape, device=self.device)
        self.manager_progress = torch.zeros(shape, device=self.device)
        self.control_step = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )

        self.terminal_seen = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_depth = torch.zeros(shape, device=self.device)
        self.terminal_success = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_lambda = torch.zeros(
            self.num_envs, 6, device=self.device
        )
        self.terminal_delivered = torch.zeros(shape, device=self.device)
        self.terminal_tracker_exists = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_tracker_started = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_tracker_finalized = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_tracker_productive = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_tracker_reason = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_tracker_delivered = torch.zeros(
            shape, device=self.device
        )
        self.terminal_tracker_contact_quality_valid = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_tracker_contact_quality_overflow = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_guideline_tracker_exists = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_guideline_tracker_initialized = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_guideline_entry = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_guideline_nail = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.terminal_guideline_next_gate = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_guideline_target_start_distance = torch.zeros(
            shape, device=self.device
        )
        self.terminal_guideline_best_target_fraction = torch.zeros(
            shape, device=self.device
        )
        self.terminal_guideline_window_new_credit = torch.zeros(
            shape, device=self.device
        )
        self.terminal_guideline_episode_credit = torch.zeros(
            shape, device=self.device
        )
        self.terminal_guideline_multi_gate_crossings = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_raw_impact_finite = torch.ones(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_raw_delivered_finite = torch.ones(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_raw_impact_call_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_raw_delivered_call_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_manager_impact = torch.zeros(shape, device=self.device)
        self.terminal_manager_delivered = torch.zeros(
            shape, device=self.device
        )
        self.terminal_manager_gate = torch.zeros(shape, device=self.device)
        self.terminal_manager_progress = torch.zeros(shape, device=self.device)
        self.terminal_control_step = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_impact_positive_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_delivered_positive_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.terminal_impact_first_positive_step = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.terminal_delivered_first_positive_step = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.terminal_impact_last_positive_step = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.terminal_delivered_last_positive_step = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.terminal_contact_seen = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.terminal_force_max = torch.zeros(shape, device=self.device)
        self.terminal_pre_qvel_max = torch.zeros(shape, device=self.device)
        self.terminal_post_qvel_max = torch.zeros(shape, device=self.device)
        self.terminal_pre_depth_max = torch.zeros(shape, device=self.device)
        self.terminal_post_depth_max = torch.zeros(shape, device=self.device)

    @staticmethod
    def _masked_copy(
        destination: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        expanded_mask = mask.reshape(
            mask.shape + (1,) * (source.ndim - mask.ndim)
        )
        destination.copy_(torch.where(expanded_mask, source, destination))

    def capture_substep(
        self,
        *,
        contact: torch.Tensor,
        axial_force: torch.Tensor,
        pre_arm_qvel: torch.Tensor,
        post_arm_qvel: torch.Tensor,
        pre_depth: torch.Tensor,
        post_depth: torch.Tensor,
    ) -> None:
        self.contact_seen |= contact
        torch.maximum(self.force_max, axial_force, out=self.force_max)
        torch.maximum(
            self.pre_qvel_max,
            pre_arm_qvel.abs().amax(dim=-1),
            out=self.pre_qvel_max,
        )
        torch.maximum(
            self.post_qvel_max,
            post_arm_qvel.abs().amax(dim=-1),
            out=self.post_qvel_max,
        )
        torch.maximum(self.pre_depth_max, pre_depth, out=self.pre_depth_max)
        torch.maximum(self.post_depth_max, post_depth, out=self.post_depth_max)

    def capture_control_step(
        self,
        *,
        reset_buf: torch.Tensor,
        terminal_depth: torch.Tensor,
        terminal_success: torch.Tensor,
        episode_lambda: torch.Tensor,
        episode_delivered: torch.Tensor,
        tracker_exists: torch.Tensor,
        tracker_started: torch.Tensor,
        tracker_finalized: torch.Tensor,
        tracker_productive: torch.Tensor,
        tracker_reason: torch.Tensor,
        tracker_delivered: torch.Tensor,
        tracker_contact_quality_valid: torch.Tensor,
        tracker_contact_quality_overflow: torch.Tensor,
        guideline_tracker_exists: torch.Tensor,
        guideline_tracker_initialized: torch.Tensor,
        guideline_entry: torch.Tensor,
        guideline_nail: torch.Tensor,
        guideline_next_gate: torch.Tensor,
        guideline_target_start_distance: torch.Tensor,
        guideline_best_target_fraction: torch.Tensor,
        guideline_window_new_credit: torch.Tensor,
        guideline_episode_credit: torch.Tensor,
        guideline_multi_gate_crossings: torch.Tensor,
        manager_impact: torch.Tensor,
        manager_delivered: torch.Tensor,
        manager_gate: torch.Tensor,
        manager_progress: torch.Tensor,
    ) -> None:
        self.manager_impact += manager_impact
        self.manager_delivered += manager_delivered
        self.manager_gate += manager_gate
        self.manager_progress += manager_progress

        first_terminal = reset_buf & ~self.terminal_seen
        sources = (
            (self.terminal_depth, terminal_depth),
            (self.terminal_success, terminal_success),
            (self.terminal_lambda, episode_lambda),
            (self.terminal_delivered, episode_delivered),
            (self.terminal_tracker_exists, tracker_exists),
            (self.terminal_tracker_started, tracker_started),
            (self.terminal_tracker_finalized, tracker_finalized),
            (self.terminal_tracker_productive, tracker_productive),
            (self.terminal_tracker_reason, tracker_reason),
            (self.terminal_tracker_delivered, tracker_delivered),
            (
                self.terminal_tracker_contact_quality_valid,
                tracker_contact_quality_valid,
            ),
            (
                self.terminal_tracker_contact_quality_overflow,
                tracker_contact_quality_overflow,
            ),
            (
                self.terminal_guideline_tracker_exists,
                guideline_tracker_exists,
            ),
            (
                self.terminal_guideline_tracker_initialized,
                guideline_tracker_initialized,
            ),
            (self.terminal_guideline_entry, guideline_entry),
            (self.terminal_guideline_nail, guideline_nail),
            (self.terminal_guideline_next_gate, guideline_next_gate),
            (
                self.terminal_guideline_target_start_distance,
                guideline_target_start_distance,
            ),
            (
                self.terminal_guideline_best_target_fraction,
                guideline_best_target_fraction,
            ),
            (
                self.terminal_guideline_window_new_credit,
                guideline_window_new_credit,
            ),
            (
                self.terminal_guideline_episode_credit,
                guideline_episode_credit,
            ),
            (
                self.terminal_guideline_multi_gate_crossings,
                guideline_multi_gate_crossings,
            ),
            (self.terminal_raw_impact_finite, self.impact_tap.all_finite),
            (
                self.terminal_raw_delivered_finite,
                self.delivered_tap.all_finite,
            ),
            (
                self.terminal_raw_impact_call_count,
                self.impact_tap.call_count,
            ),
            (
                self.terminal_raw_delivered_call_count,
                self.delivered_tap.call_count,
            ),
            (self.terminal_manager_impact, self.manager_impact),
            (self.terminal_manager_delivered, self.manager_delivered),
            (self.terminal_manager_gate, self.manager_gate),
            (self.terminal_manager_progress, self.manager_progress),
            (self.terminal_control_step, self.control_step),
            (
                self.terminal_impact_positive_count,
                self.impact_tap.positive_count,
            ),
            (
                self.terminal_delivered_positive_count,
                self.delivered_tap.positive_count,
            ),
            (
                self.terminal_impact_first_positive_step,
                self.impact_tap.first_positive_step,
            ),
            (
                self.terminal_delivered_first_positive_step,
                self.delivered_tap.first_positive_step,
            ),
            (
                self.terminal_impact_last_positive_step,
                self.impact_tap.last_positive_step,
            ),
            (
                self.terminal_delivered_last_positive_step,
                self.delivered_tap.last_positive_step,
            ),
            (self.terminal_contact_seen, self.contact_seen),
            (self.terminal_force_max, self.force_max),
            (self.terminal_pre_qvel_max, self.pre_qvel_max),
            (self.terminal_post_qvel_max, self.post_qvel_max),
            (self.terminal_pre_depth_max, self.pre_depth_max),
            (self.terminal_post_depth_max, self.post_depth_max),
        )
        for destination, source in sources:
            self._masked_copy(destination, source, first_terminal)
        self.terminal_seen |= reset_buf
        self.control_step += 1

    def install(self, env: Any) -> None:
        """Wrap the live callbacks in pre-reset order; perform no host action."""
        if self._env is not None:
            raise RuntimeError("DeviceSmokeRecorder is already installed")
        if env.num_envs != self.num_envs or torch.device(env.device) != self.device:
            raise ValueError("environment shape/device differs from reward taps")

        accumulator = getattr(env, "_hammer_substep_impulse", None)
        delivered_accumulator = getattr(env, "_hammer_substep_delivered", None)
        if accumulator is None or delivered_accumulator is None:
            raise RuntimeError("smoke recorder requires both shipped impulse accumulators")

        robot = env.scene["robot"]
        nail = env.scene["nail_block"]
        contact_sensor = env.scene["hammer_nail_contact"]
        impulse_sensor = env.scene["hammer_nail_impulse"]
        arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
        arm_cfg.resolve(env.scene)
        arm_ids = arm_cfg.joint_ids
        axis = torch.tensor(
            (0.0, 0.0, -1.0), device=self.device, dtype=torch.float32
        )
        pre_qvel = torch.zeros(
            self.num_envs, 6, device=self.device
        )
        pre_depth = torch.zeros(self.num_envs, device=self.device)
        tracker = getattr(env, "_hammer_first_strike", None)
        tracker_exists = torch.full(
            (self.num_envs,),
            tracker is not None,
            dtype=torch.bool,
            device=self.device,
        )
        tracker_false = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        tracker_reason_none = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        tracker_delivered_zero = torch.zeros(
            self.num_envs, device=self.device
        )
        guideline_tracker = getattr(env, _ENV_GUIDELINE_ATTR, None)
        if not isinstance(guideline_tracker, WaypointProgressTracker):
            guideline_tracker = None
        guideline_tracker_exists = torch.full(
            (self.num_envs,),
            guideline_tracker is not None,
            dtype=torch.bool,
            device=self.device,
        )
        guideline_geometry_zero = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        guideline_gate_zero = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        guideline_scalar_zero = torch.zeros(self.num_envs, device=self.device)
        manager_gate_zero = torch.zeros(self.num_envs, device=self.device)
        impact_idx = env.reward_manager.active_terms.index("impact_progress")
        delivered_idx = env.reward_manager.active_terms.index(
            "delivered_impulse"
        )
        gate_idx = (
            env.reward_manager.active_terms.index("r_gate")
            if "r_gate" in env.reward_manager.active_terms
            else None
        )
        progress_idx = (
            env.reward_manager.active_terms.index("r_waypoint_progress")
            if "r_waypoint_progress" in env.reward_manager.active_terms
            else None
        )

        self._env = env
        self._original_sim_step = env.sim.step
        self._original_compute_substep = env.metrics_manager.compute_substep
        self._original_compute = env.metrics_manager.compute

        def cache_preintegration_then_step(*args: Any, **kwargs: Any) -> Any:
            try:
                pre_qvel.copy_(robot.data.joint_vel[:, arm_ids])
                pre_depth.copy_(
                    nail.data.joint_pos[:, 0].clamp(0.0, NAIL_GOAL_DEPTH)
                )
                return self._original_sim_step(*args, **kwargs)
            except BaseException:
                self.uninstall()
                raise

        def capture_after_substep(*args: Any, **kwargs: Any) -> Any:
            try:
                result = self._original_compute_substep(*args, **kwargs)
                contact = (contact_sensor.data.found > 0).any(dim=-1)
                force = impulse_sensor.data.force
                force_axis = axis.to(dtype=force.dtype)
                axial_force = (
                    (force * force_axis).sum(dim=-1).sum(dim=-1).clamp_min(0.0)
                )
                self.capture_substep(
                    contact=contact,
                    axial_force=axial_force,
                    pre_arm_qvel=pre_qvel,
                    post_arm_qvel=robot.data.joint_vel[:, arm_ids],
                    pre_depth=pre_depth,
                    post_depth=nail.data.joint_pos[:, 0].clamp(
                        0.0, NAIL_GOAL_DEPTH
                    ),
                )
                return result
            except BaseException:
                self.uninstall()
                raise

        def capture_first_terminal(*args: Any, **kwargs: Any) -> Any:
            try:
                result = self._original_compute(*args, **kwargs)
                depth = nail.data.joint_pos[:, 0].clamp(
                    0.0, NAIL_GOAL_DEPTH
                )
                step_reward = env.reward_manager._step_reward
                if tracker is None:
                    tracker_started = tracker_false
                    tracker_finalized = tracker_false
                    tracker_productive = tracker_false
                    tracker_reason = tracker_reason_none
                    tracker_delivered = tracker_delivered_zero
                    tracker_contact_quality_valid = tracker_false
                    tracker_contact_quality_overflow = tracker_false
                else:
                    tracker_started = tracker.started
                    tracker_finalized = tracker.finalized
                    tracker_productive = tracker.productive
                    tracker_reason = tracker.reason
                    tracker_delivered = tracker.delivered
                    tracker_contact_quality_valid = tracker.contact_quality_valid
                    tracker_contact_quality_overflow = (
                        tracker.contact_quality_overflow
                    )
                if guideline_tracker is None:
                    guideline_initialized = tracker_false
                    guideline_entry = guideline_geometry_zero
                    guideline_nail = guideline_geometry_zero
                    guideline_next_gate = guideline_gate_zero
                    guideline_target_start_distance = guideline_scalar_zero
                    guideline_best_target_fraction = guideline_scalar_zero
                    guideline_window_new_credit = guideline_scalar_zero
                    guideline_episode_credit = guideline_scalar_zero
                    guideline_multi_gate_crossings = guideline_gate_zero
                else:
                    guideline_initialized = guideline_tracker.initialized
                    guideline_entry = guideline_tracker.entry
                    guideline_nail = guideline_tracker.nail
                    guideline_next_gate = guideline_tracker.next_gate
                    guideline_target_start_distance = (
                        guideline_tracker.target_start_distance
                    )
                    guideline_best_target_fraction = (
                        guideline_tracker.best_target_fraction
                    )
                    guideline_window_new_credit = guideline_tracker.window_new_credit
                    guideline_episode_credit = guideline_tracker.episode_credit
                    guideline_multi_gate_crossings = (
                        guideline_tracker.multi_gate_crossings
                    )
                self.capture_control_step(
                    reset_buf=env.reset_buf,
                    terminal_depth=depth,
                    terminal_success=depth >= NAIL_SUCCESS_THRESHOLD,
                    episode_lambda=accumulator._episode_peak_perjoint,
                    episode_delivered=delivered_accumulator.delivered,
                    tracker_exists=tracker_exists,
                    tracker_started=tracker_started,
                    tracker_finalized=tracker_finalized,
                    tracker_productive=tracker_productive,
                    tracker_reason=tracker_reason,
                    tracker_delivered=tracker_delivered,
                    tracker_contact_quality_valid=tracker_contact_quality_valid,
                    tracker_contact_quality_overflow=tracker_contact_quality_overflow,
                    guideline_tracker_exists=guideline_tracker_exists,
                    guideline_tracker_initialized=guideline_initialized,
                    guideline_entry=guideline_entry,
                    guideline_nail=guideline_nail,
                    guideline_next_gate=guideline_next_gate,
                    guideline_target_start_distance=guideline_target_start_distance,
                    guideline_best_target_fraction=guideline_best_target_fraction,
                    guideline_window_new_credit=guideline_window_new_credit,
                    guideline_episode_credit=guideline_episode_credit,
                    guideline_multi_gate_crossings=guideline_multi_gate_crossings,
                    manager_impact=step_reward[:, impact_idx] * env.step_dt,
                    manager_delivered=step_reward[:, delivered_idx] * env.step_dt,
                    manager_gate=(
                        step_reward[:, gate_idx] * env.step_dt
                        if gate_idx is not None
                        else manager_gate_zero
                    ),
                    manager_progress=(
                        step_reward[:, progress_idx] * env.step_dt
                        if progress_idx is not None
                        else manager_gate_zero
                    ),
                )
                return result
            except BaseException:
                self.uninstall()
                raise

        try:
            env.sim.step = cache_preintegration_then_step
            env.metrics_manager.compute_substep = capture_after_substep
            env.metrics_manager.compute = capture_first_terminal
        except BaseException:
            self.uninstall()
            raise

    @contextmanager
    def installed(self, env: Any):
        """Install for one rollout and restore callbacks on every exit path."""
        self.install(env)
        try:
            yield self
        finally:
            self.uninstall()

    def uninstall(self) -> None:
        if self._env is None:
            return
        self._env.sim.step = self._original_sim_step
        self._env.metrics_manager.compute_substep = (
            self._original_compute_substep
        )
        self._env.metrics_manager.compute = self._original_compute
        self._env = None
        self._original_sim_step = None
        self._original_compute_substep = None
        self._original_compute = None

    def finalize(self) -> dict[str, Any]:
        """Synchronize once, copy the first-terminal summary, and derive gates."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        tensor_fields = {
            name: value.detach().cpu()
            for name, value in vars(self).items()
            if name.startswith("terminal_") and isinstance(value, torch.Tensor)
        }
        records = {
            "lam": [row for row in tensor_fields["terminal_lambda"]],
            "succ": tensor_fields["terminal_success"].tolist(),
            "delivered": tensor_fields["terminal_delivered"].tolist(),
        }
        impossible, lambda_dead = eval_impulse._invariant_violations(records)
        seen = tensor_fields["terminal_seen"].bool()
        finite_signals = (
            seen
            & torch.isfinite(tensor_fields["terminal_depth"])
            & torch.isfinite(tensor_fields["terminal_lambda"]).all(dim=-1)
            & torch.isfinite(tensor_fields["terminal_delivered"])
            & torch.isfinite(tensor_fields["terminal_force_max"])
            & torch.isfinite(tensor_fields["terminal_pre_qvel_max"])
            & torch.isfinite(tensor_fields["terminal_post_qvel_max"])
            & torch.isfinite(tensor_fields["terminal_pre_depth_max"])
            & torch.isfinite(tensor_fields["terminal_post_depth_max"])
        )
        guideline_geometry_finite = (
            torch.isfinite(tensor_fields["terminal_guideline_entry"]).all(dim=-1)
            & torch.isfinite(tensor_fields["terminal_guideline_nail"]).all(dim=-1)
        )
        guideline_geometry_nondegenerate = (
            torch.linalg.vector_norm(
                tensor_fields["terminal_guideline_nail"]
                - tensor_fields["terminal_guideline_entry"],
                dim=-1,
            )
            > 0.0
        )
        guideline_progress_state_finite = (
            torch.isfinite(
                tensor_fields["terminal_guideline_target_start_distance"]
            )
            & torch.isfinite(
                tensor_fields["terminal_guideline_best_target_fraction"]
            )
            & torch.isfinite(
                tensor_fields["terminal_guideline_window_new_credit"]
            )
            & torch.isfinite(
                tensor_fields["terminal_guideline_episode_credit"]
            )
        )
        predicates = {
            "finite_signals": finite_signals,
            "hammer_nail_contact": seen
            & tensor_fields["terminal_contact_seen"],
            "normal_success": seen & tensor_fields["terminal_success"],
            "lambda_live": seen
            & (tensor_fields["terminal_lambda"].amax(dim=-1) > 0.0),
            "delivered_live": seen
            & (tensor_fields["terminal_delivered"] > 0.0),
            "raw_reward_readers_finite": (
                seen
                & tensor_fields["terminal_raw_impact_finite"]
                & tensor_fields["terminal_raw_delivered_finite"]
                & (tensor_fields["terminal_raw_impact_call_count"] > 0)
                & (tensor_fields["terminal_raw_delivered_call_count"] > 0)
            ),
            # Per-side raw-reader liveness (fq4x8): a zeroed side's weight is
            # 0.0, so mjlab's RewardManager never invokes its func at all
            # (see RewardManager.compute) -- raw_reward_readers_finite above
            # would then be permanently unsatisfiable. These let an active
            # side be checked independently of a structurally-skipped one.
            "raw_impact_reader_finite": (
                seen
                & tensor_fields["terminal_raw_impact_finite"]
                & (tensor_fields["terminal_raw_impact_call_count"] > 0)
            ),
            "raw_delivered_reader_finite": (
                seen
                & tensor_fields["terminal_raw_delivered_finite"]
                & (tensor_fields["terminal_raw_delivered_call_count"] > 0)
            ),
            "manager_impact_positive": seen
            & (tensor_fields["terminal_manager_impact"] > 0.0),
            "manager_delivered_positive": seen
            & (tensor_fields["terminal_manager_delivered"] > 0.0),
            # A weight-0 term must pay exactly zero at the terminal control
            # step, not merely "not positive" (which would also admit a NaN
            # or a negative value slipping through nan_to_num).
            "manager_impact_zero": seen
            & (tensor_fields["terminal_manager_impact"] == 0.0),
            "manager_delivered_zero": seen
            & (tensor_fields["terminal_manager_delivered"] == 0.0),
            "manager_gate_positive_finite": (
                seen
                & torch.isfinite(tensor_fields["terminal_manager_gate"])
                & (tensor_fields["terminal_manager_gate"] > 0.0)
            ),
            "manager_progress_positive_finite": (
                seen
                & torch.isfinite(tensor_fields["terminal_manager_progress"])
                & (tensor_fields["terminal_manager_progress"] > 0.0)
            ),
            # FQ-min only: the one-shot onset quality snapshot must be valid
            # and must not have overflowed its fixed contact-slot capacity.
            "quality_snapshot_valid": seen
            & tensor_fields["terminal_tracker_contact_quality_valid"],
            "quality_no_overflow": seen
            & ~tensor_fields["terminal_tracker_contact_quality_overflow"],
            "guideline_tracker_exists": seen
            & tensor_fields["terminal_guideline_tracker_exists"],
            "guideline_tracker_initialized": seen
            & tensor_fields["terminal_guideline_tracker_initialized"],
            "guideline_geometry_finite": seen & guideline_geometry_finite,
            "guideline_geometry_nondegenerate": (
                seen
                & guideline_geometry_finite
                & guideline_geometry_nondegenerate
            ),
            "guideline_all_gates_crossed": seen
            & (
                tensor_fields["terminal_guideline_next_gate"]
                == GUIDELINE_NUM_GATES
            ),
            "guideline_progress_state_finite": (
                seen
                & tensor_fields["terminal_guideline_tracker_exists"]
                & tensor_fields["terminal_guideline_tracker_initialized"]
                & guideline_progress_state_finite
            ),
            "hardware_qvel": (
                seen
                & (tensor_fields["terminal_pre_qvel_max"] <= 3.1415)
                & (tensor_fields["terminal_post_qvel_max"] <= 3.1415)
            ),
            "tracker_exists": seen
            & tensor_fields["terminal_tracker_exists"],
            "tracker_started": seen
            & tensor_fields["terminal_tracker_started"],
            "productive_success": seen
            & tensor_fields["terminal_tracker_productive"],
            "terminal_reason_success": seen
            & (tensor_fields["terminal_tracker_reason"] == 1),
            "delivered_event_impulse": seen
            & (tensor_fields["terminal_tracker_delivered"] > 0.0),
            "impact_no_early_pulse": seen
            & (
                tensor_fields["terminal_impact_first_positive_step"]
                == tensor_fields["terminal_control_step"]
            ),
            "delivered_no_early_pulse": seen
            & (
                tensor_fields["terminal_delivered_first_positive_step"]
                == tensor_fields["terminal_control_step"]
            ),
            "impact_exactly_one_terminal_pulse": seen
            & (tensor_fields["terminal_impact_positive_count"] == 1)
            & (
                tensor_fields["terminal_impact_last_positive_step"]
                == tensor_fields["terminal_control_step"]
            ),
            "delivered_exactly_one_terminal_pulse": seen
            & (tensor_fields["terminal_delivered_positive_count"] == 1)
            & (
                tensor_fields["terminal_delivered_last_positive_step"]
                == tensor_fields["terminal_control_step"]
            ),
        }
        predicate_counts = {
            name: {
                "passed": int(values.sum()),
                "total": self.num_envs,
                "failed": self.num_envs - int(values.sum()),
            }
            for name, values in predicates.items()
        }
        result: dict[str, Any] = {
            name: value.tolist() for name, value in tensor_fields.items()
        }
        result.update(
            {
                "predicate_counts": predicate_counts,
                "impossible_success_n": impossible,
                "lambda_dead_n": lambda_dead,
                "max_pre_arm_qvel_rad_s": float(
                    tensor_fields["terminal_pre_qvel_max"].amax()
                ),
                "max_post_arm_qvel_rad_s": float(
                    tensor_fields["terminal_post_qvel_max"].amax()
                ),
            }
        )
        return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_qvel_within_hardware_limit(value: Any) -> bool:
    return (
        _is_finite_number(value)
        and value <= 3.1415
    )


def _count_passes(value: Any, num_envs: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    passed = value.get("passed")
    total = value.get("total")
    if not _is_int(passed) or not _is_int(total):
        return False
    if passed < 0 or total < 0 or passed > total:
        return False
    failed = value.get("failed", total - passed)
    if not _is_int(failed) or failed != total - passed:
        return False
    return passed == num_envs and total == num_envs


def _count_is_well_formed(value: Any, num_envs: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    passed = value.get("passed")
    total = value.get("total")
    if not _is_int(passed) or not _is_int(total):
        return False
    if passed < 0 or total != num_envs or passed > total:
        return False
    failed = value.get("failed", total - passed)
    return _is_int(failed) and failed == total - passed


def _provenance_is_clean(provenance: Any, name: str) -> bool:
    if not isinstance(provenance, Mapping):
        return False
    entry = provenance.get(name)
    if not isinstance(entry, Mapping):
        return False
    canonical_path = entry.get("canonical_path")
    expected = entry.get("expected_revision")
    pre = entry.get("pre_revision")
    post = entry.get("post_revision")
    return (
        isinstance(canonical_path, str)
        and Path(canonical_path).is_absolute()
        and isinstance(expected, str)
        and _FULL_GIT_REVISION.fullmatch(expected) is not None
        and isinstance(pre, str)
        and _FULL_GIT_REVISION.fullmatch(pre) is not None
        and isinstance(post, str)
        and _FULL_GIT_REVISION.fullmatch(post) is not None
        and pre == expected
        and post == expected
        and entry.get("pre_dirty") is False
        and entry.get("post_dirty") is False
    )


def _configured_reader_name(func: Any) -> str:
    """Normalize class-valued and instantiated manager-term configurations."""
    return func.__name__ if inspect.isclass(func) else type(func).__name__


def _validate_guideline_smoke_contract(
    training_cfg: Any, contract: ArmContract
) -> dict[str, Any]:
    """Adapt the frozen C0/C-Gate validator to the shared progress-state observation."""
    compatible_cfg = copy.deepcopy(training_cfg)
    for group_name in ("actor", "critic"):
        compatible_cfg.observations[group_name].terms.pop(
            "waypoint_progress_state", None
        )

    progress = compatible_cfg.rewards.pop("r_waypoint_progress", None)
    if contract.progress_reward_required:
        if (
            progress is None
            or progress.func is not ordered_waypoint_progress_reward
            or float(progress.weight) != 8.0
            or progress.params != {}
        ):
            raise ValueError(
                f"{contract.task}: P r_waypoint_progress must be "
                "ordered_waypoint_progress_reward, weight 8.0, with empty params"
            )
        validator_task = GUIDELINE_ARM_TASKS["C0"]
    else:
        if progress is not None:
            raise ValueError(f"{contract.task}: {contract.arm} must not contain r_waypoint_progress")
        validator_task = contract.task

    digest = dict(
        eval_impulse._validate_native_guideline_env_contract(
            compatible_cfg, validator_task
        )
    )
    digest["treatment"] = contract.arm
    digest["progress_reward_enabled"] = contract.progress_reward_required
    return digest


def validate_live_contract(task: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load and fail closed on the unmodified registered task configuration."""
    if task not in TASK_CONTRACTS:
        raise ValueError(f"unknown smoke task {task!r}")

    training_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)
    contract = TASK_CONTRACTS[task]
    if contract.guideline_required:
        digest = _validate_guideline_smoke_contract(training_cfg, contract)
    else:
        digest = dict(
            eval_impulse._validate_sampled_env_contract(training_cfg, task)
        )
    impact = training_cfg.rewards["impact_progress"]
    delivered = training_cfg.rewards["delivered_impulse"]

    if _configured_reader_name(impact.func) != contract.impact_reader:
        raise ValueError(f"{task}: impact reader drift")
    if _configured_reader_name(delivered.func) != contract.delivered_reader:
        raise ValueError(f"{task}: delivered reader drift")
    if float(delivered.params["i_ref"]) != contract.event_i_ref_n_s:
        raise ValueError(f"{task}: delivered i_ref drift")
    if (
        contract.delivered_saturate is not None
        and delivered.params.get("saturate") is not contract.delivered_saturate
    ):
        raise ValueError(f"{task}: delivered saturate drift")
    if contract.impact_v_expected_n_s is not None and float(
        impact.params.get("v_expected", float("nan"))
    ) != contract.impact_v_expected_n_s:
        raise ValueError(f"{task}: impact v_expected drift")

    if contract.tracker_required:
        tracker = training_cfg.metrics.get("first_strike")
        if tracker is None or tracker.func is not FirstStrikeEventTracker:
            raise ValueError(f"{task}: missing or wrong FirstStrikeEventTracker")
        if tracker.per_substep is not True or tracker.reduce != "last":
            raise ValueError(f"{task}: FirstStrikeEventTracker timing drift")
        if contract.quality_required:
            if tracker.params.get("quality_sensor_name") != "hammer_nail_quality":
                raise ValueError(f"{task}: first-strike quality sensor drift")
            quality_sensors = [
                sensor for sensor in (training_cfg.scene.sensors or ())
                if sensor.name == "hammer_nail_quality"
            ]
            if len(quality_sensors) != 1 or quality_sensors[0].num_slots != 8:
                raise ValueError(f"{task}: missing or wrong quality sensor")

    live_limits = tuple(
        float(value) for value in training_cfg.metrics["cat_soft"].params["imp_limit"]
    )
    imported_limits = tuple(float(value) for value in IMP_J_LIMIT)
    if live_limits != LITERAL_IMPULSE_LIMITS_N_M_S:
        raise ValueError(f"{task}: impulse limits differ from manufacturer caps")
    if imported_limits != LITERAL_IMPULSE_LIMITS_N_M_S:
        raise ValueError(f"{task}: imported IMP_J_LIMIT differs from manufacturer caps")
    if float(agent_cfg.clip_actions) != 1.0:
        raise ValueError(f"{task}: RL clip_actions must be 1.0")

    row_diagnostic = training_cfg.metrics.get("substep_impulse_rows")
    if row_diagnostic is None or row_diagnostic.params.get("enabled") is not True:
        raise ValueError(f"{task}: registered substep impulse rows must be enabled")

    digest.update(
        {
            "imp_max_p": float(training_cfg.metrics["cat_soft"].params["imp_max_p"]),
            "impulse_limits_n_m_s": list(live_limits),
            "rl_clip_actions": float(agent_cfg.clip_actions),
            "registered_substep_impulse_rows_enabled": True,
        }
    )
    return training_cfg, agent_cfg, digest


def make_diagnostic_cfg(training_cfg: Any, num_envs: int) -> Any:
    """Copy a validated training config and apply only safe smoke overrides."""
    cfg = copy.deepcopy(training_cfg)
    cfg.scene.num_envs = num_envs
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.metrics["substep_impulse_rows"].params["enabled"] = False
    return cfg


def _git_output(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read Git provenance for {repo}") from error
    return result.stdout.strip()


def read_repo_provenance(path: Path, expected_revision: str) -> dict[str, Any]:
    """Return a clean, expected full-revision snapshot for one repository."""
    if not isinstance(expected_revision, str) or not _FULL_GIT_REVISION.fullmatch(
        expected_revision
    ):
        raise ValueError("expected revision must be full 40-character lowercase hexadecimal")
    canonical_path = Path(path).resolve()
    repository_root = Path(
        _git_output(canonical_path, "rev-parse", "--show-toplevel")
    ).resolve()
    if canonical_path != repository_root:
        raise ValueError("provenance path must equal the canonical repository root")
    observed_revision = _git_output(canonical_path, "rev-parse", "HEAD")
    if observed_revision != expected_revision:
        raise ValueError("repository revision differs from expected revision")
    if _git_output(canonical_path, "status", "--porcelain"):
        raise ValueError("repository is dirty")
    return {
        "canonical_path": str(canonical_path),
        "expected_revision": expected_revision,
        "revision": observed_revision,
        "dirty": False,
    }


def _is_descendant(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_path(out: Path, code_repo: Path, asset_repo: Path) -> Path:
    """Canonicalize an artifact path and reject either source repository."""
    output_path = Path(out)
    canonical_out = output_path.parent.resolve() / output_path.name
    canonical_code_repo = Path(code_repo).resolve()
    canonical_asset_repo = Path(asset_repo).resolve()
    if _is_descendant(canonical_out, canonical_code_repo) or _is_descendant(
        canonical_out, canonical_asset_repo
    ):
        raise ValueError("output path must be outside both repositories")
    return canonical_out


def write_json_atomic(out: Path, payload: Mapping[str, Any]) -> None:
    """Create one JSON artifact through a flushed, fsynced temporary sibling."""
    destination = Path(out)
    if os.path.lexists(destination):
        raise FileExistsError(destination)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, destination)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the frozen one-task smoke invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=tuple(TASK_CONTRACTS))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--expected-asset-revision", required=True)
    parser.add_argument("--asset-repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.device == "cuda:0" and args.num_envs != 256:
        parser.error("CUDA qualification requires --num-envs 256")
    for option, value in (
        ("--expected-code-revision", args.expected_code_revision),
        ("--expected-asset-revision", args.expected_asset_revision),
    ):
        if _FULL_GIT_REVISION.fullmatch(value) is None:
            parser.error(f"{option} must be a full lowercase Git revision")
    return args


def _unbound_provenance() -> dict[str, dict[str, Any]]:
    """Placeholder used only by in-memory integration; the CLI replaces it."""
    revision = "0" * 40
    return {
        name: {
            "canonical_path": f"/unbound/{name}",
            "expected_revision": revision,
            "pre_revision": revision,
            "post_revision": revision,
            "pre_dirty": False,
            "post_dirty": False,
        }
        for name in ("code", "assets")
    }


def _physical_device_identity(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    details = [name]
    for attribute in ("uuid", "pci_bus_id"):
        value = getattr(properties, attribute, None)
        if value:
            details.append(f"{attribute}={value}")
    return "; ".join(details)


def _run_fixed_rollout(
    wrapped_env: Any,
    reference: Any,
    *,
    head_position: Any,
    playback_length: int,
) -> None:
    """Run the fixed device-only playback plus final-target hold."""
    with torch.inference_mode():
        for control_step in range(1, playback_length + 7):
            target = reference.playback_target(
                min(control_step, playback_length)
            )
            action = (
                (target - head_position()) / 0.15
            ).clamp(-1.0, 1.0)
            wrapped_env.step(action)


def run_smoke(task: str, device: str, num_envs: int) -> dict[str, Any]:
    """Run the deterministic reference and return an in-memory gated record."""
    if task not in TASK_CONTRACTS:
        raise ValueError(f"unknown smoke task {task!r}")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and (
        requested_device.index not in (None, 0) or num_envs != 256
    ):
        raise ValueError("CUDA smoke requires cuda:0 and exactly 256 environments")

    training_cfg, agent_cfg, contract_digest = validate_live_contract(task)
    diagnostic_cfg = make_diagnostic_cfg(training_cfg, num_envs)
    raw_env: Any | None = None
    wrapped_env: Any | None = None
    try:
        raw_env = ManagerBasedRlEnv(
            cfg=diagnostic_cfg,
            device=device,
            render_mode=None,
        )
        actual_device = torch.device(raw_env.device)
        if requested_device.type == "cuda" and actual_device != torch.device("cuda:0"):
            raise RuntimeError(
                f"requested cuda:0 but environment is on {actual_device}"
            )

        impact_tap, delivered_tap = install_reward_taps(raw_env)
        recorder = DeviceSmokeRecorder(impact_tap, delivered_tap)
        head_cfg = SceneEntityCfg(
            "robot", site_names=(HAMMER_HEAD_SITE_NAME,)
        )
        nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
        head_cfg.resolve(raw_env.scene)
        nail_cfg.resolve(raw_env.scene)
        robot = raw_env.scene["robot"]
        nail = raw_env.scene["nail_block"]

        def head_position() -> torch.Tensor:
            return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

        def nail_top_position() -> torch.Tensor:
            return nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

        with recorder.installed(raw_env):
            wrapped_env = RslRlVecEnvWrapper(
                raw_env, clip_actions=float(agent_cfg.clip_actions)
            )
            reference = SingleStrikeReference(num_envs, raw_env.device)
            reference.update(
                head_position(),
                nail_top_position(),
                torch.zeros(
                    num_envs, dtype=torch.long, device=actual_device
                ),
            )
            playback_length = reference.playback_length()
            _run_fixed_rollout(
                wrapped_env,
                reference,
                head_position=head_position,
                playback_length=playback_length,
            )

            runtime = recorder.finalize()

        contract = TASK_CONTRACTS[task]
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "arm": contract.arm,
            "task": task,
            "requested_device": str(requested_device),
            "device_type": requested_device.type,
            "num_envs": num_envs,
            "visible_cuda_device_count": (
                torch.cuda.device_count()
                if requested_device.type == "cuda"
                else 0
            ),
            "actual_tensor_device": str(recorder.terminal_seen.device),
            "environment_device": str(actual_device),
            "physical_device_identity": _physical_device_identity(
                actual_device
            ),
            "contract_digest": contract_digest,
            "scripted_reference": {
                "class": "SingleStrikeReference",
                "geometry": "reset_head_to_below_nail_follow_through",
                "follow_through_overshoot_m": reference.overshoot,
                "delta_pos_scale_m": 0.15,
                "playback_length_steps": playback_length,
                "hold_steps": 6,
            },
            "provenance": _unbound_provenance(),
        }
        record.update(runtime)
        return evaluate_gate(record)
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif raw_env is not None:
            raw_env.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one provenance-bound smoke invocation."""
    args = parse_args(argv)
    contract = TASK_CONTRACTS[args.task]
    code_repo = Path(__file__).resolve().parents[1]

    def fail(stage: str, error: Exception) -> int:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "arm": contract.arm,
                    "task": args.task,
                    "integration_pass": False,
                    "cuda_qualification_pass": False,
                    "failed_predicates": [f"{stage}_exception"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1

    try:
        output_path = validate_output_path(
            args.out, code_repo, args.asset_repo
        )
        if os.path.lexists(output_path):
            raise FileExistsError(output_path)
    except Exception as error:
        return fail("output_validation", error)

    try:
        pre_code = read_repo_provenance(
            code_repo, args.expected_code_revision
        )
        pre_assets = read_repo_provenance(
            args.asset_repo, args.expected_asset_revision
        )
    except Exception as error:
        return fail("pre_provenance", error)

    try:
        runtime = run_smoke(
            task=args.task,
            device=args.device,
            num_envs=args.num_envs,
        )
    except Exception as error:
        return fail("runtime", error)

    try:
        post_code = read_repo_provenance(
            code_repo, args.expected_code_revision
        )
        post_assets = read_repo_provenance(
            args.asset_repo, args.expected_asset_revision
        )
    except Exception as error:
        return fail("post_provenance", error)

    runtime["provenance"] = {
        "code": {
            "canonical_path": pre_code["canonical_path"],
            "expected_revision": args.expected_code_revision,
            "pre_revision": pre_code["revision"],
            "post_revision": post_code["revision"],
            "pre_dirty": pre_code["dirty"],
            "post_dirty": post_code["dirty"],
        },
        "assets": {
            "canonical_path": pre_assets["canonical_path"],
            "expected_revision": args.expected_asset_revision,
            "pre_revision": pre_assets["revision"],
            "post_revision": post_assets["revision"],
            "pre_dirty": pre_assets["dirty"],
            "post_dirty": post_assets["dirty"],
        },
    }
    result = evaluate_gate(runtime)
    summary = {
        "schema_version": result["schema_version"],
        "arm": result["arm"],
        "task": result["task"],
        "integration_pass": result["integration_pass"],
        "cuda_qualification_pass": result["cuda_qualification_pass"],
        "failed_predicates": result["failed_predicates"],
    }
    try:
        write_json_atomic(output_path, result)
    except Exception as error:
        return fail("output_publish", error)
    print(json.dumps(summary, sort_keys=True))
    if args.device == "cuda:0":
        return 0 if result["cuda_qualification_pass"] else 1
    return 0 if result["integration_pass"] else 1


def evaluate_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a JSON-like smoke record without touching runtime resources."""
    result = dict(record)
    failures: set[str] = set()

    if record.get("schema_version") != SCHEMA_VERSION:
        failures.add("schema_version")

    arm = record.get("arm")
    task = record.get("task")
    contract = ARM_CONTRACTS.get(arm) if isinstance(arm, str) else None
    if contract is None or task != contract.task or TASK_CONTRACTS.get(task) != contract:
        failures.add("task_arm_pairing")

    num_envs = record.get("num_envs")
    if not _is_int(num_envs) or num_envs <= 0:
        failures.add("num_envs")
        num_envs = 0

    predicate_counts = record.get("predicate_counts")
    if not isinstance(predicate_counts, Mapping):
        predicate_counts = {}
    for predicate in INSTRUMENTATION_PREDICATES:
        if not _count_passes(predicate_counts.get(predicate), num_envs):
            failures.add(predicate)

    # Reward-manager payout predicates are arm-aware (fq4x8): a side whose
    # weight is contractually zeroed structurally never invokes its raw
    # reader (mjlab skips weight==0 terms) and must pay exactly zero, not
    # merely "not positive". contract is None only when arm/task itself is
    # invalid, already recorded above as task_arm_pairing -- treat that like
    # every pre-fq4x8 arm (both sides raw) rather than skip checking.
    speed_zero = contract.speed_zero if contract is not None else False
    delivered_zero = contract.delivered_zero if contract is not None else False
    impact_payout_predicate = (
        "manager_impact_zero" if speed_zero else "manager_impact_positive"
    )
    if not _count_passes(predicate_counts.get(impact_payout_predicate), num_envs):
        failures.add(impact_payout_predicate)
    delivered_payout_predicate = (
        "manager_delivered_zero" if delivered_zero else "manager_delivered_positive"
    )
    if not _count_passes(predicate_counts.get(delivered_payout_predicate), num_envs):
        failures.add(delivered_payout_predicate)
    if not speed_zero and not delivered_zero:
        if not _count_passes(
            predicate_counts.get("raw_reward_readers_finite"), num_envs
        ):
            failures.add("raw_reward_readers_finite")
    else:
        if not speed_zero and not _count_passes(
            predicate_counts.get("raw_impact_reader_finite"), num_envs
        ):
            failures.add("raw_impact_reader_finite")
        if not delivered_zero and not _count_passes(
            predicate_counts.get("raw_delivered_reader_finite"), num_envs
        ):
            failures.add("raw_delivered_reader_finite")
    if contract is not None and contract.quality_required:
        for predicate in QUALITY_PREDICATES:
            if not _count_passes(predicate_counts.get(predicate), num_envs):
                failures.add(predicate)
    if contract is not None and contract.guideline_required:
        for predicate in GUIDELINE_PREDICATES:
            if not _count_passes(predicate_counts.get(predicate), num_envs):
                failures.add(predicate)
    if contract is not None and contract.gate_reward_required:
        if not _count_passes(
            predicate_counts.get("manager_gate_positive_finite"), num_envs
        ):
            failures.add("manager_gate_positive_finite")
    if contract is not None and contract.progress_reward_required:
        if not _count_passes(
            predicate_counts.get("manager_progress_positive_finite"), num_envs
        ):
            failures.add("manager_progress_positive_finite")

    hardware_count_valid = _count_is_well_formed(
        predicate_counts.get("hardware_qvel"), num_envs
    )
    if not hardware_count_valid:
        failures.add("hardware_qvel_schema")
    pre_qvel_finite = _is_finite_number(
        record.get("max_pre_arm_qvel_rad_s")
    )
    post_qvel_finite = _is_finite_number(
        record.get("max_post_arm_qvel_rad_s")
    )
    if not pre_qvel_finite or not post_qvel_finite:
        failures.add("qvel_nonfinite")
    hardware_transport_qualified = (
        hardware_count_valid
        and _count_passes(predicate_counts.get("hardware_qvel"), num_envs)
        and _is_qvel_within_hardware_limit(
            record.get("max_pre_arm_qvel_rad_s")
        )
        and _is_qvel_within_hardware_limit(
            record.get("max_post_arm_qvel_rad_s")
        )
    )
    if not pre_qvel_finite or not post_qvel_finite:
        hardware_transport_label = "nonfinite_qvel_invalid"
    elif not hardware_count_valid:
        hardware_transport_label = "hardware_transport_evidence_invalid"
    elif hardware_transport_qualified:
        hardware_transport_label = "within_observed_qvel_rail"
    else:
        hardware_transport_label = (
            "finite_qvel_rail_exceedance_simulation_only"
        )
    if contract is not None and contract.tracker_required:
        for predicate in TRACKER_COMMON_PREDICATES:
            if not _count_passes(predicate_counts.get(predicate), num_envs):
                failures.add(predicate)
        # A zeroed side's reward-manager func is never invoked, so its pulse
        # timing never advances -- only require pulse timing from a side
        # this arm actually pays.
        if not speed_zero:
            for predicate in TRACKER_SPEED_PULSE_PREDICATES:
                if not _count_passes(predicate_counts.get(predicate), num_envs):
                    failures.add(predicate)
        if not delivered_zero:
            for predicate in TRACKER_DELIVERED_PULSE_PREDICATES:
                if not _count_passes(predicate_counts.get(predicate), num_envs):
                    failures.add(predicate)

    for sentinel in ("impossible_success_n", "lambda_dead_n"):
        if not _is_int(record.get(sentinel)) or record.get(sentinel) != 0:
            failures.add(sentinel)

    provenance = record.get("provenance")
    if not _provenance_is_clean(provenance, "code"):
        failures.add("code_provenance")
    if not _provenance_is_clean(provenance, "assets"):
        failures.add("assets_provenance")

    integration_pass = not failures

    cuda_failures: set[str] = set()
    if record.get("device_type") != "cuda":
        cuda_failures.add("device_type")
    if not _is_int(record.get("visible_cuda_device_count")) or record.get(
        "visible_cuda_device_count"
    ) != 1:
        cuda_failures.add("visible_cuda_device_count")
    if record.get("actual_tensor_device") != "cuda:0":
        cuda_failures.add("actual_tensor_device")
    if record.get("environment_device") != "cuda:0":
        cuda_failures.add("environment_device")
    identity = record.get("physical_device_identity")
    if not isinstance(identity, str) or not identity:
        cuda_failures.add("physical_device_identity")
    if num_envs != 256:
        cuda_failures.add("cuda_num_envs")

    failures.update(cuda_failures)
    result["predicate_counts"] = record.get("predicate_counts")
    result["hardware_transport_qualified"] = hardware_transport_qualified
    result["hardware_transport_label"] = hardware_transport_label
    result["integration_pass"] = integration_pass
    result["cuda_qualification_pass"] = integration_pass and not cuda_failures
    result["failed_predicates"] = sorted(failures)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

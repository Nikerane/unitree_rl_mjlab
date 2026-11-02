"""Read-only pre-reset physical snapshots shared by impact-speed evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def physical_snapshot(env, *, include_cat_dose_audit=False):
    """Read reward-time channels at policy-step cadence without mutating task state."""
    from src.tasks.hammer.mdp.rewards import clamped_nail_depth
    from src.tasks.hammer.mdp.trackability import joint_target_rmse

    tracker = env._hammer_first_strike
    robot_cfg = env.observation_manager.get_term_cfg("actor", "strike_ref_error").params["robot_cfg"]
    head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    reference = env._strike_reference
    phase = reference.preview(head, env.episode_length_buf)
    _, _, geometric_error = reference.diagnose_current_height(head)
    joint_cfg = env.metrics_manager.cfg["substep_impulse"].params["robot_cfg"]
    nail_cfg = env.observation_manager.get_term_cfg("actor", "nail_depth").params["asset_cfg"]
    hook = next(cfg.func for name, cfg in zip(env.metrics_manager.active_terms, env.metrics_manager._term_cfgs)
                if name == "cat_soft")
    cat = hook.constraint_telemetry()
    gains = env.action_manager.get_term("joint_stiffness").telemetry
    speed_state = env.action_manager.get_term("strike_speed").state
    sampled = getattr(env, "_sampled_pd500", None)
    if sampled is not None:
        if sampled.applied_kp is None or sampled.applied_kd is None:
            raise ValueError("sampled controller has no applied gains after a physics step")
        applied_kp, applied_kd = sampled.applied_kp, sampled.applied_kd
    else:
        applied_kp, applied_kd = gains.kp, gains.kd
    policy_reference_error = torch.linalg.vector_norm(reference.waypoint(phase) - head, dim=-1)
    snapshot = {
        "path_error_m": policy_reference_error,
        "policy_reference_error_m": policy_reference_error,
        "geometric_path_error_m": geometric_error,
        "joint_target_error_rad": joint_target_rmse(env, joint_cfg),
        "route_sign": env._strike_reference.route_signs(),
        # One clamped sample per policy transition; this is not a physics-rate maximum.
        "nail_depth_m": clamped_nail_depth(env, nail_cfg),
        "started": tracker.started,
        "first_contact_time_s": tracker.first_contact_time_s,
        "delivered_impulse_ns": tracker.delivered,
        "joint_impulse_nms": cat["lambda_per_joint"],
        "impulse_exceeded": cat["raw_margin_per_joint"] > 0,
        "cat_delta": cat["delta"],
        "kp": applied_kp,
        "kd": applied_kd,
        "desired_speed_mps": speed_state.goal_mps,
        "environment_reward": env.reward_manager._reward_buf,
        "timeout": env.reset_time_outs,
    }
    if include_cat_dose_audit:
        velocity_margin = hook._cat.raw_constraints.get("joint_velocity_excess")
        velocity_normalizer = hook._cat.running_maxes.get("joint_velocity_excess")
        impulse_margin = cat["raw_margin_per_joint"]
        impulse_cmax = hook._imp_cmax
        impulse_seeded = hook._imp_seeded
        if (velocity_margin is None or velocity_normalizer is None
                or velocity_margin.shape != cat["delta_velocity_per_joint"].shape
                or velocity_normalizer.shape != (1, velocity_margin.shape[1])):
            raise ValueError("CaT audit requires raw velocity margin and post-update EMA evidence")
        if (impulse_margin.shape != cat["lambda_per_joint"].shape
                or impulse_cmax.shape != (1, impulse_margin.shape[1])
                or impulse_seeded.shape != impulse_cmax.shape):
            raise ValueError("CaT audit requires aligned impulse margin, cmax, and seeded state")
        positive_impulse_margin = impulse_margin.clamp_min(0.)
        impulse_batch_max, impulse_batch_max_world = positive_impulse_margin.max(dim=0)
        snapshot.update({
            "cat_velocity_margin_per_joint": velocity_margin.detach().clone(),
            # Repeat the shared post-update population vector for the per-world snapshot schema.
            "cat_velocity_normalizer_per_joint": velocity_normalizer.expand_as(velocity_margin).detach().clone(),
            "cat_impulse_margin_per_joint": impulse_margin.detach().clone(),
            "cat_impulse_cmax_per_joint": impulse_cmax.squeeze(0).detach().clone(),
            "cat_impulse_seeded_per_joint": impulse_seeded.squeeze(0).detach().clone(),
            "cat_impulse_batch_max_margin_per_joint": impulse_batch_max.detach().clone(),
            # argmax returns the first world on ties; the index is meaningful only for a positive max.
            "cat_impulse_batch_max_world_index_per_joint": impulse_batch_max_world.detach().clone(),
            "cat_impulse_batch_max_violated_per_joint": (impulse_batch_max > 0.).detach().clone(),
            "delta_velocity_per_joint": cat["delta_velocity_per_joint"],
            "delta_impulse_per_joint": cat["delta_impulse_per_joint"],
            "cat_winner_constraint": cat["winner_constraint"],
            "cat_winner_joint": cat["winner_joint"],
        })
    return snapshot


class PreResetSnapshot:
    """Observe evaluator metrics after their normal computation, before native autoreset."""

    def __init__(self, env, reader=physical_snapshot):
        self.env, self.reader, self.latest = env, reader, None
        self.latest_evaluation = None

    def __enter__(self):
        self.original = self.env.metrics_manager.compute

        def compute():
            result = self.original()
            self.latest = {key: value.detach().clone() for key, value in self.reader(self.env).items()}
            if hasattr(self.env, "_strike_evaluation"):
                from src.tasks.hammer.mdp.strike_evaluation import evaluation_snapshot
                self.latest_evaluation = evaluation_snapshot(self.env)
            return result

        self.env.metrics_manager.compute = compute
        return self

    def __exit__(self, *exc):
        self.env.metrics_manager.compute = self.original


class FirstActionSnapshot:
    """Record reset state and the first delivered command at the integration seam."""

    _TORQUE_FIELDS = (
        ("requested_torque", "reset_last_requested_torque_nm", "first_requested_torque_nm"),
        ("torque", "reset_last_applied_torque_nm", "first_applied_torque_nm"),
        ("saturated", "reset_last_torque_saturated", "first_torque_saturated"),
    )

    def __init__(self, env):
        controller = getattr(env, "_sampled_pd500", None)
        if controller is None:
            raise ValueError("first-action snapshot requires the sampled PD controller")
        if not callable(getattr(controller, "original_step", None)):
            raise ValueError("sampled PD controller has no original integration step")
        self.env = env
        self.controller = controller
        self._original_step = None
        self._reset = None
        self._action = None
        self._first_command = None

    def __enter__(self):
        if self._original_step is not None:
            raise RuntimeError("first-action snapshot is already installed")
        self._reset = self._capture_reset_state()
        self._original_step = self.controller.original_step

        def capture_before_integration(*args, **kwargs):
            if self._first_command is None:
                if self._action is None:
                    raise RuntimeError("first policy action must be recorded before integration")
                self._first_command = self._capture_first_command()
            return self._original_step(*args, **kwargs)

        self.controller.original_step = capture_before_integration
        return self

    def __exit__(self, *exc):
        if self._original_step is not None:
            self.controller.original_step = self._original_step
            self._original_step = None

    def set_action(self, action: torch.Tensor) -> None:
        if self._original_step is None:
            raise RuntimeError("first-action snapshot must be installed before setting its action")
        if self._action is not None:
            raise RuntimeError("first policy action has already been recorded")
        if (not isinstance(action, torch.Tensor) or action.ndim < 2
                or action.shape[0] != self.env.num_envs
                or not torch.is_floating_point(action)
                or not bool(torch.isfinite(action).all())):
            raise ValueError("first policy action must be a finite floating tensor with one row per world")
        self._action = action.detach().cpu().clone()

    def _joint_state(self):
        controller = self.controller
        ids = controller.ids
        data = self.env.sim.data
        return {
            "joint_position_rad": data.qpos[:, controller.qids][:, ids].detach().cpu().clone(),
            "joint_velocity_rad_s": data.qvel[:, controller.vids][:, ids].detach().cpu().clone(),
            "ctrl_target_rad": data.ctrl[:, ids].detach().cpu().clone(),
        }

    def _optional_controller_snapshot(self, value, *, canonical: bool = False, dtype=None):
        if value is None:
            shape = (self.env.num_envs, 6)
            dtype = self.env.sim.data.qpos.dtype if dtype is None else dtype
            return torch.zeros(shape, dtype=dtype), False
        ordered = value if canonical else value[:, self.controller.ids]
        return ordered.detach().cpu().clone(), True

    def _capture_reset_state(self) -> dict[str, torch.Tensor]:
        controller = self.controller
        gains = controller.gains.telemetry
        state = self._joint_state()
        output = {
            "reset_joint_position_rad": state["joint_position_rad"],
            "reset_joint_velocity_rad_s": state["joint_velocity_rad_s"],
            "reset_ctrl_target_rad": state["ctrl_target_rad"],
            "reset_stiffness_action_raw": controller.gains.raw_action.detach().cpu().clone(),
            "reset_stiffness_coordinate": gains.p.detach().cpu().clone(),
            "reset_stiffness_kp": gains.kp.detach().cpu().clone(),
            "reset_stiffness_kd": gains.kd.detach().cpu().clone(),
            "reset_physics_time_s": torch.full(
                (self.env.num_envs,), controller.count * self.env.physics_dt,
                dtype=state["joint_position_rad"].dtype,
            ),
            "reset_controller_count": torch.full(
                (self.env.num_envs,), controller.count, dtype=torch.long,
            ),
            "reset_controller_updates": torch.full(
                (self.env.num_envs,), controller.updates, dtype=torch.long,
            ),
        }
        valid_snapshots = []
        for field, reset_name, _first_name in self._TORQUE_FIELDS:
            value, valid = self._optional_controller_snapshot(
                getattr(controller, field),
                dtype=torch.bool if field == "saturated" else None,
            )
            output[reset_name] = value
            valid_snapshots.append(valid)
        for field, suffix in (("applied_kp", "kp"), ("applied_kd", "kd")):
            value, valid = self._optional_controller_snapshot(
                getattr(controller, field), canonical=True,
            )
            output[f"reset_last_applied_{suffix}"] = value
            valid_snapshots.append(valid)
        output["reset_prior_controller_sample_valid"] = torch.full(
            (self.env.num_envs,), all(valid_snapshots), dtype=torch.bool,
        )
        return output

    def _capture_first_command(self) -> dict[str, torch.Tensor]:
        controller = self.controller
        state = self._joint_state()
        if any(getattr(controller, field) is None for field in
               ("applied_kp", "applied_kd", "requested_torque", "torque", "saturated")):
            raise RuntimeError("sampled controller has no complete first torque snapshot")
        output = {
            "first_action_raw": self._action.clone(),
            "first_preintegration_joint_position_rad": state["joint_position_rad"],
            "first_preintegration_joint_velocity_rad_s": state["joint_velocity_rad_s"],
            "first_delivered_joint_target_rad": state["ctrl_target_rad"],
            "first_applied_kp": controller.applied_kp.detach().cpu().clone(),
            "first_applied_kd": controller.applied_kd.detach().cpu().clone(),
            "first_physics_time_s": torch.full(
                (self.env.num_envs,), controller.count * self.env.physics_dt,
                dtype=state["joint_position_rad"].dtype,
            ),
            "first_controller_count": torch.full(
                (self.env.num_envs,), controller.count, dtype=torch.long,
            ),
            "first_controller_updates": torch.full(
                (self.env.num_envs,), controller.updates, dtype=torch.long,
            ),
        }
        for field, _reset_name, first_name in self._TORQUE_FIELDS:
            value, _valid = self._optional_controller_snapshot(
                getattr(controller, field),
            )
            output[first_name] = value
        return output

    def tensors(self) -> dict[str, torch.Tensor]:
        if self._reset is None or self._first_command is None:
            raise RuntimeError("first-action snapshot is incomplete")
        return {**self._reset, **self._first_command}

    def metadata(self) -> dict:
        names = tuple(self.controller.gains.joint_names)
        if len(names) != 6:
            raise ValueError("first-action snapshot requires six canonical arm-joint names")
        return {
            "schema_version": "first_action_snapshot_v1",
            "joint_names": list(names),
            "physics_dt_s": float(self.env.physics_dt),
            "policy_dt_s": float(self.env.step_dt),
            "reset_state_semantics": "post-reset, before the first policy action and physics integration",
            "first_command_semantics": "after action processing and torque calculation, before first physics integration",
            "joint_state_semantics": "native qpos and qvel before integration, ordered by canonical arm joint",
            "command_semantics": "ctrl target and sampled torque/gains for the first action, ordered by canonical arm joint",
            "policy_action_semantics": "exact raw first action from the saved action tape; cloned without modification",
        }

    def save(self, directory: Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensor_path = directory / "first-action-snapshot.pt"
        metadata_path = directory / "first-action-snapshot.json"
        with tensor_path.open("xb") as stream:
            torch.save(self.tensors(), stream)
        with metadata_path.open("x") as stream:
            json.dump(self.metadata(), stream, indent=2)
            stream.write("\n")
        return tensor_path, metadata_path

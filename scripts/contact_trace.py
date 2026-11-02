"""Read-only, first-episode physics trace for impact-speed evaluation.

The metrics manager calls this observer after its per-substep terms. Joint state
and controller commands are read after integration; site state is the native
scene view used by the strike tracker and can lag that integration by one
physics sample. No simulator forward call is made to remove that offset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH


class ContactTrace:
    """Observe first episodes without changing the environment's state."""

    def __init__(self, env, *, max_policy_steps: int | None = None,
                 post_onset_horizon_s: float | None = None,
                 hard_cap_s: float | None = None):
        controller = getattr(env, "_sampled_pd500", None)
        if controller is None:
            raise ValueError("contact trace requires the sampled PD controller")
        self.env = env
        self.controller = controller
        nail_joint_id = env.sim.mj_model.joint("nail_block/nail_slide").id
        self.nail_qpos_id = int(env.sim.mj_model.jnt_qposadr[nail_joint_id])
        self.nail_dof_id = int(env.sim.mj_model.jnt_dofadr[nail_joint_id])
        self.tracker = env._hammer_first_strike
        self.dt = float(env.physics_dt)
        self.max_samples = (env.max_episode_length + 1 if max_policy_steps is None
                            else max_policy_steps) * env.cfg.decimation
        if self.max_samples <= 0:
            raise ValueError("contact trace requires a positive evaluation horizon")
        if (post_onset_horizon_s is None) != (hard_cap_s is None):
            raise ValueError("post-onset shadow requires both a horizon and hard cap")
        self.shadow_window = (
            None if post_onset_horizon_s is None else PostOnsetTraceWindow(
                num_envs=env.num_envs, device=env.device, physics_dt_s=self.dt,
                post_onset_horizon_s=post_onset_horizon_s, hard_cap_s=hard_cap_s,
            )
        )
        self.active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.previous_started = torch.zeros_like(self.active)
        self.native_success_sample = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.previous_head = None
        self._rows: dict[str, list[torch.Tensor]] = {}
        self._installed = False

    def __enter__(self):
        if self._installed:
            raise RuntimeError("contact trace is already installed")
        manager = self.env.metrics_manager
        self.original = manager.compute_substep

        def observe(*args, **kwargs):
            result = self.original(*args, **kwargs)
            self._sample()
            return result

        manager.compute_substep = observe
        self._installed = True
        return self

    def __exit__(self, *exc):
        if self._installed:
            self.env.metrics_manager.compute_substep = self.original
            self._installed = False

    def finish_step(self, done: torch.Tensor, evaluation_records=None) -> None:
        """Mark native autoresets before the next physics sample is observed."""
        if done.shape != self.active.shape:
            raise ValueError("done must contain one value per world")
        if evaluation_records is not None:
            if len(evaluation_records) != self.env.num_envs:
                raise ValueError("evaluation records must contain one entry per world")
            samples = torch.tensor(
                [record["native_success_sample"] or 0 for record in evaluation_records],
                dtype=torch.long, device=self.active.device,
            )
            first = self.active & (self.native_success_sample == 0)
            self.native_success_sample = torch.where(first, samples, self.native_success_sample)
        done = done.to(device=self.active.device, dtype=torch.bool)
        if self.shadow_window is None:
            self.active &= ~done
        else:
            self.active = self.shadow_window.mark_done(self.active, done)

    def _append(self, name: str, value: torch.Tensor) -> None:
        mask = self.active.reshape((-1,) + (1,) * (value.ndim - 1))
        self._rows.setdefault(name, []).append(
            torch.where(mask, value, torch.zeros_like(value)).detach().clone()
        )

    def _sample_nail_slide(self) -> None:
        """Capture native nail-slide state and its physically bounded depth."""
        data = self.env.sim.data
        position = data.qpos[:, self.nail_qpos_id]
        velocity = data.qvel[:, self.nail_dof_id]
        self._append("nail_slide_position_m", position)
        self._append("nail_slide_velocity_mps", velocity)
        self._append("clamped_nail_depth_m", position.clamp(0.0, NAIL_GOAL_DEPTH))

    def _sample(self) -> None:
        if len(self._rows.get("valid", ())) >= self.max_samples:
            raise RuntimeError("contact trace exceeded its configured evaluation horizon")
        controller, tracker = self.controller, self.tracker
        if any(getattr(controller, name) is None for name in
               ("applied_kp", "applied_kd", "requested_torque", "torque", "saturated")):
            raise RuntimeError("sampled controller has no applied torque snapshot")
        active = self.active.clone()
        self._rows.setdefault("valid", []).append(active)
        self.count += active.long()
        self._append("sample_time_s", self.count.to(controller.applied_kp.dtype) * self.dt)
        started = tracker.started
        accepted_onset = started & ~self.previous_started
        self._append("accepted_onset", accepted_onset)
        self._append("contact_started", started)
        self.previous_started = torch.where(active, started, self.previous_started)
        self._append("kp", controller.applied_kp)
        self._append("kd", controller.applied_kd)
        ids = controller.ids
        self._append("requested_torque_nm", controller.requested_torque[:, ids])
        self._append("clipped_torque_nm", controller.torque[:, ids])
        self._append("saturated", controller.saturated[:, ids])
        self._append("joint_load_nms", self.env._hammer_substep_impulse.impulse)
        accumulator = self.env._hammer_substep_impulse
        self._append("rolling_joint_load_nms", accumulator._rolling)
        self._append("control_step_peak_load_nms", accumulator._pulse)
        self._append("off_contact_baseline_nm", accumulator._last_off_qfrc)
        self._append("qfrc_constraint_nm",
                     accumulator._robot.data._joint_dof_field("qfrc_constraint")[:, accumulator._joint_ids])
        self._append("raw_contact", (tracker._contact_sensor.data.found > 0).any(dim=-1))
        self._sample_nail_slide()
        joint_position = self.env.sim.data.qpos[:, controller.qids][:, ids]
        requested_position = self.env.sim.data.ctrl[:, ids]
        desired_position = requested_position
        if getattr(controller, "startup_ramp_s", 0.0) > 0.0:
            desired_position = controller.applied_joint_position
            if desired_position is None or desired_position.shape != requested_position.shape:
                raise RuntimeError("startup controller applied-target snapshot missing or misordered")
            self._append("requested_joint_position_rad", requested_position)
        self._append("joint_position_rad", joint_position)
        self._append("desired_joint_position_rad", desired_position)
        self._append("joint_position_error_rad", desired_position - joint_position)
        self._append("joint_velocity_rad_s", self.env.sim.data.qvel[:, controller.vids][:, ids])
        speed = self.env.action_manager.get_term("strike_speed").state
        self._append("desired_speed_mps", speed.goal_mps)
        self._append("measured_contact_speed_mps", tracker.v_precontact)
        self._append("measured_contact_speed_valid", started)
        head = tracker._robot.data.site_pos_w[:, tracker._site_ids].squeeze(1)
        reference = self.env._strike_reference
        phase = reference.preview(head, self.env.episode_length_buf)
        self._append("hammer_head_position_w_m", head)
        self._append("spline_reference_position_w_m", reference.waypoint(phase))
        self._append("spline_phase", phase)
        self._append("spline_error_m", torch.linalg.vector_norm(reference.waypoint(phase) - head, dim=-1))
        geometric_phase, geometric_reference, geometric_error = reference.diagnose_current_height(head)
        self._append("geometric_spline_phase", geometric_phase)
        self._append("geometric_spline_reference_position_w_m", geometric_reference)
        self._append("geometric_spline_error_m", geometric_error)
        self._append("route_sign", reference.route_signs())
        if self.previous_head is None:
            downward = torch.zeros_like(self.count, dtype=head.dtype)
            valid_velocity = torch.zeros_like(active)
        else:
            downward = (self.previous_head[:, 2] - head[:, 2]) / self.dt
            valid_velocity = active & (self.count > 1)
        self._append("head_downward_velocity_mps", downward)
        self._append("head_velocity_valid", valid_velocity)
        self.previous_head = head.detach().clone()
        if self.shadow_window is not None:
            self.active = self.shadow_window.advance(self.active, self.count, accepted_onset)

    def tensors(self) -> dict[str, torch.Tensor]:
        """Return detached CPU values; invalid worlds contain zero placeholders."""
        values = {name: torch.stack(rows).cpu() for name, rows in self._rows.items()}
        values["native_success_sample"] = self.native_success_sample.detach().clone().cpu()
        if self.shadow_window is not None:
            values["shadow_onset_sample"] = self.shadow_window.onset_sample.detach().clone().cpu()
            values["shadow_complete"] = self.shadow_window.complete.detach().clone().cpu()
            values["shadow_censored"] = self.shadow_window.censored.detach().clone().cpu()
        return values

    def save(self, directory: Path) -> None:
        """Write once, with an explicit state-timing contract beside the tensors."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "contact_trace_v1",
            "physics_dt_s": self.dt,
            "sample_cadence": "after all per-substep metrics; one sample per physics integration",
            "sample_time_semantics": "elapsed physics time in this world's first episode",
            "site_state_offset_physics_samples": -1,
            "joint_state_semantics": "native post-integration qvel",
            "nail_slide_state_semantics": "native post-integration nail_slide qpos/qvel, sampled once per physics integration",
            "clamped_nail_depth_semantics": f"nail_slide qpos clamped to the physical [0, {NAIL_GOAL_DEPTH}] m travel range; endpoint values describe state at the declared evaluation-window endpoint, not permanent retention",
            "joint_position_semantics": "native post-integration qpos for the six arm joints in controller order",
            "desired_joint_position_semantics": "actual held physical position target delivered to sampled PD in controller joint order; startup-ramp traces separately retain raw requested_joint_position_rad",
            "joint_position_error_semantics": "desired actuator position target minus native post-integration qpos",
            "spline_state_semantics": "read-only reference preview and current hammer-head site position after substep metrics; site position may lag integration by one physics sample",
            "spline_error_semantics": "spline_error_m uses the policy-latched preview phase; geometric_spline_error_m independently uses current-height geometry and can move backward on rebound",
            "head_velocity_semantics": "finite difference of native site positions; first sample invalid",
            "load_semantics": "authoritative SubstepImpulseAccumulator.impulse, six ordered arm joints",
            "load_decomposition": "rolling window, current control-step peak, frozen off-contact qfrc baseline, and canonical qfrc_constraint are sampled after the accumulator",
            "native_success_sample_semantics": "first-episode native success sample copied from the pre-reset evaluation snapshot; zero means no observed native success",
            "onset_semantics": "first rising edge of accepted FirstStrikeEventTracker.started",
            "invalid_world_semantics": "zero placeholder after first native reset; consult valid mask",
            "controller_gain_mode": self.controller.gain_mode,
            "num_samples": len(self._rows.get("valid", ())),
            "num_worlds": self.env.num_envs,
        }
        if self.shadow_window is not None:
            metadata["post_terminal_shadow"] = {
                "post_onset_horizon_s": self.shadow_window.horizon_s,
                "hard_cap_from_episode_reset_s": self.shadow_window.hard_cap_s,
                "semantics": (
                    "evaluation-only continuation after the native 50ms success-measurement boundary; "
                    "unsettled, over-range, or still-contacting states are not native retained-depth outcomes"
                ),
            }
        with (directory / "contact-trace.pt").open("xb") as stream:
            torch.save(self.tensors(), stream)
        with (directory / "contact-trace.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")


class PostOnsetTraceWindow:
    """End a diagnostic trace a fixed time after accepted onset, censoring misses."""

    def __init__(self, *, num_envs: int, device, physics_dt_s: float,
                 post_onset_horizon_s: float, hard_cap_s: float):
        values = (physics_dt_s, post_onset_horizon_s, hard_cap_s)
        if (num_envs <= 0 or any(not math.isfinite(float(value)) or value <= 0
                                  for value in values)):
            raise ValueError("shadow timing values must be finite and positive")
        self.horizon_samples = self._whole_samples(
            post_onset_horizon_s, physics_dt_s, "post-onset horizon",
        )
        self.hard_cap_samples = self._whole_samples(
            hard_cap_s, physics_dt_s, "hard cap",
        )
        if self.hard_cap_samples <= self.horizon_samples:
            raise ValueError("shadow hard cap must exceed the post-onset horizon")
        self.dt = float(physics_dt_s)
        self.horizon_s = float(post_onset_horizon_s)
        self.hard_cap_s = float(hard_cap_s)
        self.onset_sample = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.censored = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @staticmethod
    def _whole_samples(duration_s: float, dt_s: float, label: str) -> int:
        samples = round(float(duration_s) / float(dt_s))
        if samples < 1 or not math.isclose(samples * dt_s, duration_s,
                                           rel_tol=0., abs_tol=1e-9):
            raise ValueError(f"shadow {label} must be an integer number of physics samples")
        return samples

    def advance(self, active: torch.Tensor, sample_count: torch.Tensor,
                accepted_onset: torch.Tensor) -> torch.Tensor:
        """Capture the inclusive onset sample and stop at the exact elapsed horizon."""
        if (active.shape != self.onset_sample.shape or sample_count.shape != active.shape
                or accepted_onset.shape != active.shape):
            raise ValueError("shadow inputs must contain one value per environment")
        accepted_onset = accepted_onset.to(device=active.device, dtype=torch.bool)
        sample_count = sample_count.to(device=active.device, dtype=torch.long)
        new_onset = active & accepted_onset & (self.onset_sample == 0)
        self.onset_sample.copy_(torch.where(new_onset, sample_count, self.onset_sample))
        elapsed = sample_count - self.onset_sample
        complete_now = active & (self.onset_sample > 0) & (elapsed >= self.horizon_samples)
        cap_now = active & (sample_count >= self.hard_cap_samples) & ~complete_now
        self.complete |= complete_now
        self.censored |= cap_now
        return active & ~(complete_now | cap_now)

    def mark_done(self, active: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Mark worlds stopped by any native non-success termination as censored."""
        if active.shape != self.onset_sample.shape or done.shape != active.shape:
            raise ValueError("shadow completion inputs must contain one value per environment")
        done = done.to(device=active.device, dtype=torch.bool)
        self.censored |= active & done & ~self.complete
        return active & ~done


class ContactSafetySummary:
    """Compact first-episode substep safety events, without retaining raw traces.

    The observer samples after the environment's existing per-substep metrics. It
    records the first accepted-contact and raw-sensor-contact samples, each joint's
    first velocity/load crossing, and sampled-PD saturation/gain context at those
    events. It only reads simulator/controller state and never changes it.
    """

    _CONTEXT_FIELDS = (
        "joint_velocity_rad_s", "joint_load_nms", "requested_torque_nm",
        "applied_torque_nm", "torque_saturated", "kp", "kd",
    )

    def __init__(self, env):
        controller = getattr(env, "_sampled_pd500", None)
        if controller is None:
            raise ValueError("contact safety summary requires the sampled PD controller")
        params = env.cfg.metrics["cat_soft"].params
        try:
            self.velocity_limit = float(params["limit"])
            self.caps = torch.as_tensor(params["imp_limit"], dtype=torch.float64,
                                        device=env.device)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("contact safety summary requires velocity and six load limits") from exc
        if (not math.isfinite(self.velocity_limit) or self.velocity_limit <= 0
                or self.caps.shape != (6,) or not torch.isfinite(self.caps).all()
                or not (self.caps > 0).all()):
            raise ValueError("contact safety summary requires positive finite limits for six joints")
        names = tuple(controller.gains.joint_names)
        if len(names) != 6:
            raise ValueError("contact safety summary requires six ordered arm joints")

        self.env, self.controller = env, controller
        self.tracker = env._hammer_first_strike
        self.accumulator = env._hammer_substep_impulse
        self.dt = float(env.physics_dt)
        self.num_envs = env.num_envs
        self.joint_names = names
        self.active = torch.ones(self.num_envs, dtype=torch.bool, device=env.device)
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(self.num_envs, *shape, dtype=dtype, device=env.device)
        self.count = zeros(dtype=torch.long)
        self.accepted_contact_sample = zeros(dtype=torch.long)
        self.raw_contact_sample = zeros(dtype=torch.long)
        self.velocity_peak = zeros(6)
        self.velocity_exceedance_samples = zeros(6, dtype=torch.long)
        self.velocity_first_sample = zeros(6, dtype=torch.long)
        self.load_peak = zeros(6, dtype=torch.float64)
        self.load_exceedance_samples = zeros(6, dtype=torch.long)
        self.load_first_sample = zeros(6, dtype=torch.long)
        self.saturation_samples = zeros(6, dtype=torch.long)
        self.saturation_first_sample = zeros(6, dtype=torch.long)
        self.early_end_sample = math.floor(.1 / self.dt + 1e-9)
        self.velocity_split_peak = zeros(2, 6)
        self.velocity_split_exceedance = zeros(2, 6, dtype=torch.long)
        self.load_split_peak = zeros(2, 6, dtype=torch.float64)
        self.load_split_exceedance = zeros(2, 6, dtype=torch.long)
        self.reset_joint_position = env.sim.data.qpos[:, controller.qids][:, controller.ids].clone()
        if (self.reset_joint_position.shape != (self.num_envs, 6)
                or not torch.isfinite(self.reset_joint_position).all()):
            raise ValueError("contact safety summary requires six finite physical reset joint positions")
        self.first_requested_delta = zeros(6)
        self.first_applied_delta = zeros(6)
        self.requested_jump_peak = zeros(6)
        self.applied_jump_peak = zeros(6)
        self.requested_jump_count = zeros(6, dtype=torch.long)
        self.applied_jump_count = zeros(6, dtype=torch.long)
        self.applied_tracking_error_peak = zeros(6)
        self._previous_requested_target = self.reset_joint_position.clone()
        self._previous_applied_target = self.reset_joint_position.clone()
        self._first_context = torch.zeros(
            self.num_envs, 3, len(self._CONTEXT_FIELDS), 6,
            dtype=torch.float64, device=env.device,
        )
        self.original = None

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("contact safety summary is already installed")
        manager = self.env.metrics_manager
        self.original = manager.compute_substep

        def observe(*args, **kwargs):
            result = self.original(*args, **kwargs)
            self._sample()
            return result

        manager.compute_substep = observe
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.env.metrics_manager.compute_substep = self.original
            self.original = None

    def _sample(self):
        controller = self.controller
        telemetry = (controller.applied_kp, controller.applied_kd,
                     controller.requested_torque, controller.torque, controller.saturated)
        if any(value is None for value in telemetry):
            raise RuntimeError("sampled controller has no applied torque snapshot")

        ids = controller.ids
        qvel = self.env.sim.data.qvel[:, controller.vids][:, ids]
        velocity = qvel[:, :6]
        load = self.accumulator.impulse
        # SampledPD500.applied_kp/kd are already selected into the canonical
        # joint order; indexing them again by native actuator IDs permutes gains.
        kp = controller.applied_kp
        kd = controller.applied_kd
        requested = controller.requested_torque[:, ids]
        applied = controller.torque[:, ids]
        saturated = controller.saturated[:, ids]
        expected = (self.num_envs, 6)
        values = (velocity, load, kp, kd, requested, applied, saturated)
        if any(value.shape != expected for value in values):
            raise ValueError("contact safety summary requires six per-joint substep channels")
        requested_target = getattr(controller, "requested_joint_position", None)
        applied_target = getattr(controller, "applied_joint_position", None)
        if requested_target is None or applied_target is None:
            raise RuntimeError("contact safety summary requires sampled requested/applied target snapshots")
        joint_position = self.env.sim.data.qpos[:, controller.qids][:, ids]
        if any(not isinstance(value, torch.Tensor) or value.shape != expected
               for value in (requested_target, applied_target, joint_position)):
            raise ValueError("contact safety summary requires six canonical physical target channels")

        active = self.active[:, None]
        self.count += self.active.long()
        sample = self.count[:, None].expand(-1, 6)
        first_target = active & (sample == 1)
        self.first_requested_delta = torch.where(
            first_target, requested_target - self.reset_joint_position, self.first_requested_delta)
        self.first_applied_delta = torch.where(
            first_target, applied_target - self.reset_joint_position, self.first_applied_delta)
        requested_jump = (requested_target - self._previous_requested_target).abs()
        applied_jump = (applied_target - self._previous_applied_target).abs()
        self.requested_jump_peak = torch.maximum(
            self.requested_jump_peak, torch.where(active, requested_jump, 0.))
        self.applied_jump_peak = torch.maximum(
            self.applied_jump_peak, torch.where(active, applied_jump, 0.))
        self.requested_jump_count += (active & (requested_jump > 0.)).long()
        self.applied_jump_count += (active & (applied_jump > 0.)).long()
        self.applied_tracking_error_peak = torch.maximum(
            self.applied_tracking_error_peak,
            torch.where(active, (applied_target - joint_position).abs(), 0.))
        self._previous_requested_target = torch.where(active, requested_target, self._previous_requested_target)
        self._previous_applied_target = torch.where(active, applied_target, self._previous_applied_target)
        raw_contact = (self.tracker._contact_sensor.data.found > 0).any(dim=-1)
        accepted_contact = self.tracker.started.bool()
        self.raw_contact_sample = torch.where(
            active[:, 0] & raw_contact & (self.raw_contact_sample == 0),
            self.count, self.raw_contact_sample,
        )
        self.accepted_contact_sample = torch.where(
            active[:, 0] & accepted_contact & (self.accepted_contact_sample == 0),
            self.count, self.accepted_contact_sample,
        )

        abs_velocity = velocity.abs()
        velocity_crossed = abs_velocity > self.velocity_limit
        load_crossed = load > self.caps
        velocity_event = active & velocity_crossed & (self.velocity_first_sample == 0)
        load_event = active & load_crossed & (self.load_first_sample == 0)
        saturation_event = active & saturated & (self.saturation_first_sample == 0)
        early = self.count[:, None] <= self.early_end_sample
        split_active = torch.stack((active & early, active & ~early), dim=1)
        self.velocity_split_peak = torch.maximum(
            self.velocity_split_peak, torch.where(split_active, abs_velocity[:, None, :], 0.))
        self.velocity_split_exceedance += (split_active & velocity_crossed[:, None, :]).long()
        self.load_split_peak = torch.maximum(
            self.load_split_peak, torch.where(split_active, load[:, None, :], 0.))
        self.load_split_exceedance += (split_active & load_crossed[:, None, :]).long()

        self.velocity_peak = torch.maximum(
            self.velocity_peak, torch.where(active, abs_velocity, 0.),
        )
        self.velocity_exceedance_samples += (active & velocity_crossed).long()
        self.velocity_first_sample = torch.where(velocity_event, sample, self.velocity_first_sample)

        self.load_peak = torch.maximum(self.load_peak, torch.where(active, load, 0.))
        self.load_exceedance_samples += (active & load_crossed).long()
        self.load_first_sample = torch.where(load_event, sample, self.load_first_sample)

        self.saturation_samples += (active & saturated).long()
        self.saturation_first_sample = torch.where(
            saturation_event, sample, self.saturation_first_sample,
        )
        context = torch.stack((velocity, load, requested, applied, saturated, kp, kd), dim=1)
        event_mask = torch.stack((velocity_event, load_event, saturation_event), dim=1)
        self._first_context = torch.where(
            event_mask[:, :, None, :], context[:, None, :, :], self._first_context,
        )

    def _event_times(self, first_sample, contact_sample):
        first = first_sample.detach().cpu().tolist()
        contact = int(contact_sample.item())
        times, relative = [], []
        for sample in first:
            if sample == 0:
                times.append(None)
                relative.append(None)
            else:
                times.append(sample * self.dt)
                relative.append((sample - contact) * self.dt if contact else None)
        return times, relative

    def _contexts(self, index, first_sample, event_index):
        samples = first_sample.detach().cpu().tolist()
        values = self._first_context[index, event_index].detach().cpu().tolist()
        result = []
        for joint, sample in enumerate(samples):
            result.append(None if sample == 0 else {
                "joint_velocity_rad_s": values[0][joint],
                "joint_load_nms": values[1][joint],
                "requested_torque_nm": values[2][joint],
                "applied_torque_nm": values[3][joint],
                "torque_saturated": bool(values[4][joint]),
                "kp": values[5][joint],
                "kd": values[6][joint],
            })
        return result

    def _record(self, index):
        accepted = int(self.accepted_contact_sample[index].item())
        raw = int(self.raw_contact_sample[index].item())
        velocity_first = self.velocity_first_sample[index]
        load_first = self.load_first_sample[index]
        saturation_first = self.saturation_first_sample[index]
        velocity_time, velocity_accepted = self._event_times(velocity_first, self.accepted_contact_sample[index])
        _, velocity_raw = self._event_times(velocity_first, self.raw_contact_sample[index])
        load_time, load_accepted = self._event_times(load_first, self.accepted_contact_sample[index])
        _, load_raw = self._event_times(load_first, self.raw_contact_sample[index])
        saturation_time, saturation_accepted = self._event_times(
            saturation_first, self.accepted_contact_sample[index],
        )
        _, saturation_raw = self._event_times(saturation_first, self.raw_contact_sample[index])
        def floats(values):
            return values[index].detach().cpu().tolist()
        count = int(self.count[index].item())

        def split_record(phase, peak, exceeded, peak_name):
            return {
                "sample_count": min(count, self.early_end_sample) if phase == 0 else max(0, count - self.early_end_sample),
                peak_name: peak[index, phase].detach().cpu().tolist(),
                "exceedance_samples": exceeded[index, phase].detach().cpu().tolist(),
                "exceedance_duration_s": (exceeded[index, phase].to(torch.float64) * self.dt).detach().cpu().tolist(),
            }

        return {
            "schema_version": "contact_safety_timing_v1",
            "sample_count": count,
            "physics_dt_s": self.dt,
            "sample_semantics": "one sample per physics integration, after existing substep metrics; qvel is native post-integration",
            "accepted_contact_sample": accepted or None,
            "accepted_contact_time_s": accepted * self.dt if accepted else None,
            "first_raw_contact_sample": raw or None,
            "first_raw_contact_time_s": raw * self.dt if raw else None,
            "relative_time_semantics": "event sample minus contact sample; negative is before contact, zero is same sample, positive is after",
            "exceedance_duration_semantics": "cumulative physics-sample exposure above the configured metric threshold, computed as sample count times physics_dt; not contiguous duration or raw contact-force duration",
            "startup_split_semantics": "early_100ms includes samples at elapsed time <=0.100s; late_after_100ms includes all subsequent first-attempt samples, even after an earlier crossing",
            "target_delivery": {
                "joint_names": list(self.joint_names),
                "reset_joint_position_rad": floats(self.reset_joint_position),
                "first_requested_delta_from_reset_rad": floats(self.first_requested_delta) if count else [None] * 6,
                "first_applied_delta_from_reset_rad": floats(self.first_applied_delta) if count else [None] * 6,
                "max_abs_requested_jump_rad": floats(self.requested_jump_peak),
                "max_abs_applied_jump_rad": floats(self.applied_jump_peak),
                "requested_jump_count": floats(self.requested_jump_count),
                "applied_jump_count": floats(self.applied_jump_count),
                "max_abs_applied_tracking_error_rad": floats(self.applied_tracking_error_peak),
                "target_semantics": "canonical physical requested/applied target snapshots at the latest 2ms PD sample; after offset, clipping and encoder correction",
                "jump_semantics": "nonzero per-joint change between successive sampled-target values; first command compared with actual reset qpos; repeated held physics samples add no jumps; no magnitude threshold",
                "tracking_error_semantics": "maximum absolute applied sampled target minus native post-integration joint position over the complete observed first attempt",
            },
            "joint_velocity": {
                "joint_names": list(self.joint_names),
                "limit_rad_s": self.velocity_limit,
                "peak_abs_rad_s": floats(self.velocity_peak),
                "exceedance_samples": self.velocity_exceedance_samples[index].detach().cpu().tolist(),
                "exceedance_duration_s": (self.velocity_exceedance_samples[index].to(torch.float64) * self.dt).detach().cpu().tolist(),
                "first_crossing_sample": [int(x) or None for x in velocity_first.detach().cpu().tolist()],
                "first_crossing_time_s": velocity_time,
                "relative_to_accepted_contact_s": velocity_accepted,
                "relative_to_first_raw_contact_s": velocity_raw,
                "context_at_first_crossing": self._contexts(index, velocity_first, 0),
                "early_100ms": split_record(0, self.velocity_split_peak, self.velocity_split_exceedance, "peak_abs_rad_s"),
                "late_after_100ms": split_record(1, self.velocity_split_peak, self.velocity_split_exceedance, "peak_abs_rad_s"),
            },
            "joint_load": {
                "joint_names": list(self.joint_names),
                "signal": "per-joint windowed contact impulse residual from SubstepImpulseAccumulator.impulse (max of current 50-ms rolling sum and current control-step peak)",
                "caps_nms": self.caps.detach().cpu().tolist(),
                "peak_nms": floats(self.load_peak),
                "exceedance_samples": self.load_exceedance_samples[index].detach().cpu().tolist(),
                "exceedance_duration_s": (self.load_exceedance_samples[index].to(torch.float64) * self.dt).detach().cpu().tolist(),
                "first_crossing_sample": [int(x) or None for x in load_first.detach().cpu().tolist()],
                "first_crossing_time_s": load_time,
                "relative_to_accepted_contact_s": load_accepted,
                "relative_to_first_raw_contact_s": load_raw,
                "context_at_first_crossing": self._contexts(index, load_first, 1),
                "early_100ms": split_record(0, self.load_split_peak, self.load_split_exceedance, "peak_nms"),
                "late_after_100ms": split_record(1, self.load_split_peak, self.load_split_exceedance, "peak_nms"),
            },
            "torque_saturation": {
                "joint_names": list(self.joint_names),
                "saturated_samples": self.saturation_samples[index].detach().cpu().tolist(),
                "saturated_duration_s": (self.saturation_samples[index].to(torch.float64) * self.dt).detach().cpu().tolist(),
                "first_crossing_sample": [int(x) or None for x in saturation_first.detach().cpu().tolist()],
                "first_crossing_time_s": saturation_time,
                "relative_to_accepted_contact_s": saturation_accepted,
                "relative_to_first_raw_contact_s": saturation_raw,
                "context_at_first_crossing": self._contexts(index, saturation_first, 2),
                "controller_gain_mode": self.controller.gain_mode,
            },
        }

    def finish_step(self, done: torch.Tensor) -> list[dict | None]:
        """Return compact records for worlds ending now, then stop observing them."""
        if done.shape != self.active.shape:
            raise ValueError("done must contain one value per world")
        done = done.to(device=self.active.device, dtype=torch.bool)
        completed = done & self.active
        indices = torch.nonzero(completed, as_tuple=False).flatten().detach().cpu().tolist()
        records = [None] * self.num_envs
        for index in indices:
            records[index] = self._record(index)
        self.active &= ~done
        return records

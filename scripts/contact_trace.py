"""Read-only, first-episode physics trace for impact-speed evaluation.

The metrics manager calls this observer after its per-substep terms. Joint state
and controller commands are read after integration; site state is the native
scene view used by the strike tracker and can lag that integration by one
physics sample. No simulator forward call is made to remove that offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class ContactTrace:
    """Observe first episodes without changing the environment's state."""

    def __init__(self, env, *, max_policy_steps: int | None = None):
        controller = getattr(env, "_sampled_pd500", None)
        if controller is None:
            raise ValueError("contact trace requires the sampled PD controller")
        self.env = env
        self.controller = controller
        self.tracker = env._hammer_first_strike
        self.dt = float(env.physics_dt)
        self.max_samples = (env.max_episode_length + 1 if max_policy_steps is None
                            else max_policy_steps) * env.cfg.decimation
        if self.max_samples <= 0:
            raise ValueError("contact trace requires a positive evaluation horizon")
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
        self.active &= ~done.to(device=self.active.device, dtype=torch.bool)

    def _append(self, name: str, value: torch.Tensor) -> None:
        mask = self.active.reshape((-1,) + (1,) * (value.ndim - 1))
        self._rows.setdefault(name, []).append(
            torch.where(mask, value, torch.zeros_like(value)).detach().clone()
        )

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
        self._append("accepted_onset", started & ~self.previous_started)
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
        self._append("joint_velocity_rad_s", self.env.sim.data.qvel[:, controller.vids][:, ids])
        speed = self.env.action_manager.get_term("strike_speed").state
        self._append("desired_speed_mps", speed.goal_mps)
        self._append("measured_contact_speed_mps", tracker.v_precontact)
        self._append("measured_contact_speed_valid", started)
        head = tracker._robot.data.site_pos_w[:, tracker._site_ids].squeeze(1)
        if self.previous_head is None:
            downward = torch.zeros_like(self.count, dtype=head.dtype)
            valid_velocity = torch.zeros_like(active)
        else:
            downward = (self.previous_head[:, 2] - head[:, 2]) / self.dt
            valid_velocity = active & (self.count > 1)
        self._append("head_downward_velocity_mps", downward)
        self._append("head_velocity_valid", valid_velocity)
        self.previous_head = head.detach().clone()

    def tensors(self) -> dict[str, torch.Tensor]:
        """Return detached CPU values; invalid worlds contain zero placeholders."""
        values = {name: torch.stack(rows).cpu() for name, rows in self._rows.items()}
        values["native_success_sample"] = self.native_success_sample.detach().clone().cpu()
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
        with (directory / "contact-trace.pt").open("xb") as stream:
            torch.save(self.tensors(), stream)
        with (directory / "contact-trace.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")

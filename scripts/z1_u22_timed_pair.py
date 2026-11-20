"""Single-recording hammer-head position tracking with optional actor/critic preview."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

SOURCE_DT = 0.01
PREVIEW_S = (0.02, 0.04, 0.08, 0.16)


def reference_errors(points, frame, nail, head):
    """Read a batch of fractional source frames in the registered world frame."""
    frame = frame.clamp(0, len(points) - 1)
    lower = frame.long()
    upper = (lower + 1).clamp_max(len(points) - 1)
    target = torch.lerp(points[lower], points[upper], (frame - lower).unsqueeze(-1))
    return nail.unsqueeze(1) + target - head.unsqueeze(1)


def reference_exhausted(env, clip_steps):
    return env.episode_length_buf >= clip_steps


def configure_termination(cfg, clip_steps, mode):
    """Keep B's native contact+200ms/5s limits, or add the legacy clip cutoff."""
    from mjlab.managers.termination_manager import TerminationTermCfg
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete

    if mode not in ("reference", "contact"):
        raise ValueError("Unknown termination mode")
    native = cfg.terminations["nail_driven"]
    assert native.func is horizon_window_complete and native.params == {"window_s": .2}
    assert native.time_out is False and cfg.episode_length_s == 5.
    cfg.terminations.pop("reference_exhausted", None)
    if mode == "reference":
        cfg.terminations["reference_exhausted"] = TerminationTermCfg(
            func=reference_exhausted, params={"clip_steps": clip_steps}, time_out=True)
        return clip_steps
    return math.ceil(cfg.episode_length_s / (cfg.decimation * cfg.sim.mujoco.timestep))


def recorded_time_phase(step, step_dt, duration):
    """DeepMimic's wrapped clock phase, using the loaded recording period."""
    if duration <= 0:
        raise ValueError("Recorded-time phase requires a positive duration")
    return torch.remainder(step * step_dt / duration, 1.)


def position_reward(error, step_dt, coefficient=5.):
    return 0.5 * step_dt * torch.exp(-coefficient * error.square().sum(-1))


def main():
    import warp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from rsl_rl.modules import EmpiricalNormalization
    from scripts.z1_safety_factorial import build_config
    from src.tasks.hammer.mdp.human_reference import reference_spec
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller, configure_impact_speed_experiment,
        install_impact_speed_controller, wrap_impact_speed_env,
    )
    from tensordict import TensorDict
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationActor038
    from src.tasks.hammer.rl.reward_exploration032 import RewardExplorationPPO
    from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
    from src.tasks.hammer.rl.spline_exploration_pilot import PilotBaselineActor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", choices=("U20", "U22"), default="U22")
    parser.add_argument("--arm", required=True, choices=("control", "preview"))
    parser.add_argument("--seed-index", required=True, type=int, choices=range(3))
    parser.add_argument("--coefficient", type=float, choices=(5., 10., 100., 200.), default=5.)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--termination", choices=("contact", "reference"), default=None,
                        help="Training defaults to contact+200ms with a 5s cap; evaluation follows the checkpoint unless overridden")
    parser.add_argument("--recorded-phase", action=argparse.BooleanOptionalAction, default=None,
                        help="Enabled by default for training; evaluation follows checkpoint metadata")
    args = parser.parse_args()
    preview = args.arm == "preview"
    phase_enabled = args.recorded_phase is not False
    training_termination = args.termination or "contact"
    normalization = dict(method="empirical", eps=1e-2, scope="all_coordinates",
                         separate_actor_critic=True, update_timing="per_environment_step")
    # Historical evaluations retain their original observation preprocessing.
    if args.evaluate:
        saved = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        saved_objective = saved["infos"]["optimized_objective"]
        training_termination = saved_objective.get("termination", {}).get("mode", "reference")
        phase_enabled = "recorded_time_phase" in saved_objective
        if args.recorded_phase is not None and args.recorded_phase != phase_enabled:
            parser.error("The recorded-phase option must match the evaluated checkpoint")
        normalization = saved_objective.get("observation_normalization")
        if normalization is not None:
            scope = "all_except_recorded_phase" if phase_enabled else "all_coordinates"
            expected = dict(method="empirical", eps=1e-2, scope=scope,
                            separate_actor_critic=True)
            assert normalization == expected or normalization == {
                **expected, "update_timing": "per_environment_step",
            }
        del saved
    termination = args.termination or training_termination
    if phase_enabled:
        normalization["scope"] = "all_except_recorded_phase"
    updates = 1 if args.smoke else 200
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    warp.config.use_precompiled_headers = False
    if os.environ.get("Z1_STUDY_CACHE"):
        warp.config.kernel_cache_dir = os.environ["Z1_STUDY_CACHE"]
    _, cfg = build_config("B", args.seed_index)
    path = reference_spec(args.reference)
    frames = path["source_frames"]
    assert frames == list(range(frames[0], frames[-1] + 1))
    duration = (len(frames) - 1) * SOURCE_DT
    step_dt = cfg.env.decimation * cfg.env.sim.mujoco.timestep
    clip_steps = math.ceil(duration / step_dt)
    phase_indices = {"actor": 54 if preview else 42, "critic": 59 if preview else 47}
    max_steps = configure_termination(cfg.env, clip_steps, termination)
    cfg.env.is_finite_horizon = True
    cfg.env.scene.num_envs = args.worlds
    cfg.agent.save_interval = 100
    cfg.agent.max_iterations = updates
    configure_impact_speed_experiment(cfg.env, bounds=cfg.speed_bounds, cost=cfg.speed_cost, mode=cfg.speed_mode)
    configure_impact_speed_controller(cfg.env, sampled_pd500=True, gain_mode=cfg.sampled_pd500_gain_mode)
    env = ManagerBasedRlEnv(cfg.env, device=device)
    install_impact_speed_controller(env, sampled_pd500=True, gain_mode=cfg.sampled_pd500_gain_mode)
    # The inherited B recipe validates its U22 defaults during construction.
    # Bind this tracking study's clip before the wrapper performs its first reset.
    for name in ("sample_strike_route_signs", "reset_strike_route_joints"):
        env.event_manager.get_term_cfg(name).params["reference_id"] = args.reference
    for group in ("actor", "critic"):
        for name in ("strike_phase", "strike_ref_error"):
            env.observation_manager.get_term_cfg(group, name).params["reference_id"] = args.reference
    if env._strike_reference.reference_id != args.reference:
        env._strike_reference = None
    wrapped = wrap_impact_speed_env(env, clip_actions=cfg.agent.clip_actions)
    points = torch.tensor(path["points_m"], dtype=torch.float32, device=device)
    points = points - points[-1]
    frame_offsets = points.new_tensor((0., *PREVIEW_S) if preview else (0.,)) / SOURCE_DT
    robot_cfg = env.reward_manager.get_term_cfg("spline_progress").params["robot_cfg"]
    robot = env.scene[robot_cfg.name]
    assert env.step_dt == .02 and len(robot_cfg.site_ids) == 1

    def errors(step, offsets):
        ref = env._strike_reference
        assert ref.reference_id == args.reference
        head = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
        return reference_errors(points, step[:, None] * (env.step_dt / SOURCE_DT) + offsets,
                                ref._nail, head)

    def observe(obs):
        vectors = errors(env.episode_length_buf, frame_offsets)
        for group in ("actor", "critic"):
            obs[group][:, 31:34] = vectors[:, 0]
            if preview:
                features = vectors[:, 1:].flatten(1)
                raw = obs[group]
                # The actor reads its binary pending flag from the final column.
                obs[group] = (torch.cat((raw[:, :-1], features, raw[:, -1:]), dim=-1)
                              if group == "actor" else torch.cat((raw, features), dim=-1))
            if phase_enabled:
                index = phase_indices[group]
                phase = recorded_time_phase(env.episode_length_buf, env.step_dt, duration).unsqueeze(-1)
                obs[group] = torch.cat((obs[group][:, :index], phase, obs[group][:, index:]), dim=-1)
        return obs

    get_obs, step = wrapped.get_observations, wrapped.step
    wrapped.get_observations = lambda: observe(get_obs())

    class TrackingNormalization(EmpiricalNormalization):
        def __init__(self, width, *, phase_index=None):
            self.phase_index = phase_index
            super().__init__(width - int(phase_index is not None), eps=1e-2)

        def forward(self, x):
            if self.phase_index is None:
                return super().forward(x)
            index = self.phase_index
            physical = torch.cat((x[..., :index], x[..., index + 1:]), dim=-1)
            normalized = super().forward(physical)
            phase = 2. * x[..., index:index + 1] - 1.
            return torch.cat((normalized[..., :index], phase, normalized[..., index:]), dim=-1)

        def update(self, x):
            # Keep buffers writable for checkpoint reloads even when stock PPO
            # updates statistics inside its inference-mode collection loop.
            with torch.inference_mode(False), torch.no_grad():
                if self.phase_index is not None:
                    index = self.phase_index
                    x = torch.cat((x[..., :index], x[..., index + 1:]), dim=-1)
                super().update(x.detach().clone())

    class TrackingActor(ExplorationActor038):
        observation_width = (55 if preview else 43) + int(phase_enabled)

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if normalization is not None:
                self.obs_normalizer = TrackingNormalization(
                    self.obs_dim, phase_index=phase_indices["actor"] if phase_enabled else None)

    class TrackingCritic(CreditEventCritic):
        observation_width = (59 if preview else 47) + int(phase_enabled)

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if normalization is not None:
                self.obs_normalizer = TrackingNormalization(
                    self.obs_dim, phase_index=phase_indices["critic"] if phase_enabled else None)

    agent = asdict(cfg.agent)
    agent["actor"]["class_name"] = TrackingActor
    agent["critic"]["class_name"] = TrackingCritic
    # Retain exploration/CaT bookkeeping, without the 038 normalization freeze,
    # startup priming or deferred updates. Stock PPO updates after every step.
    agent["algorithm"]["class_name"] = RewardExplorationPPO
    # Networks, optimizer and storage all see the complete observations at birth.
    runner = MjlabOnPolicyRunner(wrapped, agent, str(args.output), device)
    # Canonical networks are initialization templates only. Copy into the final
    # layers, leaving all preview and extra critic-context columns at zero.
    seed = cfg.agent.seed
    with torch.random.fork_rng(devices=[0] if device.startswith("cuda") else []):
        torch.manual_seed(seed)
        x = torch.zeros(1, 43, device=device)
        reference_obs = TensorDict({"actor": x, "critic": x.clone()}, [1])
        groups = {"actor": ["actor"], "critic": ["critic"]}
        common = ("hidden_dims", "activation", "obs_normalization")
        reference_actor = PilotBaselineActor(
            reference_obs, groups, "actor", 13,
            **{key: agent["actor"][key] for key in common},
            distribution_cfg={"class_name": "src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian"},
        ).to(device)
        reference_critic = ImpactSpeedAgeModel(
            reference_obs, groups, "critic", 1,
            **{key: agent["critic"][key] for key in common}, distribution_cfg=None,
        ).to(device)
    for name, reference in (("actor", reference_actor), ("critic", reference_critic)):
        model = getattr(runner.alg, name)
        state = reference.mlp.state_dict()
        first = torch.zeros_like(model.mlp[0].weight)
        if name == "actor" and (preview or phase_enabled):
            first[:, :42] = state["0.weight"][:, :42]
            first[:, -1] = state["0.weight"][:, -1]
        else:
            first[:, :43] = state["0.weight"]
        state["0.weight"] = first
        model.mlp.load_state_dict(state, strict=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    shared_weights = {}
    for name in ("actor", "critic"):
        model = getattr(runner.alg, name)
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            if phase_enabled and key == "mlp.0.weight":
                index = phase_indices[name]
                if torch.count_nonzero(value[:, index]):
                    raise ValueError("phase weights must start at zero")
                value = torch.cat((value[:, :index], value[:, index + 1:]), dim=1)
            if preview and key == "mlp.0.weight":
                columns = value[:, 42:-1] if name == "actor" else value[:, 47:]
                if torch.count_nonzero(columns):
                    raise ValueError("preview weights must start at zero")
                value = (torch.cat((value[:, :42], value[:, -1:]), dim=1)
                         if name == "actor" else value[:, :47])
            elif preview and normalization is not None and key.startswith("obs_normalizer.") and value.ndim == 2:
                value = (torch.cat((value[:, :42], value[:, -1:]), dim=1)
                         if name == "actor" else value[:, :47])
            digest.update(key.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        shared_weights[name] = digest.hexdigest()

    objective = dict(reference=args.reference, reference_sha256=path["data_sha256"], source_frames=frames,
                     source_dt_s=SOURCE_DT, timing="fixed", arm=args.arm, seed=cfg.agent.seed,
                     preview_s=list(PREVIEW_S) if preview else [],
                     reward_weights={"timed_head_position": .5}, coefficient_per_m2=args.coefficient,
                     effective_cat_delta=0., reference_duration_s=duration,
                     episode_steps=clip_steps, finite_horizon=True)
    if training_termination == "contact":
        objective.update(episode_steps=math.ceil(cfg.env.episode_length_s / env.step_dt),
                         reference_steps=clip_steps,
                         termination=dict(mode="contact", window_s=.2, timeout_s=5.,
                                          reward_after_reference="clamped_endpoint_tracking"))
    if normalization is not None:
        objective["observation_normalization"] = normalization
    if phase_enabled:
        objective["recorded_time_phase"] = dict(
            encoding="deepmimic_wrapped_scalar", period_s=duration, phase_offset=0.,
            normalization_offset=-.5, normalization_scale=2., empirical_normalization=False,
            clock="episode_steps_times_policy_dt", reset_phase=0.,
            source_commit="1f915c52fcd4b95b5f5f15b759ae91bd81e9a801")
    captured = {}
    terminal_counts = {"contact": 0, "reference_exhausted": 0, "time_out": 0}
    reset_idx = env._reset_idx
    metrics = env.metrics_manager.compute
    hook = env.metrics_manager.cfg["cat_soft"].func

    def collect(*a, **kw):
        result = metrics(*a, **kw)
        # Refresh derived positions before reading the terminal state.
        env.sim.forward()
        error = errors(env.episode_length_buf, points.new_zeros(1))[:, 0]
        captured["vectors"] = errors(env.episode_length_buf, frame_offsets)
        captured["step"] = env.episode_length_buf.clone()
        captured["reference_endpoint"] = captured["step"] == clip_steps
        captured["reference_exhausted"] = (
            env.termination_manager.get_term("reference_exhausted").clone()
            if termination == "reference" else torch.zeros_like(env.reset_time_outs))
        captured["contact_termination"] = env.termination_manager.get_term("nail_driven").clone()
        captured["time_out"] = env.termination_manager.get_term("time_out").clone()
        captured["error_m"] = error.norm(dim=-1)
        captured["reward"] = position_reward(error, env.step_dt, args.coefficient)
        if phase_enabled:
            captured["recorded_time_phase"] = recorded_time_phase(captured["step"], env.step_dt, duration)
        if args.evaluate:
            strike = env._hammer_first_strike
            telemetry = hook.constraint_telemetry()
            head = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1).clone()
            captured.update(
                time_s=env.episode_length_buf * env.step_dt,
                head_m=head,
                target_m=head + error,
                contact=strike.started.clone(),
                contact_time_s=strike.first_contact_time_s.clone(),
                axial_speed_m_s=strike.v_precontact.clone(),
                delivered_impulse_ns=strike.delivered.clone(),
                impulse_window_complete=strike.finalized.clone(),
                depth_at_contact_m=strike.depth_at_contact.clone(),
                peak_depth_m=strike.peak_depth.clone(),
                productive=strike.productive.clone(),
                joint_load_nms=telemetry["lambda_per_joint"],
                joint_speed_peak_rad_s=hook._substep_tracker(env).peak_qv_joint.clone(),
                constraint_excess={
                    key: hook._cat.raw_constraints[key].clone()
                    for key in ("joint_impulse_excess", "joint_velocity_excess")
                },
            )
        return result

    def verify_pre_reset(ids):
        if "step" in captured:
            torch.testing.assert_close(env.episode_length_buf[ids], captured["step"][ids])
            live_error = errors(env.episode_length_buf, frame_offsets)[ids, 0]
            torch.testing.assert_close(
                captured["reward"][ids], position_reward(live_error, env.step_dt, args.coefficient)
            )
        return reset_idx(ids)

    def timed_step(actions):
        obs, raw, dones, extras = step(actions)
        reward = captured["reward"].reshape_as(raw)
        extras = dict(extras)
        extras[runner.alg.DELTA_KEY] = torch.zeros_like(extras[runner.alg.DELTA_KEY])
        extras[runner.alg.R_POS_KEY] = reward
        next_obs = observe(obs)
        ended = dones.reshape(-1).bool()
        if ended.any():
            torch.testing.assert_close(env.episode_length_buf[ended], torch.zeros_like(captured["step"][ended]))
            terminal_counts["reference_exhausted"] += int(captured["reference_exhausted"].sum())
            terminal_counts["contact"] += int(captured["contact_termination"].sum())
            terminal_counts["time_out"] += int(captured["time_out"].sum())
        expected = captured["vectors"].clone()
        if ended.any():
            expected[ended] = errors(env.episode_length_buf, frame_offsets)[ended]
        for group in ("actor", "critic"):
            torch.testing.assert_close(next_obs[group][:, 31:34], expected[:, 0], atol=1e-5, rtol=1e-5)
            if preview:
                start = 42 if group == "actor" else 47
                values = next_obs[group][:, start:start + 12]
                torch.testing.assert_close(values, expected[:, 1:].flatten(1), atol=1e-5, rtol=1e-5)
            if phase_enabled:
                phase = recorded_time_phase(env.episode_length_buf, env.step_dt, duration)
                torch.testing.assert_close(next_obs[group][:, phase_indices[group]], phase)
        return next_obs, reward, dones, extras

    env._reset_idx = verify_pre_reset
    env.metrics_manager.compute = collect
    wrapped.step = timed_step
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "objective.json").write_text(json.dumps(objective, indent=2))
    (args.output / "shared-initial-weights.json").write_text(json.dumps(shared_weights))
    if args.evaluate:
        info = runner.load(str(args.evaluate), load_cfg={"actor": True, "critic": True}, strict=True)
        assert info["optimized_objective"] == objective
        (args.output / "evaluation-protocol.json").write_text(json.dumps(dict(
            training_termination=training_termination, termination=termination,
            max_episode_steps=max_steps, reference_steps=clip_steps,
            contact_window_s=.2, timeout_s=cfg.env.episode_length_s,
            joint_load_caps_nms=list(env.metrics_manager.cfg["cat_soft"].params["imp_limit"]),
            joint_speed_limit_rad_s=env.metrics_manager.cfg["cat_soft"].params["limit"],
            joint_load_window_s=env.metrics_manager.cfg["substep_impulse"].params["event_window_substeps"] * env.physics_dt,
            joint_load_subtract_baseline=env.metrics_manager.cfg["substep_impulse"].params.get("subtract_baseline", False),
            checkpoint_objective_preserved=True), indent=2))
        policy = runner.get_inference_policy(device=device)
        obs = observe(wrapped.reset()[0])
        finished = torch.zeros(args.worlds, dtype=torch.bool, device=device)
        trace = []
        with torch.inference_mode():
            for _ in range(max_steps):
                obs, _, done, _ = wrapped.step(policy(obs))
                trace.append({"active": (~finished).cpu(), **{
                    key: value.cpu() for key, value in captured.items() if isinstance(value, torch.Tensor) and key not in ("vectors", "step")
                }, "constraint_excess": {key: value.cpu() for key, value in captured["constraint_excess"].items()}})
                finished |= done.reshape(-1).bool()
                if finished.all():
                    break
        assert finished.all()
        torch.save(trace, args.output / "tracking.pt")
        active = torch.stack([row["active"] for row in trace])
        time = torch.stack([row["time_s"] for row in trace])
        errors_m = torch.stack([row["error_m"] for row in trace])
        apex = path["apex_index"] * SOURCE_DT

        def rmse(mask):
            return float(1000 * errors_m[mask].square().mean().sqrt()) if mask.any() else None

        contact = torch.stack([row["contact"] for row in trace]) & active
        final_step = active.long().sum(0) - 1
        worlds = torch.arange(args.worlds)
        complete = torch.stack([row["impulse_window_complete"] for row in trace])[final_step, worlds]
        impulse = torch.stack([row["delivered_impulse_ns"] for row in trace])[final_step, worlds]
        complete_impulse = impulse[complete]
        reference_prefix = torch.arange(len(trace))[:, None] < clip_steps
        summary = {
            "attempts": args.worlds,
            "checked_terminal_transitions": {
                "reference_exhausted": int((active & torch.stack([row["reference_exhausted"] for row in trace])).sum()),
                "contact": int((active & torch.stack([row["contact_termination"] for row in trace])).sum()),
                "time_out": int((active & torch.stack([row["time_out"] for row in trace])).sum()),
            },
            "contacts": int(contact.any(dim=0).sum()),
            "productive": int((torch.stack([row["productive"] for row in trace]) & active).any(dim=0).sum()),
            "full_clip_rmse_mm": rmse(active & reference_prefix),
            "full_episode_rmse_mm": rmse(active),
            "post_reference_rmse_mm": rmse(active & ~reference_prefix),
            "moving_rmse_mm": rmse(active & (time <= duration)),
            "descent_rmse_mm": rmse(active & (apex <= time) & (time <= duration)),
            "endpoint_rmse_mm": rmse(active & torch.stack([row["reference_endpoint"] for row in trace])),
            "completed_impulse_windows": int(complete.sum()),
            "censored_contacts": int((contact.any(0) & ~complete).sum()),
            "complete_window_impulse_ns": dict(
                count=len(complete_impulse),
                mean=float(complete_impulse.mean()) if len(complete_impulse) else None,
                median=float(complete_impulse.quantile(.5)) if len(complete_impulse) else None),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        return

    save = runner.save
    runner.save = lambda filename, infos=None: save(filename, {**(infos or {}), "optimized_objective": objective})
    process = runner.alg.process_env_step

    def check_reward(obs, reward, dones, extras):
        index = runner.alg.storage.step
        expected = captured["reward"].clone()
        counts = [model.obs_normalizer.count.clone() for model in (runner.alg.actor, runner.alg.critic)]
        assert "time_outs" not in extras, "Finite-horizon episodes must not bootstrap PPO returns"
        process(obs, reward, dones, extras)
        for model, count in zip((runner.alg.actor, runner.alg.critic), counts):
            torch.testing.assert_close(model.obs_normalizer.count, count + args.worlds)
        assert not runner.alg.storage.soft_dones[index].any()
        torch.testing.assert_close(runner.alg.storage.rewards[index].reshape(-1), expected)
        torch.testing.assert_close(runner.alg.storage.dones[index].reshape(-1).bool(), dones.reshape(-1).bool())

    runner.alg.process_env_step = check_reward
    runner.learn(updates)
    final = args.output / f"model_{updates - 1}.pt"
    info = runner.load(str(final), load_cfg={"actor": True, "critic": True}, strict=True)
    assert info["optimized_objective"] == objective
    (args.output / "qualified.json").write_text(json.dumps({"updates": updates, "objective": objective, "checked_terminal_transitions": terminal_counts}))


if __name__ == "__main__":
    main()

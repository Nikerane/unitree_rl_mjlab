"""Bounded, safety-on impact-speed evaluation; no learning or qualification claim.

Each requested goal gets a fresh, identically seeded environment. Comparisons use
only the first episode of each world, not asynchronously diverging reset streams.
Legacy tracking remains at reward time. Evaluation-only evidence separately
reports substep precontact path coverage and onset-aligned impact/load windows.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def compare_goal_responses(low: dict, high: dict, seed: int) -> dict:
    """Paired common-hit bootstrap, always accompanied by unconditional coverage.

    ``responsive`` is a diagnostic speed contrast, NOT a promotion/safety gate.
    IDs must already be in paired order; never silently reorder or drop misses.
    """
    n = len(low["hit"])
    for data in (low, high):
        if any(len(data[key]) != n for key in ("episode_id", "reset_hash", "hit", "speed")):
            raise ValueError("paired fields must have equal lengths")
        if len(set(data["episode_id"])) != n or any(not x for x in data["reset_hash"]):
            raise ValueError("paired IDs must be unique and reset hashes present")
    if low["episode_id"] != high["episode_id"] or low["reset_hash"] != high["reset_hash"]:
        raise ValueError("paired episode IDs and reset hashes must match in order")
    valid, coverage = [], {}
    for name, data in (("low", low), ("high", high)):
        available = np.array([hit and speed is not None and math.isfinite(speed)
                              for hit, speed in zip(data["hit"], data["speed"])], dtype=bool)
        valid.append(available)
        hits = sum(bool(x) for x in data["hit"])
        coverage[name] = dict(episodes=n, hits=hits, available=int(available.sum()), misses=n - hits)
    common = valid[0] & valid[1]
    differences = np.array([high["speed"][i] - low["speed"][i] for i in np.flatnonzero(common)])
    ci = None
    if len(differences) >= 2:
        rng = np.random.default_rng(seed)
        means = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return {"seed": seed, "bootstrap_samples": 2000, "common_valid_pairs": int(common.sum()),
            "paired_speed_difference": float(differences.mean()) if len(differences) else None,
            "ci95": ci, "responsive": bool(ci is not None and ci[0] > 0), "coverage": coverage,
            "interpretation": "common-valid speed contrast only; inspect miss coverage; not a qualification gate"}


class EpisodeAccumulator:
    """Consume fresh per-transition snapshots, keeping early events until done."""

    def __init__(self, num_envs: int, *, gamma: float, coefficient: float):
        self.gamma, self.coefficient = gamma, coefficient
        self.episodes = [0] * num_envs
        self.active = [self._new(i) for i in range(num_envs)]
        self.rows: list[dict] = []

    def _new(self, world):
        return dict(world=world, episode=self.episodes[world], steps=0, charge_count=0,
                    goal_mps=None, measured_mps=None, finalized=False, productive=False,
                    hit=False, miss=False, incomplete=False, timeout=False, complete=False,
                    first_contact_time_s=None, cost=0., discounted_cost=0., selection_count=0,
                    selected_raw=None, clipping_count=0, tracking_sum_sq_m2=0., tracking_max_mm=0.,
                    joint_target_sum_sq_rad2=0., joint_impulse_peak_nms=[0.] * 6,
                    impulse_exceedance_steps=[0] * 6, delivered_impulse_ns=0.,
                    cat_delta_sum=0., cat_delta_max=0., cat_active_steps=0,
                    kp_sum=[0.] * 6, kd_sum=[0.] * 6)

    def update(self, outcome, physical, raw, pending, evaluation=None):
        # Inputs are immutable evidence for THIS transition, not live reset tensors.
        for world, row in enumerate(self.active):
            if evaluation is not None:
                row["measurement"] = evaluation[world]
            step = row["steps"]
            if bool(outcome["charged"][world]):
                if row["charge_count"]:
                    raise ValueError("speed event charged twice in one episode")
                row["charge_count"] = 1
                for key in ("finalized", "productive", "miss", "incomplete"):
                    row[key] = bool(outcome[key][world])
                row["measured_mps"] = (float(outcome["measured_mps"][world])
                                       if bool(outcome["valid_measured"][world]) else None)
                row["cost"] = float(outcome["cost"][world]) * self.coefficient
                row["discounted_cost"] = row["cost"] * self.gamma**step
            row["goal_mps"] = float(outcome["goal_mps"][world])
            row["hit"] |= bool(physical["started"][world]) or bool(outcome["started"][world])
            if row["hit"] and row["first_contact_time_s"] is None:
                row["first_contact_time_s"] = float(physical["first_contact_time_s"][world])
            if bool(pending[world]):
                row["selection_count"] += 1
                row["selected_raw"] = float(raw[world, 12])
            row["clipping_count"] += int((raw[world].abs() > 1).sum())
            path_error = float(physical["path_error_m"][world])
            row["tracking_sum_sq_m2"] += path_error**2
            row["tracking_max_mm"] = max(row["tracking_max_mm"], path_error * 1000)
            row["joint_target_sum_sq_rad2"] += float(physical["joint_target_error_rad"][world])**2
            row["delivered_impulse_ns"] = float(physical["delivered_impulse_ns"][world])
            for j in range(6):
                row["joint_impulse_peak_nms"][j] = max(row["joint_impulse_peak_nms"][j], float(physical["joint_impulse_nms"][world, j]))
                row["impulse_exceedance_steps"][j] += int(physical["impulse_exceeded"][world, j])
                for key in ("kp", "kd"):
                    row[key + "_sum"][j] += float(physical[key][world, j])
            delta = float(physical["cat_delta"][world])
            row["cat_delta_sum"] += delta
            row["cat_delta_max"] = max(row["cat_delta_max"], delta)
            row["cat_active_steps"] += int(delta > 0)
            row["steps"] += 1
            if bool(outcome["done"][world]):
                row["complete"] = True
                row["timeout"] = bool(physical["timeout"][world])
                self.rows.append(self._finish_row(row))
                self.episodes[world] += 1
                self.active[world] = self._new(world)

    @staticmethod
    def _finish_row(row):
        result = dict(row)
        n = row["steps"]
        result["tracking_rms_mm"] = math.sqrt(result.pop("tracking_sum_sq_m2") / n) * 1000
        result["joint_target_rms_rad"] = math.sqrt(result.pop("joint_target_sum_sq_rad2") / n)
        result["cat_delta_mean"] = result.pop("cat_delta_sum") / n
        for key in ("kp", "kd"):
            result[key + "_mean"] = [x / n for x in result.pop(key + "_sum")]
        return result

    def finish(self):
        return self.rows + [self._finish_row(row) for row in self.active if row["steps"]]


def physical_snapshot(env):
    """Read native reward-time channels without mutating phase or safety state."""
    from src.tasks.hammer.mdp.trackability import joint_target_rmse

    tracker = env._hammer_first_strike
    robot_cfg = env.observation_manager.get_term_cfg("actor", "strike_ref_error").params["robot_cfg"]
    head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    reference = env._strike_reference
    phase = reference.preview(head, env.episode_length_buf)
    joint_cfg = env.metrics_manager.cfg["substep_impulse"].params["robot_cfg"]
    hook = next(cfg.func for name, cfg in zip(env.metrics_manager.active_terms, env.metrics_manager._term_cfgs)
                if name == "cat_soft")
    cat = hook.constraint_telemetry()
    gains = env.action_manager.get_term("joint_stiffness").telemetry
    sampled = getattr(env, "_sampled_pd500", None)
    if sampled is not None:
        if sampled.applied_kp is None or sampled.applied_kd is None:
            raise ValueError("sampled controller has no applied gains after a physics step")
        applied_kp, applied_kd = sampled.applied_kp, sampled.applied_kd
    else:
        applied_kp, applied_kd = gains.kp, gains.kd
    return {"path_error_m": torch.linalg.vector_norm(reference.waypoint(phase) - head, dim=-1),
            "joint_target_error_rad": joint_target_rmse(env, joint_cfg),
            "started": tracker.started, "first_contact_time_s": tracker.first_contact_time_s,
            "delivered_impulse_ns": tracker.delivered,
            "joint_impulse_nms": cat["lambda_per_joint"],
            "impulse_exceeded": cat["raw_margin_per_joint"] > 0,
            "cat_delta": cat["delta"], "kp": applied_kp, "kd": applied_kd,
            "timeout": env.reset_time_outs}


class PreResetSnapshot:
    """Evaluator-only seam: after all metrics, before native autoreset.

    The original manager runs exactly once. No rewards, resets, or safety knobs
    are replaced. Copies cover all worlds every step, including terminal ones.
    """

    def __init__(self, env, reader=physical_snapshot):
        self.env, self.reader, self.latest = env, reader, None
        self.latest_evaluation = None

    def __enter__(self):
        self.original = self.env.metrics_manager.compute

        def compute():
            result = self.original()
            self.latest = {k: v.detach().clone() for k, v in self.reader(self.env).items()}
            if hasattr(self.env, "_strike_evaluation"):
                from src.tasks.hammer.mdp.strike_evaluation import evaluation_snapshot
                self.latest_evaluation = evaluation_snapshot(self.env)
            return result

        self.env.metrics_manager.compute = compute
        return self

    def __exit__(self, *exc):
        self.env.metrics_manager.compute = self.original


def override_latched_goal(env, goal: float):
    state = env.action_manager.get_term("strike_speed").state
    if bool(env._hammer_first_strike.started.any()):
        raise ValueError("cannot intervene after contact has started")
    if bool(state.pending.any()):
        raise ValueError("intervention requires an already latched goal")
    if not math.isfinite(goal) or not state.bounds[0] <= goal <= state.bounds[1]:
        raise ValueError("goal outside speed bounds")
    state.goal_mps.fill_(goal)
    # compute(False) returns the existing cache. Public group computation bypasses
    # that cache without appending a fictitious extra step to observation history.
    return {group: env.observation_manager.compute_group(group, update_history=False)
            for group in ("actor", "critic")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--speed-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--speed-cost", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--mode", choices=("external", "policy", "intervention", "smoke"), required=True)
    parser.add_argument("--goals", nargs=3, type=float, help="low middle high, in increasing order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampled-pd500", action="store_true")
    parser.add_argument("--sampled-pd500-gain-mode", choices=("fixed", "variable"), default="fixed")
    parser.add_argument("--measurement-protocol", choices=("native", "common-window"), default="native")
    parser.add_argument("--trace-contact", action="store_true",
                        help="save first-episode physics traces for sampled PD evaluation")
    args = parser.parse_args(argv)
    lo, hi = args.speed_bounds
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi and
            math.isfinite(args.speed_cost) and args.speed_cost > 0 and args.num_envs > 0):
        parser.error("require positive finite bounds/cost and positive num-envs")
    if args.mode != "smoke" and args.checkpoint is None:
        parser.error("non-smoke evaluation requires --checkpoint")
    if args.mode in ("external", "intervention"):
        if args.goals is None or not lo <= args.goals[0] < args.goals[1] < args.goals[2] <= hi:
            parser.error("require three increasing --goals inside speed bounds")
    elif args.goals is not None:
        parser.error("--goals only applies to external/intervention")
    if args.output.exists():
        parser.error("output directory already exists; refusing overwrite")
    if args.mode == "smoke" and 64 % args.num_envs:
        parser.error("smoke num-envs must divide 64 (exactly 64 world transitions)")
    if args.sampled_pd500_gain_mode == "variable" and not args.sampled_pd500:
        parser.error("variable gain mode requires sampled PD control")
    if args.trace_contact and not args.sampled_pd500:
        parser.error("contact trace requires sampled PD control")
    return args


def _digest(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _reset_evidence(env, obs):
    """Reset physical state and RNG evidence, not a portable replay checkpoint."""
    robot, nail = env.scene["robot"].data, env.scene["nail_block"].data
    return [_digest(robot.joint_pos[i], robot.joint_vel[i], nail.joint_pos[i], nail.joint_vel[i],
                    obs["actor"][i, :40], obs["critic"][i, :40]) for i in range(env.num_envs)]


def _parity_models(runner, directory):
    runner.export_policy_to_jit(str(directory), "actor.pt")
    jit = torch.jit.load(str(directory / "actor.pt"))
    backend = None
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError:
        return jit, backend
    runner.export_policy_to_onnx(str(directory), "actor.onnx")
    model = onnx.load(str(directory / "actor.onnx"))
    onnx.checker.check_model(model)
    backend = ReferenceEvaluator(model)
    return jit, backend


def _condition(args, goal, index):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from tensordict import TensorDict
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.impact_speed_wrapper import (configure_impact_speed_experiment,
        configure_impact_speed_controller, install_impact_speed_controller, wrap_impact_speed_env)
    from src.tasks.hammer.mdp.strike_evaluation import (
        configure_strike_evaluation, MeasurementProtocol, summarize_strike_evaluations,
        LOAD_SIGNAL_SEMANTICS)
    from scripts.contact_trace import ContactTrace

    # Play configs remove the finite timeout and alter reset/noise distributions.
    # Evaluation retains the training task's horizon, safety, and reset treatment.
    cfg, agent_cfg = load_env_cfg(args.task, play=False), load_rl_cfg(args.task)
    mode = "external" if args.mode == "external" else "policy"
    # A checkpoint-bearing smoke uses its saved phase, still validated exactly below.
    if args.mode == "smoke" and args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        mode = payload["infos"]["impact_speed_contract"]["speed_mode"]
    configure_impact_speed_experiment(cfg, bounds=tuple(args.speed_bounds), cost=args.speed_cost, mode=mode)
    configure_impact_speed_controller(cfg, sampled_pd500=args.sampled_pd500, gain_mode=args.sampled_pd500_gain_mode)
    configure_strike_evaluation(cfg)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    directory = args.output / f"condition-{index}"
    directory.mkdir()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        install_impact_speed_controller(env, sampled_pd500=args.sampled_pd500, gain_mode=args.sampled_pd500_gain_mode)
        wrapped = wrap_impact_speed_env(env, agent_cfg.clip_actions)
        runner = load_runner_cls(args.task)(wrapped, asdict(agent_cfg), device=args.device)
        if args.checkpoint is not None:
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        contract = build_impact_speed_metadata(env, wrapped.clip_actions)
        # Loading and runner setup may consume RNG. Reset the RNG immediately before
        # each matched reset, then preserve its post-reset state as separate evidence.
        env.seed(args.seed)
        obs, _ = wrapped.reset()
        reset_hashes = _reset_evidence(env, obs)
        rng_hash = _digest(torch.get_rng_state())
        cuda_rng_hash = _digest(*torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        if mode == "external" and goal is not None:
            obs = TensorDict(override_latched_goal(env, goal), batch_size=[env.num_envs])
        policy = runner.get_inference_policy(device=args.device)
        parity = {"pt": "not requested", "onnx": "not requested", "observations": 0}
        jit = backend = None
        if args.mode == "smoke":
            jit, backend = _parity_models(runner, directory)
            parity["onnx"] = "unavailable" if backend is None else "pending"
        gamma = agent_cfg.algorithm.gamma
        acc = EpisodeAccumulator(env.num_envs, gamma=gamma, coefficient=args.speed_cost)
        steps = 64 // args.num_envs if args.mode == "smoke" else env.max_episode_length + 1
        first_action_hash = second_action = second_kp = None
        # Runner/checkpoint validation above sees the original training termination.
        # The context restores it before the final safety/controller contract check.
        trace = (ContactTrace(env, max_policy_steps=steps)
                 if getattr(args, "trace_contact", False) else None)
        with (torch.inference_mode(), MeasurementProtocol(env, args.measurement_protocol),
              PreResetSnapshot(env) as capture, trace if trace is not None else nullcontext()):
            for step in range(steps):
                if args.mode == "intervention" and step == 1:
                    if any(acc.episodes):
                        raise ValueError("cannot intervene after first-step reset")
                    obs = TensorDict(override_latched_goal(env, goal), batch_size=[env.num_envs])
                for group in ("actor", "critic"):
                    if obs[group].shape != (env.num_envs, 42) or not torch.isfinite(obs[group]).all():
                        raise ValueError("expected finite 42D actor/critic observations")
                state = env.action_manager.get_term("strike_speed").state
                pending, previous_goal = state.pending.clone(), state.goal_mps.clone()
                raw = policy(obs)
                if raw.shape != (env.num_envs, 13) or not torch.isfinite(raw).all():
                    raise ValueError("expected finite 13D actor output")
                if step == 0:
                    first_action_hash = _digest(raw)
                elif step == 1:
                    second_action = raw[:, :12].cpu().tolist()
                if jit is not None and step < 3:
                    cpu_obs = obs["actor"].cpu()
                    torch.testing.assert_close(jit(cpu_obs), raw.cpu(), rtol=1e-5, atol=1e-6)
                    parity["pt"] = "passed"
                    if backend is not None:
                        predicted = np.concatenate([backend.run(None, {backend.input_names[0]: row[None].numpy()})[0]
                                                    for row in cpu_obs])
                        torch.testing.assert_close(torch.from_numpy(predicted), raw.cpu(), rtol=1e-5, atol=1e-6)
                        parity["onnx"] = "passed: ReferenceEvaluator"
                    parity["observations"] += 1
                obs, _, done, extras = wrapped.step(raw)
                if trace is not None:
                    trace.finish_step(done, capture.latest_evaluation)
                outcome = extras["impact_speed_outcome"]
                if not torch.equal(outcome["done"], done.bool()):
                    raise ValueError("terminal speed evidence disagrees with native done")
                lo, hi = args.speed_bounds
                expected = torch.where(pending.bool(), lo + (raw[:, 12].clamp(-1, 1) + 1) * .5 * (hi - lo), previous_goal)
                torch.testing.assert_close(outcome["goal_mps"], expected)
                torch.testing.assert_close(state.pending, done.float() if mode == "policy" else torch.zeros_like(pending))
                for key, value in capture.latest.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"nonfinite physical metric {key}")
                acc.update(outcome, capture.latest, raw, pending, capture.latest_evaluation)
                if step == 1:
                    second_kp = capture.latest["kp"].cpu().tolist()
                if args.mode != "smoke" and all(episode >= 1 for episode in acc.episodes):
                    break
        if trace is not None:
            trace.save(directory)
        if build_impact_speed_metadata(env, wrapped.clip_actions) != contract:
            raise ValueError("live safety/controller contract changed during evaluation")
        rows = acc.finish()
        for row in rows:
            row.update(condition=index, intervention_goal_mps=goal,
                       episode_id=f"{row['world']}:{row['episode']}",
                       reset_hash=reset_hashes[row["world"]] if row["episode"] == 0 else None)
        return rows, {"goal_mps": goal, "contract": contract, "reset_hashes": reset_hashes,
                      "contact_trace": ("contact-trace.pt" if trace is not None else None),
                      "measurement": {"protocol": args.measurement_protocol,
                          "all_observed_attempts": summarize_strike_evaluations([r["measurement"] for r in rows]),
                          "success_reset_override": args.measurement_protocol == "common-window",
                          "site_state_offset_physics_samples": -1,
                          "termination_check_cadence_s": env.step_dt,
                          "physics_dt_s": env.physics_dt, "required_window_samples": env._strike_evaluation.window,
                          "window_duration_s": .05, "onset_sample_included": True,
                          "path_bins": 16, "impulse_dose": cfg.metrics["cat_soft"].params["imp_max_p"],
                          "load_signal": LOAD_SIGNAL_SEMANTICS},
                      "torch_post_reset_rng_hash": rng_hash, "cuda_post_reset_rng_hash": cuda_rng_hash,
                      "first_action_hash": first_action_hash, "second_motor_action": second_action,
                      "second_kp": second_kp, "transitions": (step + 1) * env.num_envs,
                      "selection_count": sum(row["selection_count"] for row in rows), "parity": parity}
    finally:
        env.close()


def run(args):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    args.output.mkdir(parents=True, exist_ok=False)
    rows, conditions = [], []
    for index, goal in enumerate(args.goals or [None]):
        condition_rows, report = _condition(args, goal, index)
        rows.extend(condition_rows)
        conditions.append(report)
    comparisons = []
    if args.goals:
        baseline = conditions[0]
        for condition in conditions[1:]:
            for key in ("reset_hashes", "torch_post_reset_rng_hash", "cuda_post_reset_rng_hash"):
                if condition[key] != baseline[key]:
                    raise ValueError(f"paired initial state/RNG mismatch: {key}")
            if args.mode == "intervention" and condition["first_action_hash"] != baseline["first_action_hash"]:
                raise ValueError("paired intervention first actions differ")
        def response(index):
            initial = sorted((r for r in rows if r["condition"] == index and r["episode"] == 0), key=lambda r: r["world"])
            return {"episode_id": [r["episode_id"] for r in initial], "reset_hash": [r["reset_hash"] for r in initial],
                    "hit": [r["hit"] for r in initial], "speed": [r["measured_mps"] for r in initial]}
        for low, high in ((0, 1), (1, 2), (0, 2)):
            contrast = compare_goal_responses(response(low), response(high), args.seed)
            contrast.update(low_goal_mps=args.goals[low], high_goal_mps=args.goals[high])
            if args.mode == "intervention":
                contrast["second_motor_action_rms_difference"] = float(np.sqrt(np.mean((np.array(conditions[high]["second_motor_action"]) - np.array(conditions[low]["second_motor_action"]))**2)))
                contrast["second_kp_rms_difference"] = float(np.sqrt(np.mean((np.array(conditions[high]["second_kp"]) - np.array(conditions[low]["second_kp"]))**2)))
            comparisons.append(contrast)
    checkpoint_hash = None
    if args.checkpoint is not None:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint_hash = digest.hexdigest()
    summary = {"task": args.task, "mode": args.mode, "seed": args.seed,
               "checkpoint": str(args.checkpoint) if args.checkpoint else None, "checkpoint_sha256": checkpoint_hash,
               "actor_status": "checkpoint" if args.checkpoint else "untrained software smoke only",
               "qualification": "not assessed; bounds/cost are supplied experiment inputs, not physical certification",
               "transitions": sum(c["transitions"] for c in conditions), "learning_updates": 0,
               "conditions": conditions, "comparisons": comparisons,
               "timeout_count": sum(r["timeout"] for r in rows), "completed_episodes": sum(r["complete"] for r in rows),
               "cost_semantics": "positive cost magnitude; gamma-discounted from first transition; timeout is reset-boundary shaping, not a hard deadline",
               "tracking_semantics": "all sampled steps, geometric reference preview, reward-time one-substep-stale positions",
               "reset_evidence_semantics": "first episode joint/nail positions and velocities plus physical observations; fresh same-seed native construction; not a portable full-state replay"}
    with (args.output / "episodes.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    args = parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "transitions": result["transitions"],
                      "completed_episodes": result["completed_episodes"], "learning_updates": 0}))


if __name__ == "__main__":
    main()

"""Bounded three-route external-versus-policy impact-speed pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

TASK = "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed"
SPEED_BOUNDS = (1.5, 2.2)
SPEED_COST = .8493676
EXTERNAL_GOALS = (1.5, 1.85, 2.2)
EVALUATION_SEED = 20260925
EVALUATION_WORLDS = 384


def run_spec(index: int, *, iterations: int = 50, worlds: int = 1024) -> dict:
    """Six fixed run slots: three seeds paired across speed modes."""
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(6):
        raise ValueError("array index must be 0..5")
    return {
        "speed_mode": ("external", "policy")[index % 2],
        "seed": 10 + index // 2,
        "gain_mode": "variable",
        "iterations": iterations,
        "worlds": worlds,
    }


def training_config(spec: dict):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from scripts.train import TrainConfig

    cfg = replace(
        TrainConfig.from_task(TASK),
        sampled_pd500=True,
        sampled_pd500_gain_mode="variable",
        speed_bounds=SPEED_BOUNDS,
        speed_cost=SPEED_COST,
        speed_mode=spec["speed_mode"],
        qualification_checkpoint=None,
    )
    cfg.env.scene.num_envs = spec["worlds"]
    cfg.agent.seed = spec["seed"]
    cfg.agent.max_iterations = spec["iterations"]
    cfg.agent.save_interval = max(1, min(25, spec["iterations"] - 1))
    cfg.agent.logger = "tensorboard"
    return cfg


def evaluation_command(checkpoint, output, speed_mode, protocol, smoke=False):
    if speed_mode not in ("external", "policy"):
        raise ValueError("evaluation mode must be external or policy")
    command = [
        sys.executable, "scripts/evaluate_impact_speed.py", "--task", TASK,
        "--checkpoint", str(checkpoint), "--speed-bounds", "1.5", "2.2",
        "--speed-cost", ".8493676", "--seed", str(EVALUATION_SEED),
        "--num-envs", "8" if smoke else str(EVALUATION_WORLDS),
        "--mode", "smoke" if smoke else speed_mode,
        "--output", str(output), "--device", "cuda:0", "--sampled-pd500",
        "--sampled-pd500-gain-mode", "variable", "--measurement-protocol", protocol,
    ]
    if not smoke and speed_mode == "external":
        command.extend(["--goals", *(str(goal) for goal in EXTERNAL_GOALS)])
    return command


def pair_reset_evidence(external_summary: dict, policy_summary: dict) -> dict:
    """Claim paired initial states only when both hashes and routes match."""
    external = external_summary.get("conditions", [])
    policy = policy_summary.get("conditions", [])
    policy_condition = policy[0] if len(policy) == 1 else None
    comparisons = []
    for index, condition in enumerate(external):
        hashes_match = bool(
            policy_condition
            and condition.get("reset_hashes")
            and condition.get("reset_hashes") == policy_condition.get("reset_hashes")
        )
        signs = condition.get("first_episode_route_signs")
        policy_signs = (policy_condition or {}).get("first_episode_route_signs")
        routes_match = bool(signs and signs == policy_signs)
        comparisons.append({
            "external_goal_mps": condition.get("goal_mps"),
            "reset_hashes_match": hashes_match,
            "route_signs_match": routes_match,
            "paired_initial_states": hashes_match and routes_match,
        })
    complete_comparison = len(external) == len(EXTERNAL_GOALS) and len(comparisons) == len(EXTERNAL_GOALS)
    paired = bool(complete_comparison and comparisons and all(row["paired_initial_states"] for row in comparisons))
    return {
        "claim": "paired_initial_evaluation_states" if paired else "matched_training_seeds_only",
        "external_condition_count": len(external),
        "policy_condition_count": len(policy),
        "goal_conditions": comparisons,
        "pairing_semantics": "physical reset hashes and actual pre-reset route signs must both match",
    }


def _selector_preflight(checkpoint: Path, spec: dict) -> dict:
    """Strictly reload one fresh run and verify the live selector mask/latch."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_runner_cls
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller,
        configure_impact_speed_experiment,
        install_impact_speed_controller,
        wrap_impact_speed_env,
    )

    cfg = training_config(dict(spec, worlds=8))
    cfg.env.seed = spec["seed"]
    configure_impact_speed_experiment(
        cfg.env, bounds=SPEED_BOUNDS, cost=SPEED_COST, mode=spec["speed_mode"]
    )
    configure_impact_speed_controller(cfg.env, sampled_pd500=True, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
    try:
        install_impact_speed_controller(env, sampled_pd500=True, gain_mode="variable")
        wrapper = wrap_impact_speed_env(env, cfg.agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapper, asdict(cfg.agent), device="cuda:0")
        runner.load(str(checkpoint), strict=True, map_location="cuda:0")
        obs, _ = wrapper.reset()
        speed_state = env.action_manager.get_term("strike_speed").state
        pending = speed_state.pending.clone()
        policy = runner.get_inference_policy(device="cuda:0")
        with torch.inference_mode():
            action = policy(obs)
            actor_distribution = runner.alg.actor.distribution
            mask = actor_distribution.activity_mask
            expected_mask = torch.cat((torch.ones_like(action[:, :12]), pending.unsqueeze(-1)), dim=-1)
            torch.testing.assert_close(mask, expected_mask)
            torch.testing.assert_close(pending, torch.full_like(pending, float(spec["speed_mode"] == "policy")))
            previous_goal = speed_state.goal_mps.clone()
            obs, _, done, _ = wrapper.step(action)
            if bool(done.any()):
                raise ValueError("selector preflight reset before the latch stability check")
            expected_goal = previous_goal
            if spec["speed_mode"] == "policy":
                lo, hi = SPEED_BOUNDS
                expected_goal = lo + (action[:, 12].clamp(-1, 1) + 1) * .5 * (hi - lo)
                torch.testing.assert_close(speed_state.pending, torch.zeros_like(pending))
            else:
                torch.testing.assert_close(speed_state.pending, torch.zeros_like(pending))
            torch.testing.assert_close(speed_state.goal_mps, expected_goal)
            latched = speed_state.goal_mps.clone()
            next_action = policy(obs)
            next_action[:, 12] = -next_action[:, 12]
            wrapper.step(next_action)
            torch.testing.assert_close(speed_state.goal_mps, latched)
        return {
            "strict_reload": "passed",
            "pending_at_reset": int(pending.sum().item()),
            "active_speed_mask_count": int(mask[:, 12].sum().item()),
            "latch_stable_after_selection": True,
            "live_contract": build_impact_speed_metadata(env, wrapper.clip_actions),
        }
    finally:
        env.close()


def _rollout_evidence(checkpoint: Path, spec: dict, initial_checkpoint: Path | None = None) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    telemetry = infos.get("impact_speed_rollout_telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("checkpoint is missing impact-speed rollout selector telemetry")
    if telemetry.get("selection_source") != "saved_distribution_activity_mask":
        raise ValueError("checkpoint selector telemetry is not derived from the saved activity mask")
    if telemetry.get("speed_bounds_mps") != list(SPEED_BOUNDS):
        raise ValueError("checkpoint selector telemetry speed bounds disagree with the campaign")
    selections = telemetry.get("selection_count")
    valid_selection_count = (
        isinstance(selections, int) and not isinstance(selections, bool) and selections >= 0
        and (selections > 0 if spec["speed_mode"] == "policy" else selections == 0)
    )
    if not valid_selection_count:
        raise ValueError(f"training selector telemetry disagrees with {spec['speed_mode']} mode")
    evidence = {
        "selection_count": selections,
        "selection_source": telemetry.get("selection_source"),
        "selected_speed_mps": telemetry.get("selected_speed_mps"),
        "speed_bounds_mps": telemetry.get("speed_bounds_mps"),
    }
    if initial_checkpoint is not None and initial_checkpoint.is_file():
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        final_actor, initial_actor = payload.get("actor_state_dict"), initial.get("actor_state_dict")
        if isinstance(final_actor, dict) and isinstance(initial_actor, dict):
            candidates = [
                key for key, value in final_actor.items()
                if "mlp" in key and key.endswith("weight") and isinstance(value, torch.Tensor)
                and value.ndim == 2 and value.shape[0] == 13
                and isinstance(initial_actor.get(key), torch.Tensor)
                and initial_actor[key].shape == value.shape
            ]
            if candidates:
                key = candidates[-1]
                weight_delta = final_actor[key][12] - initial_actor[key][12]
                bias_key = key[:-6] + "bias"
                bias_delta = None
                if (isinstance(final_actor.get(bias_key), torch.Tensor)
                        and isinstance(initial_actor.get(bias_key), torch.Tensor)
                        and final_actor[bias_key].shape == initial_actor[bias_key].shape == (13,)):
                    bias_delta = final_actor[bias_key][12] - initial_actor[bias_key][12]
                components = [weight_delta.flatten()]
                if bias_delta is not None:
                    components.append(bias_delta.reshape(1))
                evidence["speed_head_row12_weight"] = key
                evidence["model_0_to_final_speed_head_row12_l2"] = float(
                    torch.linalg.vector_norm(torch.cat(components)).item()
                )
                evidence["parameter_change_comparison"] = (
                    "earliest saved model_0 (post-update iteration 0) to final checkpoint; not initialization"
                )
    return evidence


def train_and_evaluate(spec: dict, output: Path, *, smoke: bool = False) -> dict:
    import torch
    from mjlab.tasks.registry import load_runner_cls
    from scripts.train import run_train

    output.mkdir(parents=True, exist_ok=False)
    cfg = training_config(spec)
    train_dir = output / "train"
    train_dir.parent.mkdir(parents=True, exist_ok=True)
    run_train(TASK, cfg, train_dir)
    checkpoint = train_dir / f"model_{spec['iterations'] - 1}.pt"
    initial_checkpoint = train_dir / "model_0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"expected final checkpoint not found: {checkpoint}")
    selector = _selector_preflight(checkpoint, spec)
    rollout = _rollout_evidence(checkpoint, spec, initial_checkpoint)

    for protocol in ("native", "common-window"):
        subprocess.run(
            evaluation_command(checkpoint, output / f"eval-{protocol}", spec["speed_mode"],
                               protocol, smoke),
            check=True,
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = {
        **spec,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "strict_reload": selector,
        "training_selector_telemetry": rollout,
        "evaluation": "completed",
        "smoke": smoke,
        "iteration": payload.get("iter"),
        "fresh_initialization": True,
        "checkpoint_transfer": "none",
    }
    (output / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("THREE_ROUTE_CAMPAIGN_RUN_COMPLETED", json.dumps(result), flush=True)
    return result


def audit_pairings(campaign: Path) -> dict:
    comparisons = {}
    for seed_index, seed in enumerate((10, 11, 12)):
        external_index, policy_index = 2 * seed_index, 2 * seed_index + 1
        for protocol in ("native", "common-window"):
            external_path = campaign / "runs" / f"run-{external_index}" / f"eval-{protocol}" / "summary.json"
            policy_path = campaign / "runs" / f"run-{policy_index}" / f"eval-{protocol}" / "summary.json"
            if not external_path.is_file() or not policy_path.is_file():
                raise FileNotFoundError(f"missing completed summaries for seed {seed}, {protocol}")
            report = pair_reset_evidence(
                json.loads(external_path.read_text()), json.loads(policy_path.read_text())
            )
            comparisons[f"seed-{seed}-{protocol}"] = report
    summary = {
        "training_seed_pairing": "seed labels 10/11/12; no transition-level pairing claim",
        "evaluation_pairing": comparisons,
        "global_claim": (
            "paired_initial_evaluation_states"
            if all(item["claim"] == "paired_initial_evaluation_states" for item in comparisons.values())
            else "matched_training_seeds_only"
        ),
    }
    destination = campaign / "pairing-audit.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pairing evidence: {destination}")
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "train", "audit"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "audit":
        print(json.dumps(audit_pairings(args.output), indent=2))
    elif args.phase == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
        for index, mode in ((0, "external"), (1, "policy")):
            spec = run_spec(index, iterations=2, worlds=32)
            if spec["speed_mode"] != mode:
                raise AssertionError("preflight run matrix no longer covers both speed modes")
            train_and_evaluate(spec, args.output / mode, smoke=True)
        (args.output / "PASS").touch(exist_ok=False)
    else:
        if args.index is None:
            parser.error("--index is required for train phase")
        train_and_evaluate(run_spec(args.index), args.output)


if __name__ == "__main__":
    main()

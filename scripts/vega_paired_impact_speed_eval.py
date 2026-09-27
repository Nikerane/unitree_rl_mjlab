"""Evaluation-only matched-speed comparisons for the saved three-route policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


TASK = "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed"
BOUNDS = (1.5, 2.2)
GOALS = (1.5, 1.85, 2.2)
SPEED_COST = .8493676
EVALUATION_SEED = 20260926
EVALUATION_WORLDS = 384
CHECKPOINTS = (
    {"run_index": 1, "seed": 10,
     "expected_sha256": "5d106461e602fbbfe5a1ab9c97b4d860298bfb3760ec6384a3fdb1dc3eb43f54"},
    {"run_index": 3, "seed": 11,
     "expected_sha256": "89cd86ca904854a39143a173e0a77217ed5b82710b509b0b90a696dd07dcd689"},
    {"run_index": 5, "seed": 12,
     "expected_sha256": "bdb0bc0a7c4445a29b2b75fb06b654d30c44627a07b8bdfaaef9276d36ac1ead"},
)


def checkpoint_spec(index: int) -> dict:
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(len(CHECKPOINTS)):
        raise ValueError("array index must be 0..2")
    return dict(CHECKPOINTS[index])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_command(checkpoint: Path, output: Path, evaluator_script: Path) -> list[str]:
    return [
        sys.executable, str(evaluator_script),
        "--task", TASK,
        "--checkpoint", str(checkpoint),
        "--speed-bounds", str(BOUNDS[0]), str(BOUNDS[1]),
        "--speed-cost", str(SPEED_COST),
        "--seed", str(EVALUATION_SEED),
        "--num-envs", str(EVALUATION_WORLDS),
        "--mode", "paired",
        "--goals", *(str(goal) for goal in GOALS),
        "--output", str(output),
        "--device", "cuda:0",
        "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
        "--measurement-protocol", "common-window",
    ]


def validate_summary(summary: dict, expected_checkpoint_hash: str) -> None:
    """Refuse completion records that drift from the preregistered run contract."""
    try:
        conditions = summary["conditions"]
        roles = [condition["condition_role"] for condition in conditions]
        goals = [condition["goal_mps"] for condition in conditions]
        contracts = [condition["contract"] for condition in conditions]
        measurement_protocols = [condition["measurement"]["protocol"]
                                 for condition in conditions]
        expected_roles = ["autonomous_policy_selector", "preaction_fixed_target",
                          "preaction_fixed_target", "preaction_fixed_target"]
        if (summary["task"] != TASK or summary["seed"] != EVALUATION_SEED
                or summary["actor_status"] != "checkpoint" or summary["mode"] != "paired"
                or summary["learning_updates"] != 0
                or summary["checkpoint_sha256"] != expected_checkpoint_hash
                or summary["initial_state_pairing"] != "verified_policy_vs_preaction_fixed_targets"
                or goals != [None, *GOALS] or roles != expected_roles
                or measurement_protocols != ["common-window"] * 4
                or len(conditions) != 4 or len(summary["comparisons"]) != 6):
            raise ValueError("summary identity, goals, conditions, or pairing marker differs")
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError("paired condition contracts differ")

        contract = contracts[0]
        sampled = contract["sampled_pd"]
        cat = contract["safety"]["cat_soft"]
        impulse = contract["safety"]["substep_impulse"]
        expected_caps = [.369, .246, .738, .369, .246, .0164]
        if (contract["speed_mode"] != "policy"
                or contract["speed_bounds_mps"] != list(BOUNDS)
                or not math.isclose(contract["speed_cost"]["coefficient"], SPEED_COST,
                                    rel_tol=0., abs_tol=1e-12)
                or contract["controller_identity"] != "sampled_vic500_v1"
                or contract["action_type"] != "joint_position_variable_impedance_sampled_pd"
                or sampled["gain_mode"] != "variable"
                or sampled["version"] != "variable_impedance_pd500_v1"
                or cat["imp_limit"] != expected_caps or cat["imp_max_p"] != .2
                or cat["limit"] != 3.1415 or cat["use_vel"] is not True
                or cat["use_impulse"] is not True or cat["vel_detection"] != "substep"
                or impulse["event_window_substeps"] != 200):
            raise ValueError("bounds, cost, sampled controller, or safety settings differ")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("frozen run contract is missing required evidence") from exc
    except ValueError as exc:
        if "frozen run contract" in str(exc):
            raise
        raise ValueError(f"frozen run contract violation: {exc}") from exc


def run_evaluation(index: int, training_campaign: Path, output_root: Path,
                   evaluator_script: Path) -> dict:
    spec = checkpoint_spec(index)
    checkpoint = (Path(training_campaign) / "runs" / f"run-{spec['run_index']}"
                  / "train" / "model_49.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"saved policy checkpoint missing: {checkpoint}")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != spec["expected_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for run-{spec['run_index']}")

    result_dir = Path(output_root) / "evaluations" / f"seed-{spec['seed']}"
    if result_dir.exists():
        raise FileExistsError(f"refusing to overwrite paired evaluation: {result_dir}")
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    command = evaluation_command(checkpoint, result_dir, Path(evaluator_script))
    subprocess.run(command, check=True)

    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("paired evaluator did not produce summary.json")
    summary = json.loads(summary_path.read_text())
    validate_summary(summary, checkpoint_hash)

    manifest = {
        "status": "completed",
        "training_run_index": spec["run_index"],
        "training_seed": spec["seed"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "evaluator_sha256": file_sha256(evaluator_script),
        "evaluation_seed": EVALUATION_SEED,
        "worlds_per_condition": EVALUATION_WORLDS,
        "conditions": [None, *GOALS],
        "measurement_protocol": "common-window",
        "gain_mode": "variable sampled joint impedance",
        "learning_updates": 0,
        "summary": "summary.json",
    }
    (result_dir / "completion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--training-campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluator-script", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_evaluation(args.index, args.training_campaign, args.output, args.evaluator_script)
    print("PAIRED_IMPACT_SPEED_EVAL_COMPLETED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Preparation-only release validator. No simulator imports or execution path.

This validates a proposed six-policy evaluation ledger, not admission, runtime
qualification, checkpoint tensor contents, or scientific completion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_TASK = "Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit"
TASKS = {arm: BASE_TASK + "-Horizon-" + arm for arm in ("native50", "extended200")}
BANKS = (20261041, 2026101601)
CASES = ("original-route-1", "start-minus10mm", "original-route+0", "start-plus10mm", "original-route+1")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_proposal(value: dict) -> dict:
    if value.get("schema") != "z1_horizon_cpu_proposal_v1":
        raise ValueError("wrong preparation schema")
    if value.get("status") != "PREPARATION_ONLY_NOT_ADMITTED":
        raise ValueError("this tool cannot admit or execute a release")
    if value.get("retry_allowed") is not False or value.get("resume_allowed") is not False:
        raise ValueError("retry/resume must be disabled")
    policies = value["policies"]
    if len(policies) != 6 or len({p["policy_id"] for p in policies}) != 6:
        raise ValueError("exactly six distinct policy identities required")
    pairs = {}
    outputs = set()
    for policy in policies:
        arm = policy["arm"]
        if arm not in TASKS or policy["task"] != TASKS[arm]:
            raise ValueError("declared horizon task differs from arm")
        seed = policy["training_seed"]
        if type(seed) is not int:
            raise ValueError("training seed must be an exact integer")
        pairs.setdefault(seed, []).append(arm)
        for field in ("checkpoint", "training_completion", "training_config", "training_manifest", "training_backup_receipt"):
            binding = policy[field]
            if not Path(binding["path"]).is_absolute() or sha256(binding["path"]) != binding["sha256"]:
                raise ValueError("changed training evidence: " + field)
        cells = policy["evaluations"]
        if len(cells) != 3:
            raise ValueError("own CPU32 and both CPU160 banks required")
        kinds = [(c["scope"], c["evaluation_seed"]) for c in cells]
        if kinds[1:] != [("science160", bank) for bank in BANKS] or kinds[0][0] != "own_cpu32":
            raise ValueError("evaluation order/banks differ")
        if type(cells[0]["evaluation_seed"]) is not int:
            raise ValueError("own qualification seed must be declared")
        for cell in cells:
            expected = ["original-route+1"] if cell["scope"] == "own_cpu32" else list(CASES)
            if cell["cases"] != expected or cell["worlds_per_case"] != 32:
                raise ValueError("case population differs")
            path = Path(cell["output_root"])
            if not path.is_absolute() or str(path.resolve()) in outputs or path.exists():
                raise ValueError("output must be distinct, absolute, and unstarted")
            outputs.add(str(path.resolve()))
    if len(pairs) != 3 or any(sorted(arms) != ["extended200", "native50"] for arms in pairs.values()):
        raise ValueError("three matched training-seed pairs required")
    return {"status": "PROPOSAL_STRUCTURE_CHECKED_NOT_ADMITTED", "policies": 6,
            "engineering_starts": 192, "scientific_starts": 1920,
            "banks_are_reused": True, "simulator_imported": False}


if __name__ == "__main__":
    import sys
    print(json.dumps(validate_proposal(json.loads(Path(sys.argv[1]).read_text())), sort_keys=True))

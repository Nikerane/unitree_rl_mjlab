"""Run the artifact contract against all 12 rendered Wave-2 leaves.

The device requirement, the checkpoint/reset binding, the substep-trace contract
and the media properties are only worth anything if something actually executes
them against the real artifacts. Wave 1 had no such driver -- the validator lived
only in the test suite -- so this closes that gap for Wave 2.

Usage:  PYTHONPATH=. python evaluation/results/2026-08-03_wave2_waypoint/validate_wave2_artifacts.py <analysis_revision>
"""

import json
import sys
from pathlib import Path

from evaluation.analysis.fixed_reset_video_library import (
    validate_wave1_policy_artifacts,
)

ROOT = Path("evaluation/results/2026-08-03_wave2_waypoint")
TRAINING_REVISION = "e1a0282c9dc7b0283ae632a46a78debfd80bdf8c"
ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
RESET_DIGEST = "bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319"
SHORT = {"C0": "c0", "G": "g", "P": "p"}


def main(analysis_revision: str) -> int:
    inventory = json.loads((ROOT / "wave2_training_inventory.json").read_text())
    runs = {(r["arm"], r["seed"]): r for r in inventory["runs"]}
    order = [(arm, seed) for arm in ("C0", "G", "P") for seed in (4, 5, 6, 7)]

    failures = []
    for arm, seed in order:
        leaf = ROOT / "videos" / f"wave2_{SHORT[arm]}_seed{seed}"
        expectations = {
            "arm": arm,
            "training_seed": seed,
            "checkpoint_sha256": runs[(arm, seed)]["checkpoint_sha256"],
            "training_revision": TRAINING_REVISION,
            "asset_revision": ASSET_REVISION,
            "analysis_revision": analysis_revision,
            "reset_state_digest": RESET_DIGEST,
        }
        try:
            report = validate_wave1_policy_artifacts(
                leaf, expectations, campaign="wave2"
            )
        except (ValueError, OSError) as error:
            failures.append((arm, seed, str(error)))
            print(f"  {arm}/{seed}  FAIL: {error}")
            continue
        device = report["execution_device"]
        print(
            f"  {arm}/{seed}  PASS  substeps={report['substep_count']:>4} "
            f"steps={report['executed_control_steps']:>2} "
            f"fps={report['measured_fps']:.0f} "
            f"device={device['actual_env_device']}/{device['actual_tensor_device']} "
            f"({device['platform']})"
        )

    print()
    print(f"leaves={len(order)} pass={len(order) - len(failures)} fail={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))

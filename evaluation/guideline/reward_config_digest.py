"""Reward-config digest gate (in-repo, so the preregistration's MATCH rows are reproducible).

Usage:  python evaluation/guideline/reward_config_digest.py     # exit 0 = all digests match

Wave-3 and impulse6 launch gate.

Three jobs:
  (a) re-derive the THREE FROZEN Wave-2 per-arm reward-config digests at this revision and
      prove they still match -- this is what shows the src/ change did not perturb the
      controls P/G/C0 that Wave 3 is compared against;
  (b) derive and print the NEW P+V digest, to be frozen into the preregistration;
  (c) validate all six impulse6 reward cells and bind live S8/D4 to the independently
      reproduced digest from its historical training revision.
"""

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

# Running this file by path otherwise resolves an older editable checkout before this one.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

from mjlab.tasks.registry import load_env_cfg

from src.tasks.hammer.config.z1 import z1_hammer_env_cfg
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT

FROZEN = {
    "C0": (
        dict(cat_impulse=True, event_correct=True, event_linear=True, guideline=True),
        "47d993698852dc939c753d978e41c9124e24c470e0595439f004d2b561d76fb9",
    ),
    "CGate": (
        dict(cat_impulse=True, event_correct=True, event_linear=True, guideline=True,
             gate_reward=True),
        "a5b767b22a77ecfc068eea9885ff88c4bedcd2035e5622627650a20216431e0f",
    ),
    "CProgress": (
        dict(cat_impulse=True, event_correct=True, event_linear=True, guideline=True,
             progress_reward=True),
        "dca539d938e6eb15cb0ececebf78e9d2972870045edf6b52967d16f6945ed264",
    ),
}
NEW = {
    "CProgress-Vel": dict(cat_impulse=True, event_correct=True, event_linear=True,
                          guideline=True, progress_reward=True, cat_soft=True,
                          vel_cat_substep=True),
}
CAPS = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]

# Exact published screen identities.  The S8/D4 centre deliberately reuses the existing
# presentation3 P+V+D4 registration: it is not an alias and must stay task-identical.
SCREEN = {
    "s0d0": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D0",
        0.0,
        0.0,
    ),
    "s0d4": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D4",
        0.0,
        4.0,
    ),
    "s0d16": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S0-D16",
        0.0,
        16.0,
    ),
    "s8d0": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D0",
        8.0,
        0.0,
    ),
    "s8d4": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
        8.0,
        4.0,
    ),
    "s8d16": (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-S8-D16",
        8.0,
        16.0,
    ),
}
PRESENTATION3_PVD4_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4"
)
# Independently reproduced from the registered P+V+D4 task in a detached export of
# ``HISTORICAL_S8D4_TRAINING_REVISION``.  This is deliberately a literal rather than a value
# derived from the live screen config: live S8/D4 drift must make this command fail.
HISTORICAL_S8D4_TRAINING_REVISION = "ba6119c767fe92a8eb4b6131e0c0b0d3c120f0fe"
HISTORICAL_S8D4_REWARD_CONFIG_SHA256 = (
    "cc52cd7319a2845d85da3ea22b685a329c51a927b9edd2ac93d283460323b3d9"
)


def canon(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        fields = sorted(dataclasses.fields(o), key=lambda f: f.name)
        return {f.name: canon(getattr(o, f.name)) for f in fields}
    if isinstance(o, dict):
        return {str(k): canon(o[k]) for k in sorted(o, key=str)}
    if isinstance(o, (list, tuple)):
        return [canon(x) for x in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    if callable(o):
        mod = getattr(o, "__module__", "")
        qn = getattr(o, "__qualname__", type(o).__name__)
        return mod + "." + qn
    return repr(o)


def digest(kw):
    rewards = canon(z1_hammer_env_cfg(**kw).rewards)
    return digest_rewards(rewards)


def digest_rewards(rewards):
    blob = json.dumps(rewards, sort_keys=True, separators=(",", ":"))
    active = {k: v["weight"] for k, v in rewards.items()
              if isinstance(v, dict) and v.get("weight") is not None}
    return hashlib.sha256(blob.encode()).hexdigest(), active


def registered_digest(task):
    """Digest exactly the registered config rendered and trained by a screen arm."""
    return digest_rewards(canon(load_env_cfg(task).rewards))


def normalized_screen_config(task):
    """Erase only the declared S/D treatment so all six cells must otherwise coincide."""
    cfg = copy.deepcopy(load_env_cfg(task))
    cfg.rewards["impact_progress"].weight = 8.0
    cfg.rewards["delivered_impulse"].weight = 4.0
    return canon(cfg)


def screen_cell_ok(task, impact_weight, delivered_weight, baseline):
    """Verify the six cells differ solely by their matrix-declared reward weights."""
    cfg = load_env_cfg(task)
    params = cfg.metrics["cat_soft"].params
    return (
        cfg.rewards["impact_progress"].weight == impact_weight
        and cfg.rewards["delivered_impulse"].weight == delivered_weight
        and cfg.rewards["r_waypoint_progress"].weight == 8.0
        and "r_gate" not in cfg.rewards
        and params["use_vel"] is True
        and params["max_p"] == 0.5
        and params["vel_detection"] == "substep"
        and params["imp_max_p"] == 0.0
        and normalized_screen_config(task) == baseline
    )


bad = 0
print("=== frozen Wave-2 control digests (must still match at this revision) ===")
for arm, (kw, want) in FROZEN.items():
    got, active = digest(kw)
    ok = got == want
    bad += 0 if ok else 1
    print("  %-14s %s  %s" % (arm, got, "MATCH" if ok else "*** MISMATCH ***"))
    print("                 terms=%d %s" % (len(active), json.dumps(active, sort_keys=True)))

print()
print("=== new Wave-3 treatment digest (to freeze into the preregistration) ===")
for arm, kw in NEW.items():
    got, active = digest(kw)
    print("  %-14s %s" % (arm, got))
    print("                 terms=%d %s" % (len(active), json.dumps(active, sort_keys=True)))
    p_digest, p_active = digest(FROZEN["CProgress"][0])
    same = active == p_active
    print("                 reward set identical to frozen P: %s" % same)
    if not same:
        bad += 1
    # The reward CONFIG digest must also match P: P+V changes only the CaT hook, which is a
    # metrics term, not a reward term. A differing reward digest would mean the treatment
    # leaked into the reward function.
    print("                 reward-config digest identical to frozen P: %s" % (got == p_digest))
    if got != p_digest:
        bad += 1

print()
print("=== impulse6 registered reward-config digests (six-cell screen) ===")
screen_baseline = normalized_screen_config(SCREEN["s0d0"][0])
screen_digests = {}
for short, (task, impact_weight, delivered_weight) in SCREEN.items():
    got, active = registered_digest(task)
    screen_digests[short] = got
    ok = screen_cell_ok(task, impact_weight, delivered_weight, screen_baseline)
    bad += 0 if ok else 1
    print(
        "  %-6s %s  task=%s  S=%g D=%g  %s"
        % (short, got, task, impact_weight, delivered_weight, "MATCH" if ok else "*** MISMATCH ***")
    )
    print("                 terms=%d %s" % (len(active), json.dumps(active, sort_keys=True)))

centre_task = SCREEN["s8d4"][0]
centre_ok = centre_task == PRESENTATION3_PVD4_TASK
bad += 0 if centre_ok else 1
print(
    "  s8d4 existing presentation3 P+V+D4 identity: %s"
    % ("MATCH" if centre_ok else "*** MISMATCH ***")
)
historical_centre_ok = (
    screen_digests["s8d4"] == HISTORICAL_S8D4_REWARD_CONFIG_SHA256
)
bad += 0 if historical_centre_ok else 1
print(
    "  s8d4 historical %s reward-config digest: %s  %s"
    % (
        HISTORICAL_S8D4_TRAINING_REVISION[:7],
        HISTORICAL_S8D4_REWARD_CONFIG_SHA256,
        "MATCH" if historical_centre_ok else "*** MISMATCH ***",
    )
)

print()
caps_ok = list(map(float, IMP_J_LIMIT)) == CAPS
bad += 0 if caps_ok else 1
print("  caps IMP_J_LIMIT = %s  %s" % (IMP_J_LIMIT, "MATCH" if caps_ok else "*** MISMATCH ***"))
sys.exit(3 if bad else 0)

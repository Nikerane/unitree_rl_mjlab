"""Reward-config digest gate (in-repo, so the preregistration's MATCH rows are reproducible).

Usage:  python evaluation/guideline/reward_config_digest.py     # exit 0 = all digests match

Wave-3 launch gate.

Two jobs:
  (a) re-derive the THREE FROZEN Wave-2 per-arm reward-config digests at this revision and
      prove they still match -- this is what shows the src/ change did not perturb the
      controls P/G/C0 that Wave 3 is compared against;
  (b) derive and print the NEW P+V digest, to be frozen into the preregistration.
"""

import dataclasses
import hashlib
import json
import sys

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
    blob = json.dumps(rewards, sort_keys=True, separators=(",", ":"))
    active = {k: v["weight"] for k, v in rewards.items()
              if isinstance(v, dict) and v.get("weight") is not None}
    return hashlib.sha256(blob.encode()).hexdigest(), active


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
caps_ok = list(map(float, IMP_J_LIMIT)) == CAPS
bad += 0 if caps_ok else 1
print("  caps IMP_J_LIMIT = %s  %s" % (IMP_J_LIMIT, "MATCH" if caps_ok else "*** MISMATCH ***"))
sys.exit(3 if bad else 0)

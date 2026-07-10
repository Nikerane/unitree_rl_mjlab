"""C2 enforcement gate (IMPULSE_CAT_IMPL_PLAN.md C2 + spec amendments 2026-07-06; probe
recalibrated 2026-07-10, plan commit 74672f6).

Probe calibration (MEASURED on the 2026-07-06 windup NEAR_NAIL pose, this task): heights run
MOST-VIOLENT-FIRST, which under this geometry means ASCENDING (0.06, 0.10, 0.15) — a LOWER apex
gives a LARGER strike impulse here (measured worst-joint Λ: 0.047 @ 0.06 > 0.043 @ 0.10 >
0.036 @ 0.15 N·m·s), the opposite of the original descending assumption. The graded-δ gate [2]
needs the biggest excess FIRST: the first over-limit strike seeds c_max, so the later, smaller
excesses grade 0 < δ < max_p; an increasing-excess order keeps every sample above the lagging
τ=0.95 EMA and saturates δ=max_p on all strikes (measured, both orders). PROBE_SCALE = 0.02 sits
below the measured reference-strike ceiling (binding_ratio ≈ 0.029 of J_limit; corroborated by
derive_impulse_thresholds.py's 3.8% p95 binding-ness) so every strike crosses the probe limit —
the original 0.05 sat ABOVE the ceiling and no strike ever violated.

Per imp_max_p ∈ {0.25, 0.5}:
  PROBE pass (limit × 0.02, heights most-violent-first — machinery hard gates):
    [1] c_max bounded: finite, above the 1e-3 seed on ≥1 column, ≤ 100× worst observed excess.
    [2] graded δ: ≥1 over-limit strike with 0 < δ < max_p (non-degenerate grading; the FIRST
        over-limit strike seeds c_max ⇒ δ=max_p there is EXPECTED, not a failure).
    [4a] regression tripwire: nail depths identical to the log-only baseline run (δ has no physics
        pathway in playback — this catches future wiring that mutates sim state).
    [4b] δ delivery: probe pass has cat_delta > 0 on ≥1 contact step; log-only baseline δ ≡ 0.
  REPORT pass (real IMP_J_LIMIT — statistics only, no failure):
    [3] band_fraction (strikes with worst-joint Λ in [0.8·J,J]), binding_ratio (max Λ_j/J_limit_j).
  REWARD-SHARE table (once, log-only cfg): per-term weighted episode return on the reference
  strikes incl. delivered_impulse's share of the positive total — fulfils the config's "C2 tunes
  the weight" promise. Hard-gate: delivered_impulse fires, positive, finite. The weight decision
  itself is the USER's — surface the share and stop short of changing weight.
Usage: $PY docs/research/reward-design/c2_enforcement_gate.py   (exits 1 on FAIL)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parent))  # reward-design is not a package
from reward_design_util import run_reference_strikes  # noqa: E402

MAX_PS = (0.25, 0.5)
HEIGHTS = (0.06, 0.10, 0.15)  # MOST-VIOLENT-FIRST under the windup pose — see module docstring
PROBE_SCALE = 0.02  # below the measured ~0.029 binding-ratio ceiling — see module docstring


def build_cfg(imp_max_p: float):
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  cfg.metrics["cat_soft"].params["imp_max_p"] = imp_max_p
  return cfg


def main() -> int:
  failures: list[str] = []
  base = run_reference_strikes(build_cfg(0.0), heights=HEIGHTS)

  print("=== REWARD SHARE (log-only cfg, reference strikes) ===")
  total_pos = sum(v for v in base["reward_terms"].values() if v > 0)
  for name, v in sorted(base["reward_terms"].items(), key=lambda kv: -abs(kv[1])):
    share = (v / total_pos * 100) if total_pos > 0 else float("nan")
    print(f"  {name:24s} {v:12.4f}   {share:6.1f}% of positive")
  di = base["reward_terms"].get("delivered_impulse", 0.0)
  if not (di > 0 and torch.isfinite(torch.tensor(di))):
    failures.append(f"delivered_impulse term degenerate on reference strikes: {di}")
  if max(base["max_delta"]) != 0.0:
    failures.append(f"log-only baseline has nonzero δ: {base['max_delta']}")

  for mp in MAX_PS:
    probe = run_reference_strikes(build_cfg(mp), heights=HEIGHTS, limit_override=PROBE_SCALE)
    print(f"\n=== PROBE imp_max_p={mp} (limit × {PROBE_SCALE}, heights most-violent-first) ===")
    print(f"excess: {probe['max_excess']}\nδ:      {probe['max_delta']}\nc_max:  {probe['cmax']}")
    cmax = torch.as_tensor(probe["cmax"])
    worst_excess = max(probe["max_excess"])
    if not torch.isfinite(cmax).all() or (cmax <= 1e-3).all():
      failures.append(f"[{mp}] c_max degenerate (pinned at seed / non-finite): {probe['cmax']}")
    if worst_excess > 0 and (cmax > 100 * worst_excess).any():
      failures.append(f"[{mp}] c_max single-event-spiked (>100× worst excess {worst_excess:.3g})")
    over_deltas = [d for e, d in zip(probe["max_excess"], probe["max_delta"]) if e > 0]
    if len(over_deltas) < 2:
      failures.append(f"[{mp}] probe produced <2 over-limit strikes — probe scale too loose")
    elif not any(0.0 < d < mp - 1e-6 for d in over_deltas):
      failures.append(f"[{mp}] δ degenerate: no over-limit strike with 0 < δ < max_p: {over_deltas}")
    if not any(d > 0 for d in probe["max_delta"]):
      failures.append(f"[{mp}] probe never delivered δ>0 on contact — hook wiring broken")
    if not torch.allclose(torch.as_tensor(probe["depth"]), torch.as_tensor(base["depth"]), atol=1e-6):
      failures.append(f"[{mp}] enforcement branch mutated sim state: {probe['depth']} vs {base['depth']}")

    report = run_reference_strikes(build_cfg(mp), heights=HEIGHTS)
    print(f"=== REPORT imp_max_p={mp} (real caps) ===")
    print(f"band_fraction [0.8·J,J]: {report['band_fraction']:.3f}")
    print(f"binding_ratio max Λ/J:   {report['binding_ratio']:.3f}  (reference strikes are gentle by design; the LEARNED policy is what approaches the cap — vacuous-check at C3 analysis)")

  if failures:
    print("\nC2 GATE FAIL:\n  - " + "\n  - ".join(failures))
    return 1
  print(f"\nC2 GATE PASS for imp_max_p ∈ {MAX_PS}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

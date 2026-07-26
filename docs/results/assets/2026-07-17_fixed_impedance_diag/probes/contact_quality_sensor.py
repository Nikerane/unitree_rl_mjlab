"""CPU qualification probe for the multi-slot first-contact quality sensor.

Runs the existing scripted reference independently at 8, 16, and 64 retained
contact slots. The smallest non-overflowing candidate is accepted only when its
latched contact centroid and radial error agree with the next larger candidate
to 1e-6 m. Contact-frame normal-force sign and ``found`` count semantics are
measured rather than assumed.

Run:
  PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
    docs/results/assets/2026-07-17_fixed_impedance_diag/probes/contact_quality_sensor.py \
    --out /tmp/contact_quality_sensor_cpu.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.contact_quality import contact_point_quality
from src.tasks.hammer.mdp.first_strike import _ENV_FIRST_STRIKE_ATTR
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.nail_block import NAIL_TOP_SITE_NAME


CANDIDATE_SLOTS = (8, 16, 64)
AGREEMENT_ATOL_M = 1e-6
HOLD_STEPS = 10


def _quality_sensor_cfg(cfg):
  matches = [
    sensor for sensor in cfg.scene.sensors
    if sensor.name == "hammer_nail_quality"
  ]
  if len(matches) != 1:
    raise RuntimeError(
      "expected exactly one hammer_nail_quality sensor config, got "
      f"{len(matches)}"
    )
  return matches[0]


def run_candidate(candidate_slots: int) -> dict:
  started = time.perf_counter()
  cfg = z1_hammer_env_cfg(
    play=True,
    cat_impulse=True,
    event_correct=True,
  )
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  quality_cfg = _quality_sensor_cfg(cfg)
  quality_cfg.num_slots = candidate_slots

  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    quality_sensor = env.scene["hammer_nail_quality"]
    tracker = getattr(env, _ENV_FIRST_STRIKE_ATTR)

    robot_cfg = SceneEntityCfg(
      "robot", site_names=(HAMMER_HEAD_SITE_NAME,)
    )
    nail_cfg = SceneEntityCfg(
      "nail_block", site_names=(NAIL_TOP_SITE_NAME,)
    )
    robot_cfg.resolve(env.scene)
    nail_cfg.resolve(env.scene)

    def head() -> torch.Tensor:
      return robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)

    def nail_top() -> torch.Tensor:
      return nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    axis = torch.tensor(
      [0.0, 0.0, -1.0], dtype=torch.float32, device=env.device
    )
    radius = float(
      env.sim.mj_model.geom("nail_block/nail_head").size[0]
    )
    contact_substeps = 0
    overflow_n = 0
    quality_valid_n = 0
    max_reported_found = 0.0
    max_active_slots = 0
    found_finite = True
    found_integral = True
    found_nonnegative = True
    found_patterns: Counter[tuple[int, ...]] = Counter()
    normal_force_sum = 0.0
    normal_force_positive_sum = 0.0
    normal_force_negative_sum = 0.0
    normal_force_positive_n = 0
    normal_force_negative_n = 0

    original_compute_substep = env.metrics_manager.compute_substep

    def capture_substep() -> None:
      nonlocal contact_substeps
      nonlocal overflow_n
      nonlocal quality_valid_n
      nonlocal max_reported_found
      nonlocal max_active_slots
      nonlocal found_finite
      nonlocal found_integral
      nonlocal found_nonnegative
      nonlocal normal_force_sum
      nonlocal normal_force_positive_sum
      nonlocal normal_force_negative_sum
      nonlocal normal_force_positive_n
      nonlocal normal_force_negative_n

      original_compute_substep()
      data = quality_sensor.data
      found = data.found[0].detach()
      finite = torch.isfinite(found)
      found_finite = found_finite and bool(finite.all())
      found_integral = found_integral and bool(
        torch.equal(found, found.round())
      )
      found_nonnegative = found_nonnegative and bool((found >= 0).all())
      if not bool(finite.all()):
        return

      max_found = float(found.max())
      max_reported_found = max(max_reported_found, max_found)
      active_slots = int((found > 0).sum())
      max_active_slots = max(max_active_slots, active_slots)
      if active_slots == 0:
        return

      contact_substeps += 1
      found_patterns[tuple(int(value) for value in found.tolist())] += 1
      if max_found > candidate_slots:
        overflow_n += 1

      _, _, _, valid, _ = contact_point_quality(
        found=data.found,
        force_contact=data.force,
        position_w=data.pos,
        nail_top_w=nail_top(),
        nail_axis_w=axis,
        nail_radius_m=radius,
        num_slots=candidate_slots,
      )
      quality_valid_n += int(valid[0])

      normal_force = data.force[0, :, 0]
      matched_force = normal_force[found > 0]
      normal_force_sum += float(matched_force.sum())
      positive = matched_force[matched_force > 0]
      negative = matched_force[matched_force < 0]
      normal_force_positive_sum += float(positive.sum())
      normal_force_negative_sum += float(negative.sum())
      normal_force_positive_n += int(positive.numel())
      normal_force_negative_n += int(negative.numel())

    env.metrics_manager.compute_substep = capture_substep
    env.reset()
    reference = SingleStrikeReference(1, env.device)
    reference.update(
      head(),
      nail_top(),
      torch.zeros(1, dtype=torch.long, device=env.device),
    )
    playback_length = reference.playback_length()
    for step in range(1, playback_length + HOLD_STEPS + 1):
      target = reference.playback_target(min(step, playback_length))
      action = (
        (target - head()) / Z1_HAMMER_DELTA_POS_SCALE
      ).clamp(-1.0, 1.0)
      env.step(action)
      if bool(env.reset_terminated.any()):
        break
    env.metrics_manager.compute_substep = original_compute_substep

    patterns_json = [
      {"found": list(pattern), "substeps": count}
      for pattern, count in sorted(found_patterns.items())
    ]
    return {
      "candidate_slots": candidate_slots,
      "contact_substeps": contact_substeps,
      "max_reported_found": max_reported_found,
      "max_active_slots": max_active_slots,
      "overflow_n": overflow_n,
      "quality_valid_n": quality_valid_n,
      "contact_point_w": [
        float(value) for value in tracker.contact_point_w[0]
      ],
      "contact_error_m": float(tracker.contact_error_m[0]),
      "contact_quality": float(tracker.contact_quality[0]),
      "contact_quality_valid": bool(tracker.contact_quality_valid[0]),
      "contact_quality_overflow": bool(
        tracker.contact_quality_overflow[0]
      ),
      "contact_normal_axiality": float(
        tracker.contact_normal_axiality[0]
      ),
      "normal_force_sum": normal_force_sum,
      "normal_force_positive_sum": normal_force_positive_sum,
      "normal_force_negative_sum": normal_force_negative_sum,
      "normal_force_positive_n": normal_force_positive_n,
      "normal_force_negative_n": normal_force_negative_n,
      "found_finite": found_finite,
      "found_integral": found_integral,
      "found_nonnegative": found_nonnegative,
      "found_patterns": patterns_json,
      "wall_seconds": time.perf_counter() - started,
    }
  finally:
    env.close()


def _agrees(left: dict, right: dict) -> bool:
  left_point = torch.tensor(left["contact_point_w"])
  right_point = torch.tensor(right["contact_point_w"])
  centroid_ok = bool(
    torch.allclose(
      left_point,
      right_point,
      rtol=0.0,
      atol=AGREEMENT_ATOL_M,
    )
  )
  error_ok = math.isclose(
    left["contact_error_m"],
    right["contact_error_m"],
    rel_tol=0.0,
    abs_tol=AGREEMENT_ATOL_M,
  )
  return centroid_ok and error_ok


def qualify(rows: list[dict]) -> tuple[int | None, list[dict], list[str]]:
  agreement = []
  for left, right in zip(rows[:-1], rows[1:], strict=True):
    agreement.append(
      {
        "candidate_slots": left["candidate_slots"],
        "next_larger_slots": right["candidate_slots"],
        "centroid_and_error_agree_atol_1e-6_m": _agrees(left, right),
      }
    )

  failures: list[str] = []
  for row in rows:
    slots = row["candidate_slots"]
    if row["contact_substeps"] == 0:
      failures.append(f"{slots}: scripted reference made no quality-sensor contact")
    if not (
      row["found_finite"]
      and row["found_integral"]
      and row["found_nonnegative"]
    ):
      failures.append(f"{slots}: found count was nonfinite/nonintegral/negative")
    if row["normal_force_positive_n"] == 0:
      failures.append(f"{slots}: no positive contact-frame normal force")
    if row["normal_force_negative_n"] > 0:
      failures.append(
        f"{slots}: observed negative contact-frame normal force on retained contact"
      )
    # Invalid quality is expected when a smaller candidate overflows; that
    # candidate is rejected below and the probe may qualify a larger fallback.
    if row["overflow_n"] == 0 and not row["contact_quality_valid"]:
      failures.append(f"{slots}: onset contact-quality snapshot was invalid")

  if rows[-1]["overflow_n"] > 0:
    failures.append("64: contact slots overflowed; no larger qualified fallback")

  chosen = None
  for index, row in enumerate(rows[:-1]):
    if (
      row["overflow_n"] == 0
      and rows[index + 1]["overflow_n"] == 0
      and agreement[index]["centroid_and_error_agree_atol_1e-6_m"]
    ):
      chosen = row["candidate_slots"]
      break
  if chosen is None:
    failures.append(
      "no candidate below 64 passed overflow plus next-larger agreement"
    )
  return chosen, agreement, failures


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  rows = [run_candidate(candidate) for candidate in CANDIDATE_SLOTS]
  chosen, agreement, failures = qualify(rows)
  payload = {
    "probe": "first-contact-quality-sensor-cpu",
    "device": "cpu",
    "candidate_order": list(CANDIDATE_SLOTS),
    "agreement_atol_m": AGREEMENT_ATOL_M,
    "candidates": rows,
    "agreement": agreement,
    "chosen_num_slots": chosen,
    "pass": not failures,
    "failures": failures,
  }
  output_path = Path(args.out)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(payload, indent=2, sort_keys=True))
  print(f"[contact-quality-sensor] wrote {output_path}")
  return 0 if not failures else 1


if __name__ == "__main__":
  raise SystemExit(main())

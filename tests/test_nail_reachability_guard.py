"""Recurrence guard for the CLAW-DRIVE exploit class (2026-07-14).

The Lightning smoke found a 50-iter policy driving the nail to full-depth SUCCESS with the forked
claw — a NON-head geom — so the head-only impulse machinery (contact sensor + delivered accumulator
+ Λ constraint) read Λ≡delivered≡0 while success=1.0. Root cause: a collision geom that (a) could
drive the nail but (b) was NOT counted as a head strike. The fix deleted the claw collision and put
the nail on a channel only the head shares.

This guard makes the fix permanent and, more importantly, protects the PLANNED future head-mesh
swap (a real cross-peen model). The invariant it enforces:

    {geoms that can collide with nail_head}  ⊆  {geoms the contact sensor counts as a strike}

If a future head swap puts any non-head striking surface (e.g. a cross-peen wedge named something
other than the sensor's `hammer_head_.*` pattern) on the nail collision channel, this test FAILS
before any training — turning a silent, GPU-run-costing exploit into a red unit test. The counted
pattern is read from the LIVE ContactSensorCfg, so it tracks any change to the sensor definition.
"""

from __future__ import annotations

import re

import pytest

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

pytestmark = pytest.mark.integration

_NAIL_GEOM = "nail_block/nail_head"


def _nail_reachable_and_counted() -> tuple[list[str], re.Pattern, list[str]]:
  """Build the impulse env and return (nail-reachable geom names, counted-strike pattern,
  uncounted-but-reachable geom names) from the COMPILED model — the ground truth, not the XML text."""
  cfg = z1_hammer_env_cfg(cat_impulse=True)
  cfg.scene.num_envs = 1
  # Derive the counted-strike pattern from the sensor cfg itself (no hardcoded copy to drift).
  sensor = next(s for s in cfg.scene.sensors if s.name == "hammer_nail_contact")
  pat = re.compile(f"{sensor.primary.entity}/{sensor.primary.pattern}")
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    m = env.sim.mj_model
    nail = m.geom(_NAIL_GEOM)
    # .item() before int(): m.geom(name).contype is a length-1 array; int() on an ndim>0 array is
    # a numpy DeprecationWarning locally and a hard ERROR on newer numpy (Lightning box, 2026-07-14).
    nt, na = int(nail.contype.item()), int(nail.conaffinity.item())
    reachable = [
      m.geom(i).name
      for i in range(m.ngeom)
      if m.geom(i).name != _NAIL_GEOM
      and ((int(m.geom_contype[i]) & na) or (nt & int(m.geom_conaffinity[i])))
    ]
  finally:
    env.close()
  uncounted = [g for g in reachable if not pat.fullmatch(g)]
  return reachable, pat, uncounted


def test_only_counted_head_geoms_can_drive_the_nail():
  reachable, pat, uncounted = _nail_reachable_and_counted()
  # The head must actually be able to strike (guard against over-restriction that silently disables
  # all contact — the opposite failure, a nail nothing can reach).
  assert reachable, (
    f"no geom can collide with {_NAIL_GEOM} — the head can never strike it (over-restricted "
    "collision channels; the nail is unreachable)."
  )
  # THE guard: every nail-reachable geom must be a counted head strike.
  assert not uncounted, (
    f"CLAW-DRIVE GUARD FAILED: geom(s) can drive the nail but are NOT counted as a head strike: "
    f"{uncounted}. A policy could drive the nail with these and register success while the impulse "
    f"machinery reads Λ=0 (the claw-drive exploit). Fix EITHER: name them to match the sensor "
    f"pattern {pat.pattern!r} AND keep them on the nail collision channel, OR take them OFF the "
    f"nail collision channel (contype/conaffinity) so they physically cannot reach the nail — the "
    f"way the claw and shaft are handled now."
  )

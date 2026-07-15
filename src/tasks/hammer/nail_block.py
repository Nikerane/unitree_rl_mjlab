"""Nail + block scene entity for the hammer task.

The nail slides downward (-Z) into the block when struck.
The block is fixed to the world (no joint). Both are loaded from
the standalone nail_block_scene.xml in safe_impact_manipulation.
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

_SCENE_XML: Path = (
  Path(__file__).resolve().parents[4]   # up to ~/repos
  / "safe_impact_manipulation"
  / "hammer_z1_env"
  / "assets"
  / "nail_block_scene.xml"
)

assert _SCENE_XML.exists(), f"Nail/block scene XML not found: {_SCENE_XML}"


def get_nail_block_spec() -> mujoco.MjSpec:
  """Return MjSpec for the nail+block scene entity."""
  return mujoco.MjSpec.from_file(str(_SCENE_XML))


def get_nail_block_entity_cfg() -> EntityCfg:
  """Return a fresh EntityCfg for the nail+block.

  The nail starts at qpos=0 (not driven). Goal is qpos=0.032 (fully driven:
  nail head flush with the block top — geometry fix 6a, 2026-06-10).
  """
  return EntityCfg(
    spec_fn=get_nail_block_spec,
    articulation=None,
  )


# Scene geometry names — referenced by SceneEntityCfg in mdp terms.
NAIL_SLIDE_JOINT_NAME = "nail_slide"
NAIL_TOP_SITE_NAME = "nail_top"

# Nail fully-driven depth (metres) — goal for the task.
# 0.075 -> 0.032 (2026-06-10, docs/archive/FUTURE_UPDATES.md 6a): with block top at z=0.060 and
# the head bottom at 0.092 - qpos, full drive now ends with the head flush with
# the block surface instead of passing through it. Matches XML range "0 0.032".
NAIL_GOAL_DEPTH: float = 0.032
# Success threshold (metres). Re-pinned 0.027 -> 0.030 (2026-07-15): the neck-drive fix
# makes the flat FACE the only geom that can drive the nail, and the strengthened
# post-degeneracy reference now drives the nail to its 0.032 physical cap in a single
# strike (playback_reference.py, all heights) rather than the old weak-reference 28.3 mm.
# So single-strike-reachability no longer pins the bar low: 0.030 ~= 0.94 x the 0.032 cap
# requires a near-full drive and correctly FAILS a weak strike (e.g. the v2 none_seed1
# that stopped at 28.7 mm), while staying single-strike-reachable.
# History: 0.030 -> 0.027 (2026-06-17) held only while the best SHAPED single strike
# reached 28.3 mm (old weak reference); that constraint is now stale.
# CAVEAT: face-only trained policies (post neck-fix) are untested; if a retrain shows they
# cap below ~30 mm, lower this toward the observed best strike -- keep threshold <= best
# clean strike, else the reference and the completion bonus pull in opposite directions.
# nail_driven's Gaussian still centres on the goal (0.032) and per-episode max nail_depth
# stays the real strike-quality metric (already in eval_impulse summary.csv).
# See OPEN_QUESTIONS.md Q1 and docs/archive/REAL_HAMMER_PLAN.md.
NAIL_SUCCESS_THRESHOLD: float = 0.030

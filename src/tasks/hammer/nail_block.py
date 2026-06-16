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
# 0.075 -> 0.032 (2026-06-10, FUTURE_UPDATES 6a): with block top at z=0.060 and
# the head bottom at 0.092 - qpos, full drive now ends with the head flush with
# the block surface instead of passing through it. Matches XML range "0 0.032".
NAIL_GOAL_DEPTH: float = 0.032
# Success threshold (metres). Re-pinned 0.030 -> 0.027 (2026-06-17): with the real
# claw-hammer (0.5 kg head) the best clean SHAPED single strike reaches 28.3 mm
# (playback_reference.py @ approach 0.06 m), below the old 0.030 line. The RL reward
# is anchored to a single-strike reference, so success must be single-strike-reachable
# or the reference and the completion bonus pull in opposite directions; 0.027 =
# 0.95 x best strike, leaving ~1.3 mm margin for an imperfect trained swing.
# nail_driven's Gaussian still centres on the goal (0.032), so the policy keeps driving
# deeper after success and the per-episode max-depth distribution is the real Q1 metric.
# Invariant: press-stall < threshold <= best clean strike (here the press slow-succeeds
# at ~step 93, so the reward design, not the threshold, must out-score it).
# See OPEN_QUESTIONS.md Q1 and REAL_HAMMER_PLAN.md.
NAIL_SUCCESS_THRESHOLD: float = 0.027

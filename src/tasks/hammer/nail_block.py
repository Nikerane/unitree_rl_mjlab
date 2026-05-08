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

  The nail starts at qpos=0 (not driven). Goal is qpos=0.075 (fully driven).
  """
  return EntityCfg(
    spec_fn=get_nail_block_spec,
    articulation=None,
  )


# Scene geometry names — referenced by SceneEntityCfg in mdp terms.
NAIL_SLIDE_JOINT_NAME = "nail_slide"
NAIL_TOP_SITE_NAME = "nail_top"

# Nail fully-driven depth (metres) — goal for the task.
NAIL_GOAL_DEPTH: float = 0.075
# Success threshold: nail driven more than this fraction of goal depth.
NAIL_SUCCESS_THRESHOLD: float = 0.07

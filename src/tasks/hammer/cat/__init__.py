"""Faithful soft γ(1−δ) CaT (Chane-Sane et al., IROS 2024) for the Z1 hammer task.

C0 layer (this package, pure δ-math + constraint funcs); the env-coupled ConstraintManager hook
(C3) and the rsl_rl learner adapter (C1/C2, under src/tasks/hammer/rl/) build on top. See
docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md.
"""

from .constraint_manager import CaT
from .constraints import joint_velocity_excess
from .hook import CatSoftHook

__all__ = ["CaT", "joint_velocity_excess", "CatSoftHook"]

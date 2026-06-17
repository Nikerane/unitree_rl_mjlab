"""Shared extras keys for the soft-CaT env-hook <-> CatPPO handshake.

Kept in one dependency-free place so the env side (CatSoftHook, writes these into env.extras) and the
learner side (CatPPO.process_env_step, reads them) can never drift apart.
"""

CAT_DELTA_KEY = "cat_delta"   # δ ∈ [0, max_p], shape (B,)
CAT_R_POS_KEY = "cat_r_pos"   # dt-scaled sum of the positive reward terms, shape (B,)

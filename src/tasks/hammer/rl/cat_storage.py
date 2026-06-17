"""Float-dones rollout storage for soft CaT (C1).

rsl_rl's ``RolloutStorage`` allocates the dones buffer as ``.byte()`` (rollout_storage.py:149) and
writes it with ``copy_`` (line 180), which truncates the continuous CaT termination probability
δ ∈ (0,1) to 0 (only an exact 1.0 survives). The dual-mask GAE (``CatPPO``, C2) needs δ intact.

This subclass keeps the inherited HARD ``dones`` (byte {0,1} — used for episode reset, episode-length
logging, and recurrent trajectory masks) and ADDS a float ``soft_dones`` buffer carrying δ. The
dual-mask bootstrap then uses ``(1 − soft_dones)·(1 − dones)`` (Decisions 3–5 of
FAITHFUL_SOFT_CAT_IMPL_PLAN.md). ``CatPPO.process_env_step`` sets ``transition.soft_dones = δ``; we
write it un-truncated here.
"""

from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage


class CatRolloutStorage(RolloutStorage):
  """RolloutStorage with an extra float ``soft_dones`` (δ) buffer alongside the byte hard ``dones``."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # Float buffer for the soft CaT δ; the inherited byte ``self.dones`` would truncate it.
    self.soft_dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    # Write δ BEFORE super() increments self.step. If a step set no soft δ (e.g. a non-CaT run), fall
    # back to the hard done: the dual mask (1−soft)(1−hard) then collapses to the stock single mask,
    # so a CatRolloutStorage with no CaT wiring behaves identically to the stock storage.
    soft = getattr(transition, "soft_dones", None)
    src = transition.dones if soft is None else soft
    self.soft_dones[self.step].copy_(src.view(-1, 1))
    super().add_transition(transition)

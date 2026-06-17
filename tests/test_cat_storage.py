"""C1 unit tests: the float ``soft_dones`` buffer preserves the CaT δ that the stock byte buffer
truncates, while the inherited hard ``dones`` stays {0,1} for reset/logging.
"""

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage

from src.tasks.hammer.rl.cat_storage import CatRolloutStorage


def _make(cls, num_envs: int = 4, T: int = 2):
  obs = TensorDict({"policy": torch.zeros(num_envs, 3)}, batch_size=[num_envs])
  return cls("rl", num_envs, T, obs, [2], device="cpu")


def _transition(num_envs: int, hard, soft=None, n_dist: int = 2):
  t = RolloutStorage.Transition()
  t.observations = TensorDict({"policy": torch.zeros(num_envs, 3)}, batch_size=[num_envs])
  t.actions = torch.zeros(num_envs, 2)
  t.rewards = torch.zeros(num_envs)
  t.dones = torch.as_tensor(hard)
  t.values = torch.zeros(num_envs, 1)
  t.actions_log_prob = torch.zeros(num_envs)
  t.distribution_params = tuple(torch.zeros(num_envs, 2) for _ in range(n_dist))
  if soft is not None:
    t.soft_dones = torch.as_tensor(soft)
  return t


def test_stock_dones_is_byte_and_truncates_delta():
  s = _make(RolloutStorage)
  assert s.dones.dtype == torch.uint8  # the truncation hazard
  s.dones[0].copy_(torch.full((4, 1), 0.37))
  assert torch.equal(s.dones[0], torch.zeros(4, 1, dtype=torch.uint8))  # 0.37 -> 0


def test_cat_soft_dones_is_float_and_preserves_delta():
  s = _make(CatRolloutStorage)
  assert s.soft_dones.dtype == torch.float32
  assert s.dones.dtype == torch.uint8  # hard channel kept byte for reset/logging
  s.soft_dones[0].copy_(torch.full((4, 1), 0.37))
  assert torch.allclose(s.soft_dones[0], torch.full((4, 1), 0.37))


def test_add_transition_writes_soft_delta_untruncated():
  n = 4
  s = _make(CatRolloutStorage, num_envs=n)
  t = _transition(n, hard=[1, 0, 0, 0], soft=[1.0, 0.37, 0.0, 0.62])  # env0 real-terminated
  s.add_transition(t)
  assert torch.allclose(s.soft_dones[0].squeeze(-1), torch.tensor([1.0, 0.37, 0.0, 0.62]))
  assert torch.equal(s.dones[0].squeeze(-1), torch.tensor([1, 0, 0, 0], dtype=torch.uint8))
  assert s.step == 1


def test_add_transition_falls_back_to_hard_done_when_no_soft():
  n = 3
  s = _make(CatRolloutStorage, num_envs=n)
  t = _transition(n, hard=[0, 1, 0], soft=None, n_dist=1)  # no soft δ set -> fall back to hard
  s.add_transition(t)
  assert torch.allclose(s.soft_dones[0].squeeze(-1), torch.tensor([0.0, 1.0, 0.0]))

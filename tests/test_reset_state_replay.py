"""D2 — exact-episode-replay contract for schema-v3 sampled episodes.

Downstream (Codex) must be able to replay ONE selected stochastic evaluation
episode and reproduce it EXACTLY from the banked artifact alone. Before this,
a schema-v3 sampled episode recorded ``action_tape`` + ``physical`` + event
traces but not the reset state that produced the episode -- replaying the
action tape from a FRESH (re-randomized) reset would not reproduce the same
physics.

This adds three per-episode fields (``reset_contract_version``,
``reset_state``, ``reset_state_digest``) and the replay contract:

  1. regenerate the reset state from the banked regeneration inputs;
  2. verify the recomputed full-state digest equals the banked
     ``reset_state_digest`` -- BEFORE stepping;
  3. only then replay the stored ``action_tape``;
  4. require the resulting physical trace digest to equal the banked
     ``trace_digest``.

Reuses the fixed-action-tape replay machinery already proven in
``test_eval_impulse_hook.py`` (``_SampledTraceCollector`` driven directly,
one env, one control loop) rather than inventing new idioms.
"""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
  "eval_impulse_script", _REPO / "scripts" / "eval_impulse.py"
)
eval_impulse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_impulse)

_TASK = eval_impulse.QUALITY_ARM_TASKS["F8"]
_NAIL_GEOMETRY = {
  "nail_axis": [0.0, 0.0, -1.0],
  "nail_xy_m": [0.5, 0.0],
  "nail_radius_m": 0.012,
  "source_sha256": "0" * 64,
}
_NUM_STEPS = 6


def _short_stochastic_trace(*, reset_seed: int) -> dict:
  """One completed episode from a real (noisy) reset, timed out deliberately
  short so the test stays fast -- completion (not success) is all D2 needs."""
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(_TASK, play=False)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = float(
    _NUM_STEPS * cfg.sim.mujoco.timestep * cfg.decimation
  )
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=eval_impulse.QUALITY_TASK_TO_ARM[_TASK],
      task=_TASK,
      gamma=0.99,
      event_i_ref_n_s=0.3088,
      nail_geometry=_NAIL_GEOMETRY,
      reset_seed=reset_seed,
    )
    env.reset()
    action = torch.zeros(1, 3)
    for _ in range(_NUM_STEPS + 2):
      env.step(action)
      if collector.completed:
        break
    assert len(collector.completed) == 1, "fixture episode never completed"
    return copy.deepcopy(collector.completed[0])
  finally:
    env.close()


@pytest.fixture(scope="module")
def captured_trace() -> dict:
  return _short_stochastic_trace(reset_seed=4242)


# ---------------------------------------------------------------------------
# Acceptance test: restore reset_state + replay action_tape -> identical
# physical trace digest.
# ---------------------------------------------------------------------------


def test_restoring_banked_reset_state_and_replaying_action_tape_reproduces_trace_digest(
  captured_trace,
):
  trace_a = captured_trace
  assert trace_a["reset_contract_version"] == eval_impulse.RESET_CONTRACT_VERSION
  assert trace_a["reset_state_digest"]

  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(_TASK, play=False)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = float(
    _NUM_STEPS * cfg.sim.mujoco.timestep * cfg.decimation
  )
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()  # deterministically initializes every manager; qpos/qvel overwritten next.
    eval_impulse.restore_reset_state(env, trace_a)

    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=eval_impulse.QUALITY_TASK_TO_ARM[_TASK],
      task=_TASK,
      gamma=0.99,
      event_i_ref_n_s=0.3088,
      nail_geometry=_NAIL_GEOMETRY,
      reset_seed=trace_a["reset_state"]["regeneration_inputs"]["reset_rng_seed"],
      initial_reset_states={0: trace_a["reset_state"]},
    )
    for row in trace_a["action_tape"]:
      env.step(torch.tensor([row], dtype=torch.float32))
      if collector.completed:
        break
    assert len(collector.completed) == 1, "replay episode never completed"
    trace_b = collector.completed[0]
  finally:
    env.close()

  assert trace_b["reset_state_digest"] == trace_a["reset_state_digest"]
  assert trace_b["action_tape"] == trace_a["action_tape"]
  assert trace_b["trace_digest"] == trace_a["trace_digest"]


# ---------------------------------------------------------------------------
# Fail-closed tests: each corruption of the banked reset state must RAISE,
# never warn-and-continue. Pure-dict mutations -- no env needed.
# ---------------------------------------------------------------------------


def test_missing_reset_state_raises(captured_trace):
  trace = copy.deepcopy(captured_trace)
  del trace["reset_state"]
  with pytest.raises(ValueError, match="reset_state"):
    eval_impulse._validated_reset_state_digest(trace, require_recorded_digest=True)


def test_malformed_short_reset_state_raises(captured_trace):
  trace = copy.deepcopy(captured_trace)
  trace["reset_state"]["realized"]["robot_joint_pos"] = trace["reset_state"][
    "realized"
  ]["robot_joint_pos"][:-1]
  with pytest.raises(ValueError, match="length mismatch"):
    eval_impulse._validated_reset_state_digest(trace, require_recorded_digest=True)


def test_nonfinite_reset_state_value_raises(captured_trace):
  trace = copy.deepcopy(captured_trace)
  trace["reset_state"]["realized"]["robot_joint_pos"][0] = float("nan")
  with pytest.raises(ValueError, match="nonfinite"):
    eval_impulse._validated_reset_state_digest(trace, require_recorded_digest=True)


def test_reset_state_digest_mismatch_raises(captured_trace):
  trace = copy.deepcopy(captured_trace)
  trace["reset_state_digest"] = "0" * 64
  with pytest.raises(ValueError, match="mismatch"):
    eval_impulse._validated_reset_state_digest(trace, require_recorded_digest=True)


def test_missing_reset_contract_version_raises(captured_trace):
  trace = copy.deepcopy(captured_trace)
  del trace["reset_contract_version"]
  with pytest.raises(ValueError, match="reset_contract_version"):
    eval_impulse._validated_reset_state_digest(trace, require_recorded_digest=True)


def test_restore_reset_state_fails_closed_on_tampered_digest(captured_trace):
  """The same fail-closed guarantee, exercised through the actual replay
  entry point (not just the raw validator) -- and BEFORE any sim write."""
  trace = copy.deepcopy(captured_trace)
  trace["reset_state"]["realized"]["nail_joint_vel"] = [math.inf]

  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(_TASK, play=False)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    with pytest.raises(ValueError, match="nonfinite"):
      eval_impulse.restore_reset_state(env, trace)
  finally:
    env.close()

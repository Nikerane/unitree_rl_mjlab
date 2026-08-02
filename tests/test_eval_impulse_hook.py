"""Adversarial-review I12 (2026-07-14): positive control for ``scripts/eval_impulse.py``'s
pre-reset snapshot hook — the C3-CSV correctness core.

The bug class the hook exists for: with mjlab's default ``auto_reset=True``, an env that
terminates inside ``env.step()`` is reset IN-STEP (``_reset_idx`` runs after
``metrics_manager.compute()`` but before ``env.step()`` returns), zeroing the shipped
accumulators' buffers. Reading Λ_j / delivered / depth after ``env.step()`` returns is therefore
too late for exactly the episodes the eval cares about (the completed ones). The hook patches
``metrics_manager.compute`` to snapshot the live buffers every control step, so the snapshot
taken on the terminal step still holds the pre-reset values.

This test drives the open-loop reference strike to success in an ``auto_reset=True`` env (the
same configuration ``eval_impulse.py`` runs under) and asserts BOTH halves of the control:
the snapshot holds nonzero strike measurements, AND the live buffers are already zeroed when
``env.step()`` returns — i.e. the hook is load-bearing, not decorative. Task 11 ran this
control interactively; it was never committed (the I12 finding).
"""

from __future__ import annotations

import importlib.util
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
  HAMMER_HEAD_SITE_NAME,
  Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[1]

# scripts/ is not a package — load the script as a module the same way its .sh runner shells it.
_spec = importlib.util.spec_from_file_location(
  "eval_impulse_script", _REPO / "scripts" / "eval_impulse.py"
)
eval_impulse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_impulse)

HOLD_STEPS = 6  # post-playback settle steps, mirrors derive_impulse_thresholds.py


def test_fq3x8_rng_contract_accepts_the_preregistered_streams():
  """The three frozen streams are accepted for every fq3x8 treatment arm."""
  for task in (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
  ):
    eval_impulse._validate_evaluation_campaign(
      "fq3x8",
      task=task,
      reset_seed=2036073019,
      observation_seed=2046073033,
      action_seed=2056073041,
    )


@pytest.mark.parametrize(
  "stream, reset_seed, observation_seed, action_seed",
  (
    ("reset", 2036073020, 2046073033, 2056073041),
    ("observation", 2036073019, 2046073034, 2056073041),
    ("action", 2036073019, 2046073033, 2056073042),
  ),
)
def test_fq3x8_rng_contract_rejects_each_single_stream_drift(
  stream, reset_seed, observation_seed, action_seed
):
  """A one-field RNG mutation must fail before any fq3x8 rollout begins."""
  with pytest.raises(ValueError, match=f"fq3x8.*{stream}"):
    eval_impulse._validate_evaluation_campaign(
      "fq3x8",
      task="Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
      reset_seed=reset_seed,
      observation_seed=observation_seed,
      action_seed=action_seed,
    )


def test_fq3x8_rng_contract_rejects_non_preregistered_arm():
  """The campaign selector cannot silently evaluate a historical F0 arm."""
  with pytest.raises(ValueError, match="fq3x8.*task"):
    eval_impulse._validate_evaluation_campaign(
      "fq3x8",
      task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
      reset_seed=2036073019,
      observation_seed=2046073033,
      action_seed=2056073041,
    )


def test_unscoped_evaluation_preserves_historical_rng_flexibility():
  """No campaign selector leaves historical task and stream behavior unchanged."""
  eval_impulse._validate_evaluation_campaign(
    None,
    task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    reset_seed=11,
    observation_seed=22,
    action_seed=33,
  )


def test_fq3x8_main_rejects_rng_drift_before_checkpoint_or_rollout(monkeypatch):
  """The CLI contract fires before the evaluator can load a checkpoint."""
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "eval_impulse.py",
      "--campaign", "fq3x8",
      "--task", "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
      "--ckpt", "checkpoint-not-opened.pt",
      "--training-seed", "16",
      "--reset-seed", "2036073020",
      "--observation-seed", "2046073033",
      "--action-seed", "2056073041",
    ],
  )

  with pytest.raises(ValueError, match="fq3x8.*reset"):
    eval_impulse.main()


def _fixed_reference_action_tape() -> list[torch.Tensor]:
  task = eval_impulse.QUALITY_ARM_TASKS["F8"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    robot = env.scene["robot"]
    nail = env.scene["nail_block"]
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_cfg.resolve(env.scene)
    head = lambda: robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)
    nail_top = lambda: nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
    env.reset()
    reference = SingleStrikeReference(1, env.device)
    reference.update(head(), nail_top(), torch.zeros(1, dtype=torch.long))
    tape = []
    for step in range(1, reference.playback_length() + HOLD_STEPS + 1):
      target = reference.playback_target(min(step, reference.playback_length()))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      tape.append(action.detach().clone())
      env.step(action)
      if bool(env.reset_terminated.any()):
        break
    assert bool(env.reset_terminated.any()), "reference tape did not reach contact/success"
    return tape
  finally:
    env.close()


def _single_control_trace(
  task: str, *, strict_quality: bool, action_tape: list[torch.Tensor]
) -> dict:
  if strict_quality:
    _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=True)
  else:
    cfg = eval_impulse.load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = float(
    len(action_tape) * cfg.sim.mujoco.timestep * cfg.decimation
  )
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.0
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=(
        eval_impulse.QUALITY_TASK_TO_ARM[task] if strict_quality else "F8"
      ),
      task=task,
      gamma=0.99,
      event_i_ref_n_s=0.3088,
      nail_geometry={
        "nail_axis": [0.0, 0.0, -1.0],
        "nail_xy_m": [0.5, 0.0],
        "nail_radius_m": 0.012,
        "source_sha256": "0" * 64,
      },
    )
    env.reset()
    for action in action_tape:
      env.step(action)
      if collector.completed:
        break
    assert len(collector.completed) == 1
    return copy.deepcopy(collector.completed[0])
  finally:
    env.close()


def _first_payload_difference(left, right, path="payload"):
  if type(left) is not type(right):
    return path, type(left).__name__, type(right).__name__
  if isinstance(left, dict):
    for key in sorted(set(left) | set(right)):
      if key not in left or key not in right:
        return f"{path}.{key}", left.get(key, "<missing>"), right.get(key, "<missing>")
      difference = _first_payload_difference(left[key], right[key], f"{path}.{key}")
      if difference is not None:
        return difference
    return None
  if isinstance(left, list):
    if len(left) != len(right):
      return f"{path}.length", len(left), len(right)
    for index, (left_item, right_item) in enumerate(zip(left, right)):
      difference = _first_payload_difference(left_item, right_item, f"{path}[{index}]")
      if difference is not None:
        return difference
    return None
  return None if left == right else (path, left, right)


def test_episode_hook_snapshots_pre_reset_buffers_on_the_terminal_step():
  cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
  cfg.scene.num_envs = 1
  # auto_reset stays at its default (True) — the exact configuration the hook exists for.
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    snap = eval_impulse._install_episode_hook(env)

    robot = env.scene["robot"]
    nail_e = env.scene["nail_block"]
    head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    head_cfg.resolve(env.scene)
    nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    nail_cfg.resolve(env.scene)

    def head() -> torch.Tensor:
      return robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)

    def nail_top() -> torch.Tensor:
      return nail_e.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    env.reset()
    ref = SingleStrikeReference(1, env.device)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()

    struck = False
    for k in range(1, n + HOLD_STEPS + 1):
      target = ref.playback_target(min(k, n))
      action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
      env.step(action)
      if bool(env.reset_terminated.any()):
        struck = True
        break
    assert struck, "reference strike never triggered the success termination — cannot exercise the hook"

    # The terminal step just ran: compute() (hook fires, snapshot taken) -> _reset_idx (buffers
    # zeroed) -> env.step() returned. Positive control, both halves:
    # (a) the snapshot holds the pre-reset strike measurements...
    assert snap["perjoint"] is not None
    assert float(snap["perjoint"].max()) > 0.0, "hook snapshot lost the per-joint Λ episode peaks"
    assert float(snap["delivered"][0]) > 0.0, "hook snapshot lost the delivered impulse"
    assert float(snap["depth"][0]) > 0.02, (
      f"hook snapshot depth {float(snap['depth'][0]):.4f} m below the success regime — "
      "the strike terminated the episode, so the pre-reset depth must show it"
    )
    # (b) ...while the live buffers are ALREADY ZEROED — reading them post-step (the naive
    # pattern the hook replaces) would report an empty episode. If this half ever fails, the
    # reset path changed and the hook may no longer be needed; if (a) fails, the eval CSV is
    # silently recording zeros for every completed episode.
    acc = getattr(env, eval_impulse._ENV_SUBSTEP_IMPULSE_ATTR)
    dacc = getattr(env, eval_impulse._ENV_SUBSTEP_DELIVERED_ATTR)
    assert float(acc._episode_peak_perjoint.max()) == 0.0, (
      "live Λ buffer survived the in-step auto-reset — the hook's premise changed"
    )
    assert float(dacc.delivered[0]) == 0.0, (
      "live delivered buffer survived the in-step auto-reset — the hook's premise changed"
    )
  finally:
    env.close()


def test_install_episode_hook_requires_both_accumulators():
  # The guard path: a non-impulse task (no stashed accumulators) must be rejected loudly at
  # install time, not fail with silent Nones at CSV-write time.
  class _FakeEnv:
    pass

  with pytest.raises(RuntimeError, match="requires BOTH"):
    eval_impulse._install_episode_hook(_FakeEnv())


# (arm, resolved treatment, expected (impact, delivered) weights).
# The resolved treatment is NOT always the arm label -- see the docstring below.
_QUALITY_CONTRACT_WEIGHTS = (
  ("F8", "F", (8.0, 2.0)),
  ("F0", "F0", (0.0, 2.0)),
  ("D0", "D0", (8.0, 0.0)),
  ("FQ", "FQ", (8.0, 0.0)),
  ("B8", "B8", (8.0, 0.0)),
)


@pytest.mark.parametrize(
  ("arm", "expected_treatment", "expected_weights"), _QUALITY_CONTRACT_WEIGHTS
)
def test_sampled_env_contract_weights_match_the_four_arm_table(
  arm, expected_treatment, expected_weights
):
  """The strict evaluator must accept each strict-quality arm's shipped weights.

  ``_validate_sampled_env_contract`` resolves the treatment as
  ``TASK_TO_ARM.get(task, QUALITY_TASK_TO_ARM.get(task))`` -- the LEGACY map is
  consulted first and wins. The F8 arm's task string is byte-identical to the
  legacy "F" task, so the legacy map never misses for it and F8 always resolves
  to treatment "F" (never "F8"); its weight contract is 8/2 under either label.
  F0/D0/FQ miss the legacy map and resolve through ``QUALITY_TASK_TO_ARM``.
  FQ is the bug this test pins: it used to fall through to the (8,2) default
  even though its shipped weights are (8,0).
  """
  task = eval_impulse.QUALITY_ARM_TASKS[arm]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  contract = eval_impulse._validate_sampled_env_contract(cfg, task)
  assert contract["treatment"] == expected_treatment
  assert (contract["impact_weight"], contract["delivered_weight"]) == expected_weights


def test_sampled_env_contract_rejects_fq_with_legacy_delivered_weight():
  """Mutation test: if FQ's delivered weight ever regressed to the legacy 2.0,
  the strict evaluator must reject it, not silently accept -- this is the
  fail-closed half of the fix (no default treatment gets a free pass)."""
  task = eval_impulse.QUALITY_ARM_TASKS["FQ"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  cfg.rewards["delivered_impulse"].weight = 2.0
  with pytest.raises(ValueError, match="configured maximize weights"):
    eval_impulse._validate_sampled_env_contract(cfg, task)


def test_b8_evaluator_contract_pins_bounded_reader_normalizer_and_identity_digest():
  """Changing B8's reader or full normalizer must alter its frozen treatment identity."""
  task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded"
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)

  contract = eval_impulse._validate_sampled_env_contract(cfg, task)
  baseline_digest = eval_impulse._treatment_config_digest(task=task, contract=contract)

  assert contract["treatment"] == "B8"
  assert (contract["impact_weight"], contract["delivered_weight"]) == (8.0, 0.0)
  assert contract["impact_reader"] == "FirstStrikeBoundedImpactRewardTerm"
  assert contract["impact_v_expected_n_s"] == pytest.approx(1.4598331451416016)
  assert contract["delivered_saturate"] is False

  changed_contract = dict(contract)
  changed_contract["impact_v_expected_n_s"] = 1.0
  changed_digest = eval_impulse._treatment_config_digest(
    task=task, contract=changed_contract
  )
  assert changed_digest != baseline_digest


@pytest.mark.parametrize(
  ("arm", "impact_reader"),
  (
    ("FQ", "FirstStrikeQualityImpactRewardTerm"),
    ("B8", "FirstStrikeBoundedImpactRewardTerm"),
  ),
)
def test_b8_and_fq_treatment_identities_pin_reader_normalizer_and_delivered_semantics(
  arm, impact_reader
):
  """The same 8/0 weights must not make the B8/FQ readers interchangeable."""
  task = eval_impulse.QUALITY_ARM_TASKS[arm]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  contract = eval_impulse._validate_sampled_env_contract(cfg, task)
  baseline_digest = eval_impulse._treatment_config_digest(task=task, contract=contract)

  assert contract["impact_reader"] == impact_reader
  assert contract["impact_v_expected_n_s"] == pytest.approx(1.4598331451416016)
  assert contract["delivered_reader"] == "FirstStrikeDeliveredRewardTerm"
  assert contract["delivered_saturate"] is False

  changed = dict(contract)
  changed["impact_reader"] = "wrong-reader"
  assert eval_impulse._treatment_config_digest(task=task, contract=changed) != baseline_digest
  changed = dict(contract)
  changed["impact_v_expected_n_s"] = 1.0
  assert eval_impulse._treatment_config_digest(task=task, contract=changed) != baseline_digest
  changed = dict(contract)
  changed["delivered_saturate"] = True
  assert eval_impulse._treatment_config_digest(task=task, contract=changed) != baseline_digest


@pytest.mark.parametrize(
  ("mutate", "message"),
  (
    (
      lambda cfg: cfg.rewards["impact_progress"].__setattr__("func", object),
      "impact reader",
    ),
    (
      lambda cfg: cfg.rewards["impact_progress"].params.__setitem__("v_expected", 1.0),
      "v_expected",
    ),
    (
      lambda cfg: cfg.rewards["delivered_impulse"].params.__setitem__("saturate", True),
      "delivered saturate",
    ),
  ),
)
def test_fq_evaluator_rejects_reader_normalizer_or_delivered_semantic_drift(mutate, message):
  """FQ's strict treatment identity must fail before rollout on semantic drift."""
  task = eval_impulse.QUALITY_ARM_TASKS["FQ"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  mutate(cfg)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_sampled_env_contract(cfg, task)


@pytest.mark.parametrize(
  ("mutate", "message"),
  (
    (
      lambda cfg: cfg.rewards["impact_progress"].__setattr__("func", object),
      "impact reader",
    ),
    (
      lambda cfg: cfg.rewards["impact_progress"].params.__setitem__("v_expected", 1.0),
      "v_expected",
    ),
    (
      lambda cfg: cfg.rewards["delivered_impulse"].__setattr__("weight", 2.0),
      "configured maximize weights",
    ),
    (
      lambda cfg: cfg.rewards["delivered_impulse"].params.__setitem__("saturate", True),
      "delivered saturate",
    ),
    (
      lambda cfg: cfg.actions["ik_hammer_head"].__setattr__("max_dq", 0.29),
      "fixed action signature",
    ),
    (
      lambda cfg: cfg.scene.entities["robot"].articulation.actuators[0].__setattr__("stiffness", 1.0),
      "fixed-impedance actuator signature",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_limit", [1.0] * 6),
      "impulse limits",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_max_p", 0.5),
      "imp_max_p",
    ),
  ),
)
def test_b8_evaluator_rejects_identity_or_fixed_plant_mutations(mutate, message):
  """Any B8 reader, payout, action/gain, or CaT-cap drift must fail before rollout."""
  task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded"
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  mutate(cfg)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_sampled_env_contract(cfg, task)


def test_b8_evaluator_rejects_coordinated_imported_and_live_cap_drift(monkeypatch):
  """A shared mutable cap constant must not redefine B8's frozen safety rails."""
  task = eval_impulse.QUALITY_ARM_TASKS["B8"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  drifted = (1.0,) * 6
  cfg.metrics["cat_soft"].params["imp_limit"] = list(drifted)
  monkeypatch.setattr(eval_impulse, "IMP_J_LIMIT", drifted)

  with pytest.raises(ValueError, match="frozen impulse limits"):
    eval_impulse._validate_sampled_env_contract(cfg, task)


@pytest.mark.parametrize("arm", ("FQ", "B8"))
@pytest.mark.parametrize("mutation", ("missing_sensor", "missing_tracker"))
def test_strict_native_quality_arms_reject_missing_native_quality_path(
  monkeypatch, arm, mutation
):
  """B8/FQ must not be repaired from a reference config at evaluation time."""
  task = eval_impulse.QUALITY_ARM_TASKS[arm]
  source_load = eval_impulse.load_env_cfg
  cfg = source_load(task, play=False)
  if mutation == "missing_sensor":
    cfg.scene.sensors = tuple(
      sensor for sensor in cfg.scene.sensors
      if sensor.name != "hammer_nail_quality"
    )
  else:
    del cfg.metrics["first_strike"]
  monkeypatch.setattr(
    eval_impulse,
    "load_env_cfg",
    lambda requested, play=False: cfg if requested == task else source_load(requested, play=play),
  )

  with pytest.raises(ValueError, match="native quality"):
    eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)


@pytest.mark.parametrize("arm", ("FQ", "B8"))
def test_strict_native_quality_arms_reject_delivered_reader_swap(arm):
  """An equal-weight raw delivered reader must not alias the strict quality arms."""
  task = eval_impulse.QUALITY_ARM_TASKS[arm]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  cfg.rewards["delivered_impulse"].func = object

  with pytest.raises(ValueError, match="delivered reader"):
    eval_impulse._validate_sampled_env_contract(cfg, task)


GUIDELINE_TASKS = {
  "C0": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
  "C-Gate": "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
}


@pytest.mark.parametrize(("arm", "task"), tuple(GUIDELINE_TASKS.items()))
def test_native_guideline_contract_accepts_only_the_two_registered_training_configs(
  arm, task
):
  cfg = eval_impulse.load_env_cfg(task, play=False)

  contract = eval_impulse._validate_native_guideline_env_contract(cfg, task)

  assert eval_impulse.GUIDELINE_ARM_TASKS == GUIDELINE_TASKS
  assert contract["treatment"] == arm
  assert (contract["impact_weight"], contract["delivered_weight"]) == (8.0, 2.0)
  assert contract["event_i_ref_n_s"] == pytest.approx(0.3088)
  assert contract["impact_reader"] == "FirstStrikeImpactRewardTerm"
  assert contract["delivered_reader"] == "FirstStrikeDeliveredRewardTerm"
  assert contract["delivered_saturate"] is False
  assert contract["reset_position_noise_min_rad"] == 0.0
  assert contract["reset_position_noise_max_rad"] == 0.0
  assert contract["guideline_num_gates"] == 6


def test_native_guideline_mapping_does_not_expand_strict_checkpoint_evaluator_tasks():
  guideline_tasks = set(eval_impulse.GUIDELINE_ARM_TASKS.values())
  advertised_checkpoint_tasks = set(
    eval_impulse.TASK_TO_ARM | eval_impulse.QUALITY_TASK_TO_ARM
  )

  assert guideline_tasks.isdisjoint(advertised_checkpoint_tasks)
  assert advertised_checkpoint_tasks == {
    "Unitree-Z1-Hammer-CaT-Impulse",
    "Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear",
    "Unitree-Z1-Hammer-CaT-Impulse-Event",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Quality",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Bounded",
  }
  near_match = GUIDELINE_TASKS["C-Gate"] + "-near-match"
  cfg = eval_impulse.load_env_cfg(GUIDELINE_TASKS["C-Gate"], play=False)
  with pytest.raises(ValueError, match="native guideline"):
    eval_impulse._validate_native_guideline_env_contract(cfg, near_match)


@pytest.mark.parametrize(
  ("mutate", "message"),
  (
    (
      lambda cfg: cfg.events["reset_robot_joints"].params.__setitem__(
        "position_range", (-0.05, 0.05)
      ),
      "fixed reset",
    ),
    (
      lambda cfg: cfg.rewards["impact_progress"].__setattr__("weight", 7.0),
      "maximize weights",
    ),
    (
      lambda cfg: cfg.rewards["impact_progress"].__setattr__("func", object),
      "impact reader",
    ),
    (
      lambda cfg: cfg.rewards["delivered_impulse"].params.__setitem__(
        "i_ref", 1.0
      ),
      "i_ref",
    ),
    (
      lambda cfg: cfg.rewards["delivered_impulse"].params.__setitem__(
        "saturate", True
      ),
      "saturate",
    ),
    (
      lambda cfg: cfg.actions["ik_hammer_head"].__setattr__("max_dq", 0.29),
      "fixed action signature",
    ),
    (
      lambda cfg: cfg.scene.entities["robot"].articulation.actuators[0].__setattr__(
        "stiffness", 1.0
      ),
      "fixed-impedance actuator signature",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_max_p", 0.5),
      "imp_max_p",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__(
        "imp_limit", [1.0] * 6
      ),
      "impulse limits",
    ),
    (
      lambda cfg: cfg.sim.mujoco.__setattr__("timestep", 0.004),
      "physics timestep",
    ),
    (lambda cfg: cfg.__setattr__("decimation", 5), "control decimation"),
    (
      lambda cfg: cfg.rewards.__setitem__(
        "r_imit", copy.deepcopy(cfg.rewards["approach"])
      ),
      "r_imit",
    ),
  ),
)
def test_native_guideline_contract_rejects_reward_reset_or_fixed_plant_drift(
  mutate, message
):
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  mutate(cfg)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_native_guideline_env_contract(cfg, task)


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("missing", "WaypointProgressTracker"),
    ("wrong_func", "WaypointProgressTracker"),
    ("not_substep", "WaypointProgressTracker"),
    ("wrong_robot", "robot site binding"),
    ("wrong_nail", "nail site binding"),
  ),
)
def test_native_guideline_contract_rejects_tracker_drift(mutation, message):
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  if mutation == "missing":
    del cfg.metrics["waypoint_progress"]
  elif mutation == "wrong_func":
    cfg.metrics["waypoint_progress"].func = object
  elif mutation == "not_substep":
    cfg.metrics["waypoint_progress"].per_substep = False
  elif mutation == "wrong_robot":
    cfg.metrics["waypoint_progress"].params["robot_cfg"].site_names = ("wrong",)
  else:
    cfg.metrics["waypoint_progress"].params["nail_cfg"].site_names = ("wrong",)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_native_guideline_env_contract(cfg, task)


@pytest.mark.parametrize("group", ("actor", "critic"))
@pytest.mark.parametrize(
  "term_name",
  (
    "next_gate_vector",
    "completed_gate_fraction",
    "guideline_perpendicular_error",
  ),
)
def test_native_guideline_contract_requires_each_guideline_observation_in_both_groups(
  group, term_name
):
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  del cfg.observations[group].terms[term_name]

  with pytest.raises(ValueError, match=term_name):
    eval_impulse._validate_native_guideline_env_contract(cfg, task)


@pytest.mark.parametrize(
  ("term_name", "field", "value"),
  (
    ("next_gate_vector", "reader", object),
    ("completed_gate_fraction", "width", 2),
    ("guideline_perpendicular_error", "reader", object),
  ),
)
def test_native_guideline_contract_rejects_observation_reader_or_width_drift(
  term_name, field, value
):
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.observations["actor"].terms[term_name].params[field] = value

  with pytest.raises(ValueError, match=term_name):
    eval_impulse._validate_native_guideline_env_contract(cfg, task)


def test_native_guideline_contract_rejects_c0_r_gate_key_with_none_value():
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.rewards["r_gate"] = None

  with pytest.raises(ValueError, match="C0.*r_gate"):
    eval_impulse._validate_native_guideline_env_contract(cfg, task)


def test_native_guideline_contract_rejects_c0_gate_reward_or_cgate_gate_reward_drift():
  c0_task = GUIDELINE_TASKS["C0"]
  cgate_task = GUIDELINE_TASKS["C-Gate"]
  c0 = eval_impulse.load_env_cfg(c0_task, play=False)
  cgate = eval_impulse.load_env_cfg(cgate_task, play=False)

  c0.rewards["r_gate"] = copy.deepcopy(cgate.rewards["r_gate"])
  with pytest.raises(ValueError, match="C0.*r_gate"):
    eval_impulse._validate_native_guideline_env_contract(c0, c0_task)

  mutations = (
    lambda cfg: cfg.rewards.pop("r_gate"),
    lambda cfg: cfg.rewards["r_gate"].__setattr__("func", object),
    lambda cfg: cfg.rewards["r_gate"].__setattr__("weight", 7.0),
    lambda cfg: cfg.rewards["r_gate"].params.__setitem__("extra", 1),
  )
  for mutate in mutations:
    cfg = eval_impulse.load_env_cfg(cgate_task, play=False)
    mutate(cfg)
    with pytest.raises(ValueError, match="C-Gate.*r_gate"):
      eval_impulse._validate_native_guideline_env_contract(cfg, cgate_task)


@pytest.mark.integration
def test_fixed_action_tape_preserves_physics_across_strict_quality_arms():
  """Passive quality sensing changes no plant channel; payouts are intentionally absent."""
  action_tape = _fixed_reference_action_tape()
  traces = {
    arm: _single_control_trace(task, strict_quality=True, action_tape=action_tape)
    for arm, task in eval_impulse.QUALITY_ARM_TASKS.items()
  }
  assert eval_impulse.compare_action_tape_physics(traces)["arms"] == (
    "B8", "D0", "F0", "F8", "FQ",
  )
  assert len({json.dumps(trace["action_tape"]) for trace in traces.values()}) == 1
  assert traces["F8"]["first_strike"]["started"] is True
  assert max(traces["F8"]["episode_peak_lambda"]) > 0.0

  uninstrumented_f8 = _single_control_trace(
    eval_impulse.QUALITY_ARM_TASKS["F8"],
    strict_quality=False,
    action_tape=action_tape,
  )
  instrumented_payload = eval_impulse._plant_replay_payload(traces["F8"])
  uninstrumented_payload = eval_impulse._plant_replay_payload(uninstrumented_f8)
  assert _first_payload_difference(
    instrumented_payload, uninstrumented_payload
  ) is None

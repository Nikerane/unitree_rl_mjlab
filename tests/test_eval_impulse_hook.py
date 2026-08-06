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
import csv
import json
import math
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
from evaluation.analysis import guideline_campaign as guideline_analysis

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


PRESENTATION3_CAMPAIGN = "presentation3"
PRESENTATION3_RNG = {
  "reset_seed": 2036072919,
  "observation_seed": 2046072933,
  "action_seed": 2056072941,
}


@pytest.mark.parametrize(
  "task",
  (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
  ),
)
def test_presentation3_campaign_accepts_only_its_three_tasks_with_frozen_rng(task):
  """Removing the campaign arm allowlist or frozen tuple must fail this call."""
  eval_impulse._validate_evaluation_campaign(
    PRESENTATION3_CAMPAIGN,
    task=task,
    **PRESENTATION3_RNG,
  )


def test_presentation3_campaign_rejects_unscoped_execution():
  """A P3 task may never fall through the historical unscoped evaluator."""
  with pytest.raises(ValueError, match="presentation3.*requires --campaign"):
    eval_impulse._validate_evaluation_campaign(
      None,
      task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
      reset_seed=11,
      observation_seed=22,
      action_seed=33,
    )


@pytest.mark.parametrize("stream", tuple(PRESENTATION3_RNG))
def test_presentation3_campaign_rejects_each_rng_stream_mutation(stream):
  """Each RNG stream is independently load-bearing for the shared population."""
  streams = dict(PRESENTATION3_RNG)
  streams[stream] += 1
  with pytest.raises(ValueError, match=f"presentation3.*{stream.removesuffix('_seed')}"):
    eval_impulse._validate_evaluation_campaign(
      PRESENTATION3_CAMPAIGN,
      task="Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
      **streams,
    )


def test_presentation3_campaign_rejects_a_nonpresentation_task():
  with pytest.raises(ValueError, match="presentation3.*task"):
    eval_impulse._validate_evaluation_campaign(
      PRESENTATION3_CAMPAIGN,
      task=eval_impulse.QUALITY_ARM_TASKS["FQ"],
      **PRESENTATION3_RNG,
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


# These are the three already-trained Presentation3 identities.  The evaluator
# must bind their task semantics before a stochastic sampled rollout, rather
# than treating the task ID as an interchangeable label.
_PRESENTATION3_CONTRACTS = (
  (
    "M",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4",
    4.0,
    False,
  ),
  (
    "V",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
    2.0,
    True,
  ),
  (
    "V+M",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
    4.0,
    True,
  ),
)


def _presentation3_identity_kwargs(arm="V", **overrides):
  task = next(task for name, task, _, _ in _PRESENTATION3_CONTRACTS if name == arm)
  asset_revision = "b" * 40
  values = {
    "campaign": PRESENTATION3_CAMPAIGN,
    "env_cfg": eval_impulse.load_env_cfg(task, play=False),
    "task": task,
    "training_seed": 2,
    "expected_checkpoint_sha256": "c" * 64,
    "accepted_manifest_sha256": "d" * 64,
    # Deliberately differs from clean evaluation code revision below.
    "training_code_revision": "e" * 40,
    "training_asset_revision": asset_revision,
    "code_git": {"revision": "a" * 40, "dirty": False, "status": ""},
    "asset_git": {
      "revision": asset_revision,
      "dirty": False,
      "status": "",
    },
  }
  values.update(overrides)
  return values


@pytest.mark.parametrize("arm", ("M", "V", "V+M"))
def test_presentation3_row_identity_accepts_each_arm_and_distinct_evaluation_revision(arm):
  """Requiring eval/training code equality would wrongly reject reviewed evaluators."""
  validator = getattr(eval_impulse, "_validate_presentation3_identity", None)
  assert callable(validator), "Presentation3 row identity validator is missing"

  contract = validator(**_presentation3_identity_kwargs(arm))

  assert contract["treatment"] == arm
  assert contract["training_seed"] == 2
  assert contract["expected_checkpoint_sha256"] == "c" * 64
  assert contract["accepted_manifest_sha256"] == "d" * 64


@pytest.mark.parametrize(
  ("overrides", "message"),
  (
    ({"campaign": None}, "exact campaign"),
    ({"training_seed": 1}, "seeds 2..7"),
    ({"training_seed": 8}, "seeds 2..7"),
    ({"training_seed": True}, "seeds 2..7"),
    ({"expected_checkpoint_sha256": ""}, "checkpoint SHA-256"),
    ({"expected_checkpoint_sha256": "z" * 64}, "checkpoint SHA-256"),
    ({"accepted_manifest_sha256": ""}, "manifest SHA-256"),
    ({"accepted_manifest_sha256": "z" * 64}, "manifest SHA-256"),
    ({"training_code_revision": ""}, "training code revision"),
    ({"training_code_revision": "z" * 40}, "training code revision"),
    ({"training_asset_revision": ""}, "training asset revision"),
    ({"training_asset_revision": "z" * 40}, "training asset revision"),
    (
      {"code_git": {"revision": "a" * 40, "dirty": True, "status": " M file"}},
      "clean code provenance",
    ),
    (
      {"asset_git": {"revision": "b" * 40, "dirty": True, "status": " M asset"}},
      "clean asset provenance",
    ),
    ({"training_asset_revision": "f" * 40}, "asset revision mismatch"),
  ),
)
def test_presentation3_row_identity_rejects_unfrozen_or_dirty_inputs(
  overrides, message
):
  validator = getattr(eval_impulse, "_validate_presentation3_identity", None)
  assert callable(validator), "Presentation3 row identity validator is missing"
  with pytest.raises((ValueError, RuntimeError), match=message):
    validator(**_presentation3_identity_kwargs(**overrides))


@pytest.mark.parametrize(
  ("arm", "task", "delivered_weight", "velocity_cat"),
  _PRESENTATION3_CONTRACTS,
)
def test_sampled_contract_binds_each_presentation3_task_identity(
  arm, task, delivered_weight, velocity_cat
):
  """Wrong task mapping, dose, CaT setting, caps, or observations must fail.

  This catches a future change that admits a Presentation3 checkpoint under a
  different registered task contract.  All expected values are hand-frozen
  from the preregistered 2x2, not derived by the evaluator.
  """
  cfg = eval_impulse.load_env_cfg(task, play=False)

  contract = eval_impulse._validate_sampled_env_contract(cfg, task)

  assert contract["treatment"] == arm
  assert contract["guidance_weight"] == 8.0
  assert contract["delivered_weight"] == delivered_weight
  assert contract["velocity_cat_enabled"] is velocity_cat
  assert contract["velocity_detection"] == (
    "substep" if velocity_cat else "disabled"
  )
  assert contract["imp_max_p"] == 0.0
  assert contract["impulse_limits_n_m_s"] == [1.64, 3.28, 1.64, 1.64, 1.64, 1.64]
  assert contract["reset_position_noise_min_rad"] == 0.0
  assert contract["reset_position_noise_max_rad"] == 0.0
  assert contract["guideline_observation_width"] == 7


@pytest.mark.parametrize(
  ("task", "mutate", "message"),
  (
    (
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4",
      lambda cfg: cfg.rewards["delivered_impulse"].__setattr__("weight", 2.0),
      "configured maximize weights",
    ),
    (
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel",
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__(
        "vel_detection", "control_rate"
      ),
      "velocity-CaT",
    ),
    (
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__(
        "imp_limit", [1.0] * 6
      ),
      "impulse limits",
    ),
    (
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Delivered4",
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("use_impulse", False),
      "impulse-CaT",
    ),
    (
      "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel-Delivered4",
      lambda cfg: cfg.observations["actor"].terms.pop("waypoint_progress_state"),
      "guideline observation",
    ),
  ),
)
def test_sampled_contract_rejects_presentation3_identity_mutations(
  task, mutate, message
):
  """Mutation probes: every frozen Presentation3 axis is fail-closed."""
  cfg = eval_impulse.load_env_cfg(task, play=False)
  mutate(cfg)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_sampled_env_contract(cfg, task)


@pytest.mark.parametrize(
  ("mutate", "message"),
  (
    (
      lambda cfg: cfg.events["reset_robot_joints"].params.__setitem__(
        "position_range", (-0.05, 0.05)
      ),
      "reset range",
    ),
    (
      lambda cfg: cfg.events["reset_robot_joints"].params.__setitem__(
        "velocity_range", (-1.0, 1.0)
      ),
      "reset velocity",
    ),
    (lambda cfg: setattr(cfg, "scale_rewards_by_dt", 1), "reward dt scaling"),
    (
      lambda cfg: setattr(cfg.rewards["r_waypoint_progress"], "func", object),
      "guidance reward",
    ),
    (
      lambda cfg: setattr(cfg.rewards["r_waypoint_progress"], "weight", 7.0),
      "guidance reward",
    ),
    (
      lambda cfg: cfg.metrics["waypoint_progress"].params.__setitem__("extra", 1),
      "WaypointProgressTracker params",
    ),
    (
      lambda cfg: setattr(
        cfg.metrics["waypoint_progress"].params["robot_cfg"],
        "site_names",
        ("wrong",),
      ),
      "robot site binding",
    ),
    (
      lambda cfg: setattr(
        cfg.metrics["waypoint_progress"].params["nail_cfg"],
        "site_names",
        ("wrong",),
      ),
      "nail site binding",
    ),
    (
      lambda cfg: setattr(cfg.observations["actor"], "enable_corruption", 1),
      "literal observation corruption",
    ),
    (
      lambda cfg: setattr(cfg.observations["critic"], "enable_corruption", 0),
      "literal observation corruption",
    ),
    (
      lambda cfg: cfg.observations["critic"].terms.pop("waypoint_progress_state"),
      "critic guideline observation",
    ),
    (
      lambda cfg: setattr(cfg.metrics["cat_soft"], "func", object),
      "CaT metric",
    ),
    (
      lambda cfg: setattr(cfg.metrics["cat_soft"], "per_substep", True),
      "CaT metric",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("use_impulse", 1),
      "literal CaT booleans",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("use_vel", 1),
      "literal CaT booleans",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("imp_seed", 0.002),
      "impulse-CaT setting",
    ),
    (
      lambda cfg: setattr(
        cfg.metrics["cat_soft"].params["robot_cfg"], "name", "wrong"
      ),
      "CaT robot binding",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("limit", 3.0),
      "velocity-CaT setting",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("max_p", 0.4),
      "velocity-CaT setting",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("min_p", 0.1),
      "velocity-CaT setting",
    ),
    (
      lambda cfg: cfg.metrics["cat_soft"].params.__setitem__("tau", 0.9),
      "velocity-CaT setting",
    ),
    (
      lambda cfg: setattr(cfg.metrics["substep_peak_qv"], "func", object),
      "velocity-CaT setting",
    ),
    (
      lambda cfg: setattr(cfg.actions["ik_hammer_head"], "max_dq", 0.29),
      "fixed action signature",
    ),
    (
      lambda cfg: setattr(
        cfg.scene.entities["robot"].articulation.actuators[0], "stiffness", 1.0
      ),
      "fixed-impedance actuator signature",
    ),
  ),
)
def test_presentation3_contract_rejects_every_frozen_axis_mutation(mutate, message):
  """Each mutation changes behavior or identity and must fail before rollout."""
  task = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
  )
  cfg = eval_impulse.load_env_cfg(task, play=False)
  mutate(cfg)

  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_sampled_env_contract(cfg, task)


def test_presentation3_main_rejects_native_imp_max_before_evaluator_overwrite(
  monkeypatch
):
  """The evaluator must not heal an active-impulse training config to log-only."""
  task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.25
  monkeypatch.setattr(eval_impulse, "load_env_cfg", lambda *args, **kwargs: cfg)
  monkeypatch.setattr(
    eval_impulse,
    "_git_provenance",
    lambda path: {"revision": "b" * 40, "dirty": False, "status": ""},
  )

  def reached_post_validation(*args, **kwargs):
    raise RuntimeError("past-presentation3-validation")

  monkeypatch.setattr(eval_impulse, "load_rl_cfg", reached_post_validation)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "eval_impulse.py",
      "--campaign", PRESENTATION3_CAMPAIGN,
      "--task", task,
      "--ckpt", "model_499.pt",
      "--training-seed", "2",
      "--expected-checkpoint-sha256", "c" * 64,
      "--accepted-manifest-sha256", "d" * 64,
      "--training-code-revision", "e" * 40,
      "--training-asset-revision", "b" * 40,
    ],
  )

  with pytest.raises(ValueError, match="imp_max_p"):
    eval_impulse.main()


def test_presentation3_main_requires_full_row_identity_before_checkpoint_load(
  monkeypatch
):
  """Calling main without frozen hashes/revisions must stop before runner setup."""
  task = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"

  def reached_runner_setup(*args, **kwargs):
    raise RuntimeError("past-presentation3-row-identity")

  monkeypatch.setattr(eval_impulse, "load_rl_cfg", reached_runner_setup)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "eval_impulse.py",
      "--campaign", PRESENTATION3_CAMPAIGN,
      "--task", task,
      "--ckpt", "model_199.pt",
      "--training-seed", "2",
    ],
  )

  with pytest.raises(ValueError, match="checkpoint SHA-256"):
    eval_impulse.main()


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
GUIDELINE_CAMPAIGN = "cartesian-guideline-pilot"
GUIDELINE_RNG = {
  "reset_seed": 2036072919,
  "observation_seed": 2046072933,
  "action_seed": 2056072941,
}


def _guideline_identity_kwargs(
  arm="C0", **overrides
):
  asset_revision = "b" * 40
  values = {
    "campaign": GUIDELINE_CAMPAIGN,
    "env_cfg": eval_impulse.load_env_cfg(GUIDELINE_TASKS[arm], play=False),
    "task": GUIDELINE_TASKS[arm],
    "training_seed": 0,
    "checkpoint_path": Path("/frozen/run/model_499.pt"),
    "expected_checkpoint_sha256": "c" * 64,
    "accepted_manifest_sha256": "d" * 64,
    "training_code_revision": "e" * 40,
    "training_asset_revision": asset_revision,
    "code_git": {"revision": "a" * 40, "dirty": False, "status": ""},
    "asset_git": {
      "revision": asset_revision,
      "dirty": False,
      "status": "",
    },
  }
  values.update(overrides)
  return values


@pytest.mark.parametrize("task", tuple(GUIDELINE_TASKS.values()))
def test_guideline_campaign_accepts_only_the_frozen_rng_tuple(task):
  eval_impulse._validate_evaluation_campaign(
    GUIDELINE_CAMPAIGN,
    task=task,
    **GUIDELINE_RNG,
  )


@pytest.mark.parametrize(
  ("campaign", "task", "message"),
  (
    (None, GUIDELINE_TASKS["C0"], "requires --campaign"),
    (GUIDELINE_CAMPAIGN, eval_impulse.QUALITY_ARM_TASKS["FQ"], "guideline.*task"),
    (GUIDELINE_CAMPAIGN, GUIDELINE_TASKS["C0"] + "-near-match", "guideline.*task"),
  ),
)
def test_guideline_campaign_rejects_unscoped_or_nonexact_tasks(
  campaign, task, message
):
  with pytest.raises(ValueError, match=message):
    eval_impulse._validate_evaluation_campaign(
      campaign,
      task=task,
      **GUIDELINE_RNG,
    )


@pytest.mark.parametrize("stream", tuple(GUIDELINE_RNG))
def test_guideline_campaign_rejects_each_rng_stream_drift(stream):
  streams = dict(GUIDELINE_RNG)
  streams[stream] += 1
  with pytest.raises(ValueError, match=f"guideline.*{stream.removesuffix('_seed')}"):
    eval_impulse._validate_evaluation_campaign(
      GUIDELINE_CAMPAIGN,
      task=GUIDELINE_TASKS["C-Gate"],
      **streams,
    )


def test_guideline_dispatch_normalizes_literal_treatment_identity_fields():
  c0 = eval_impulse._validate_guideline_pilot_identity(
    **_guideline_identity_kwargs("C0")
  )
  cgate = eval_impulse._validate_guideline_pilot_identity(
    **_guideline_identity_kwargs("C-Gate")
  )

  expected_shared = {
    "reset_position_range_rad": (0.0, 0.0),
    "windup_enabled": False,
    "impedance_mode": "fixed",
    "imp_max_p": 0.0,
  }
  assert {key: c0[key] for key in expected_shared} == expected_shared
  assert {key: cgate[key] for key in expected_shared} == expected_shared
  assert c0["treatment_base_identity"] == cgate["treatment_base_identity"]
  assert c0["r_gate_present"] is False
  assert c0["r_gate_weight"] is None
  assert cgate["r_gate_present"] is True
  assert cgate["r_gate_weight"] == 8.0

  drifted_cfg = eval_impulse.load_env_cfg(GUIDELINE_TASKS["C0"], play=False)
  drifted_cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  drifted = eval_impulse._validate_guideline_pilot_identity(
    **_guideline_identity_kwargs("C0", env_cfg=drifted_cfg)
  )
  assert drifted["treatment_base_identity"] != c0["treatment_base_identity"]


@pytest.mark.parametrize(
  ("overrides", "message"),
  (
    ({"training_seed": 2}, "seeds 0/1"),
    ({"checkpoint_path": Path("/frozen/run/model_498.pt")}, "model_499.pt"),
    ({"expected_checkpoint_sha256": ""}, "expected checkpoint SHA-256"),
    ({"accepted_manifest_sha256": ""}, "accepted manifest SHA-256"),
    ({"training_code_revision": ""}, "training code revision"),
    ({"training_asset_revision": ""}, "training asset revision"),
    (
      {"code_git": {"revision": "a" * 40, "dirty": True, "status": " M file"}},
      "clean code provenance",
    ),
    (
      {"code_git": {"revision": "unknown", "dirty": True, "status": "unavailable"}},
      "clean code provenance",
    ),
    (
      {"asset_git": {"revision": "b" * 40, "dirty": True, "status": " M asset"}},
      "clean asset provenance",
    ),
    (
      {"asset_git": {"revision": "unknown", "dirty": True, "status": "unavailable"}},
      "clean asset provenance",
    ),
  ),
)
def test_guideline_dispatch_rejects_nonfrozen_identity_or_provenance(
  overrides, message
):
  with pytest.raises((ValueError, RuntimeError), match=message):
    eval_impulse._validate_guideline_pilot_identity(
      **_guideline_identity_kwargs(**overrides)
    )


def test_guideline_main_loads_play_false_and_validates_before_checkpoint(
  monkeypatch
):
  task = GUIDELINE_TASKS["C0"]
  source_load = eval_impulse.load_env_cfg
  calls = []

  def load_training_cfg(requested, *, play=False):
    calls.append((requested, play))
    return source_load(requested, play=play)

  def reject_native(cfg, requested):
    assert requested == task
    raise RuntimeError("native-guideline-sentinel")

  monkeypatch.setattr(eval_impulse, "load_env_cfg", load_training_cfg)
  monkeypatch.setattr(
    eval_impulse, "_validate_native_guideline_env_contract", reject_native
  )
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "eval_impulse.py",
      "--campaign", GUIDELINE_CAMPAIGN,
      "--task", task,
      "--ckpt", "model_499.pt",
      "--training-seed", "0",
      "--expected-checkpoint-sha256", "c" * 64,
      "--accepted-manifest-sha256", "d" * 64,
      "--training-code-revision", "e" * 40,
      "--training-asset-revision", "b" * 40,
    ],
  )

  with pytest.raises(RuntimeError, match="native-guideline-sentinel"):
    eval_impulse.main()
  assert calls == [(task, False)]


def test_guideline_main_rejects_native_imp_max_before_evaluator_overwrite(
  monkeypatch
):
  task = GUIDELINE_TASKS["C0"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.metrics["cat_soft"].params["imp_max_p"] = 0.25
  monkeypatch.setattr(eval_impulse, "load_env_cfg", lambda *args, **kwargs: cfg)
  monkeypatch.setattr(
    eval_impulse,
    "_git_provenance",
    lambda path: {"revision": "b" * 40, "dirty": False, "status": ""},
  )

  def reached_post_validation(*args, **kwargs):
    raise RuntimeError("past-native-validation")

  monkeypatch.setattr(eval_impulse, "load_rl_cfg", reached_post_validation)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "eval_impulse.py",
      "--campaign", GUIDELINE_CAMPAIGN,
      "--task", task,
      "--ckpt", "model_499.pt",
      "--training-seed", "0",
      "--expected-checkpoint-sha256", "c" * 64,
      "--accepted-manifest-sha256", "d" * 64,
      "--training-code-revision", "e" * 40,
      "--training-asset-revision", "b" * 40,
    ],
  )

  with pytest.raises(ValueError, match="imp_max_p"):
    eval_impulse.main()


def test_guideline_extension_preserves_legacy_digest_literals():
  task = eval_impulse.QUALITY_ARM_TASKS["FQ"]
  _, cfg, _ = eval_impulse.build_strict_quality_evaluation_cfg(task, play=False)
  contract = eval_impulse._validate_sampled_env_contract(cfg, task)

  assert eval_impulse._campaign_config_digest(
    contract=contract,
    num_envs=256,
    episode_len_s=4.0,
    physics_dt_s=contract["physics_dt_s"],
    decimation=contract["control_decimation"],
    reset_seed=2036073019,
    observation_seed=2046073033,
    action_seed=2056073041,
  ) == "831177002bdd74f661ef74f5aaec5a964f57d257721f66926605d4a07e3e5822"
  assert eval_impulse._treatment_config_digest(
    task=task, contract=contract
  ) == "37146ea3b5cf0e90262179c1dfea83bda63fc377bb5445ecf38145bc4500758b"


def _guideline_pilot_csv_rows():
  common = {
    "checkpoint_filename": "model_499.pt",
    "num_envs": 256,
    "episode_len_s": 4.0,
    "n_episodes_sampled": 512,
    "episodes_per_env_sampled": 2,
    "reset_digest": "1" * 64,
    "guideline_geometry_digest": "2" * 64,
    "reset_position_range_rad": (0.0, 0.0),
    "windup_enabled": False,
    "impedance_mode": "fixed",
    "imp_max_p": 0.0,
    "treatment_base_identity": "3" * 64,
    "reset_rng_seed": GUIDELINE_RNG["reset_seed"],
    "observation_rng_seed": GUIDELINE_RNG["observation_seed"],
    "action_rng_seed": GUIDELINE_RNG["action_seed"],
    "git_dirty": False,
    "asset_git_dirty": False,
    "git_revision": "a" * 40,
    "asset_git_revision": "b" * 40,
    "accepted_manifest_sha256": "4" * 64,
    "training_code_revision": "c" * 40,
    "training_asset_revision": "b" * 40,
    "campaign_config_sha256": "5" * 64,
    "impossible_success_n": 0,
    "lambda_dead_n": 0,
    "qvel_nonfinite_rate_sampled": 0.0,
    "q90_terminal_descent_perpendicular_error_m_sampled": 0.004,
    "all_six_gates_rate_sampled": 0.75,
    "corridor_occupancy_mean_sampled": 0.80,
    "backward_progress_count_mean_sampled": 0.25,
  }
  return [
    dict(
      common, task=GUIDELINE_TASKS["C0"], treatment="C0", training_seed=0,
      checkpoint_sha256="6" * 64, accepted_checkpoint_sha256="6" * 64,
      treatment_config_sha256="7" * 64,
      sampled_trace_digest="8" * 64,
      sampled_trace_artifact_sha256="9" * 64, r_gate_present=False,
      r_gate_weight=None, gate_reward_present=False,
      actual_gate_return_total_sampled=0.0, success_rate_sampled=0.30,
      first_strike_useful_speed_mean_sampled=0.41,
      worst_ratio_max_sampled=0.61,
      qvel_finite_exceedance_rate_sampled=0.01,
      qvel_max_abs_rad_s_sampled=2.1,
    ),
    dict(
      common, task=GUIDELINE_TASKS["C0"], treatment="C0", training_seed=1,
      checkpoint_sha256="a" * 64, accepted_checkpoint_sha256="a" * 64,
      treatment_config_sha256="7" * 64,
      sampled_trace_digest="b" * 64,
      sampled_trace_artifact_sha256="c" * 64, r_gate_present=False,
      r_gate_weight=None, gate_reward_present=False,
      actual_gate_return_total_sampled=0.0, success_rate_sampled=0.10,
      first_strike_useful_speed_mean_sampled=0.42,
      worst_ratio_max_sampled=0.62,
      qvel_finite_exceedance_rate_sampled=0.02,
      qvel_max_abs_rad_s_sampled=2.2,
    ),
    dict(
      common, task=GUIDELINE_TASKS["C-Gate"], treatment="C-Gate", training_seed=0,
      checkpoint_sha256="d" * 64, accepted_checkpoint_sha256="d" * 64,
      treatment_config_sha256="e" * 64,
      sampled_trace_digest="f" * 64,
      sampled_trace_artifact_sha256="0" * 64, r_gate_present=True,
      r_gate_weight=8.0, gate_reward_present=True,
      actual_gate_return_total_sampled=1.0, success_rate_sampled=0.25,
      first_strike_useful_speed_mean_sampled=0.43,
      worst_ratio_max_sampled=0.63,
      qvel_finite_exceedance_rate_sampled=0.03,
      qvel_max_abs_rad_s_sampled=2.3,
    ),
    dict(
      common, task=GUIDELINE_TASKS["C-Gate"], treatment="C-Gate", training_seed=1,
      checkpoint_sha256="1" * 64, accepted_checkpoint_sha256="1" * 64,
      treatment_config_sha256="e" * 64,
      sampled_trace_digest="2" * 64,
      sampled_trace_artifact_sha256="3" * 64, r_gate_present=True,
      r_gate_weight=8.0, gate_reward_present=True,
      actual_gate_return_total_sampled=0.5, success_rate_sampled=0.10,
      first_strike_useful_speed_mean_sampled=0.44,
      worst_ratio_max_sampled=0.64,
      qvel_finite_exceedance_rate_sampled=0.04,
      qvel_max_abs_rad_s_sampled=2.4,
    ),
  ]


def _write_actual_guideline_csv(path, rows):
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=eval_impulse.FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def test_guideline_csv_round_trip_reaches_strict_four_row_validator(tmp_path):
  rows = _guideline_pilot_csv_rows()
  csv_path = tmp_path / "summary.csv"
  _write_actual_guideline_csv(csv_path, rows)

  result = guideline_analysis.load_and_validate_guideline_pilot_csv(csv_path)

  assert result["valid"] is True
  assert result["row_identities"] == [
    ("C0", 0), ("C0", 1), ("C-Gate", 0), ("C-Gate", 1)
  ]
  assert result["validated_secondary_results"] == [
    {
      "treatment": "C0", "training_seed": 0,
      "first_strike_useful_speed_mean_sampled": 0.41,
      "worst_ratio_max_sampled": 0.61,
      "qvel_finite_exceedance_rate_sampled": 0.01,
      "qvel_max_abs_rad_s_sampled": 2.1,
    },
    {
      "treatment": "C0", "training_seed": 1,
      "first_strike_useful_speed_mean_sampled": 0.42,
      "worst_ratio_max_sampled": 0.62,
      "qvel_finite_exceedance_rate_sampled": 0.02,
      "qvel_max_abs_rad_s_sampled": 2.2,
    },
    {
      "treatment": "C-Gate", "training_seed": 0,
      "first_strike_useful_speed_mean_sampled": 0.43,
      "worst_ratio_max_sampled": 0.63,
      "qvel_finite_exceedance_rate_sampled": 0.03,
      "qvel_max_abs_rad_s_sampled": 2.3,
    },
    {
      "treatment": "C-Gate", "training_seed": 1,
      "first_strike_useful_speed_mean_sampled": 0.44,
      "worst_ratio_max_sampled": 0.64,
      "qvel_finite_exceedance_rate_sampled": 0.04,
      "qvel_max_abs_rad_s_sampled": 2.4,
    },
  ]


@pytest.mark.parametrize(
  ("field", "mode"),
  (
    ("first_strike_useful_speed_mean_sampled", "blank"),
    ("worst_ratio_max_sampled", "blank"),
    ("qvel_finite_exceedance_rate_sampled", "blank"),
    ("qvel_max_abs_rad_s_sampled", "blank"),
    ("first_strike_useful_speed_mean_sampled", "missing"),
    ("worst_ratio_max_sampled", "missing"),
    ("qvel_finite_exceedance_rate_sampled", "missing"),
    ("qvel_max_abs_rad_s_sampled", "missing"),
    ("first_strike_useful_speed_mean_sampled", "nonfinite"),
    ("worst_ratio_max_sampled", "nonfinite"),
    ("qvel_finite_exceedance_rate_sampled", "nonfinite"),
    ("qvel_max_abs_rad_s_sampled", "nonfinite"),
    ("first_strike_useful_speed_mean_sampled", "negative"),
    ("worst_ratio_max_sampled", "negative"),
    ("qvel_finite_exceedance_rate_sampled", "negative"),
    ("qvel_max_abs_rad_s_sampled", "negative"),
    ("qvel_finite_exceedance_rate_sampled", "above-one"),
  ),
)
def test_actual_evaluator_csv_rejects_invalid_secondary_outputs(
  tmp_path, field, mode
):
  rows = _guideline_pilot_csv_rows()
  if mode == "missing":
    rows[0].pop(field)
  elif mode == "blank":
    rows[0][field] = ""
  elif mode == "nonfinite":
    rows[0][field] = math.nan
  elif mode == "negative":
    rows[0][field] = -0.01
  else:
    rows[0][field] = 1.01
  csv_path = tmp_path / "summary.csv"
  _write_actual_guideline_csv(csv_path, rows)

  with pytest.raises(ValueError, match=field):
    guideline_analysis.load_and_validate_guideline_pilot_csv(csv_path)


@pytest.mark.parametrize(
  "mutate",
  (
    lambda rows: [row.__setitem__("reset_rng_seed", 999) for row in rows],
    lambda rows: [row.pop("git_dirty") for row in rows],
    lambda rows: [row.pop("q90_terminal_descent_perpendicular_error_m_sampled") for row in rows],
    lambda rows: rows[0].__setitem__("accepted_checkpoint_sha256", "f" * 64),
    lambda rows: rows[0].__setitem__("campaign_config_sha256", "f" * 64),
  ),
  ids=("frozen-rng", "missing-dirty", "missing-q90", "checkpoint", "config"),
)
def test_actual_evaluator_csv_rejects_load_bearing_tampering(tmp_path, mutate):
  rows = _guideline_pilot_csv_rows()
  mutate(rows)
  csv_path = tmp_path / "summary.csv"
  _write_actual_guideline_csv(csv_path, rows)

  with pytest.raises(ValueError):
    guideline_analysis.load_and_validate_guideline_pilot_csv(csv_path)


@pytest.fixture(scope="module")
def guideline_autoreset_records():
  records = {}
  for arm, task in GUIDELINE_TASKS.items():
    cfg = eval_impulse.load_env_cfg(task, play=False)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = float(cfg.sim.mujoco.timestep * cfg.decimation)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=arm,
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
    try:
      env.reset()
      action = torch.zeros(
        (1, env.action_manager.total_action_dim), device=env.device
      )
      env.step(action)
      assert len(collector.completed) == 1
      first_reference = collector.completed[0]
      first_frozen = copy.deepcopy(first_reference)
      env.step(action)
      assert len(collector.completed) == 2
      records[arm] = {
        "first": copy.deepcopy(first_reference),
        "second": copy.deepcopy(collector.completed[1]),
        "first_unchanged": first_reference == first_frozen,
      }
    finally:
      env.close()
  return records


@pytest.fixture(scope="module")
def presentation3_autoreset_records():
  """One real timeout completion from every registered Presentation3 task."""
  records = {}
  for arm, task, _, _ in _PRESENTATION3_CONTRACTS:
    cfg = eval_impulse.load_env_cfg(task, play=False)
    contract = eval_impulse._validate_sampled_env_contract(cfg, task)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = float(cfg.sim.mujoco.timestep * cfg.decimation)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    snapshot = eval_impulse._install_episode_hook(env)
    collector = eval_impulse._SampledTraceCollector(
      env,
      snapshot=snapshot,
      treatment=arm,
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
    try:
      env.reset()
      action = torch.zeros(
        (1, env.action_manager.total_action_dim), device=env.device
      )
      env.step(action)
      assert len(collector.completed) == 1
      records[arm] = {
        "task": task,
        "contract": contract,
        "trace": copy.deepcopy(collector.completed[0]),
      }
    finally:
      env.close()
  return records


@pytest.mark.parametrize("arm", ("M", "V", "V+M"))
def test_presentation3_real_completion_validates_and_persists_progress_schema(
  tmp_path, presentation3_autoreset_records, arm
):
  """A real completion must survive validation and schema-v3 persistence."""
  record = presentation3_autoreset_records[arm]
  trace = record["trace"]
  assert eval_impulse._validated_physical_trace_digest(
    trace, require_recorded_digest=True
  ) == trace["trace_digest"]
  assert set(trace["guideline"]) == {
    "entry_m",
    "nail_m",
    "next_gate",
    "perpendicular_error_m",
    "disarmed",
    "progress_reward_name",
    "progress_reward_present",
    "progress_payout",
  }
  assert trace["guideline"]["progress_reward_name"] == "r_waypoint_progress"
  assert trace["guideline"]["progress_reward_present"] is True
  assert len(trace["guideline"]["progress_payout"]) == len(trace["action_tape"])

  artifact = eval_impulse._persist_sampled_traces(
    out_dir=tmp_path,
    name=f"presentation3-{arm}-seed2",
    sampled_rec={"control_steps": 1, "episodes": [trace]},
    task=record["task"],
    contract=record["contract"],
    training_seed=2,
    reset_seed=PRESENTATION3_RNG["reset_seed"],
    observation_seed=PRESENTATION3_RNG["observation_seed"],
    action_seed=PRESENTATION3_RNG["action_seed"],
    nail_geometry=trace["nail_geometry"],
    provenance={"checkpoint_sha256": "c" * 64},
    mean_rollout_invariants={},
  )
  with eval_impulse.np.load(artifact["path"], allow_pickle=False) as saved:
    payload = json.loads(eval_impulse.decode_payload_json(saved))
  assert payload["guideline_trace_contract_version"] == 1
  assert payload["weights"]["r_waypoint_progress"] == 8.0
  assert payload["episodes"][0]["guideline"] == trace["guideline"]


def test_presentation3_collector_records_actual_waypoint_progress_payout():
  """Swapping the progress reward index for r_gate/zero must fail this test."""
  arm, task, _, _ = _PRESENTATION3_CONTRACTS[0]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  snapshot = eval_impulse._install_episode_hook(env)
  collector = eval_impulse._SampledTraceCollector(
    env,
    snapshot=snapshot,
    treatment=arm,
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
  try:
    env.reset()
    env.sim.step()
    env.metrics_manager.compute_substep()
    tracker = getattr(env, eval_impulse._ENV_GUIDELINE_ATTR)
    tracker.window_new_credit[:] = 0.25
    before = env.reward_manager._episode_sums["r_waypoint_progress"].clone()
    env.reward_manager.compute(env.step_dt)
    actual = env.reward_manager._episode_sums["r_waypoint_progress"] - before
    assert float(actual[0]) > 0.0

    env.reset_buf = torch.ones(1, dtype=torch.bool, device=env.device)
    env.reset_terminated = torch.zeros(1, dtype=torch.bool, device=env.device)
    env.metrics_manager.compute()
    assert collector.completed[0]["guideline"]["progress_payout"] == pytest.approx(
      actual.tolist()
    )
  finally:
    env.close()


def test_presentation3_progress_identity_and_payout_are_digest_bound(
  presentation3_autoreset_records,
):
  trace = copy.deepcopy(presentation3_autoreset_records["V"]["trace"])
  baseline = eval_impulse._physical_trace_digest(trace)
  wrong_name = copy.deepcopy(trace)
  wrong_name["guideline"]["progress_reward_name"] = "r_gate"
  assert eval_impulse._physical_trace_digest(wrong_name) != baseline
  with pytest.raises(ValueError, match="progress reward identity"):
    eval_impulse._validated_physical_trace_digest(
      wrong_name, require_recorded_digest=False
    )

  wrong_payout = copy.deepcopy(trace)
  wrong_payout["guideline"]["progress_payout"][0] += 0.125
  assert eval_impulse._physical_trace_digest(wrong_payout) != baseline


def test_presentation3_guideline_seed_fields_reduce_progress_not_gate_return(
  presentation3_autoreset_records,
):
  record = presentation3_autoreset_records["M"]
  fields = eval_impulse._guideline_seed_trace_fields(
    [record["trace"]], contract=record["contract"], expected_episode_count=1
  )
  assert fields["waypoint_progress_reward_present"] is True
  assert "actual_gate_return_total_sampled" not in fields
  assert fields["actual_waypoint_progress_return_total_sampled"] == pytest.approx(
    sum(record["trace"]["guideline"]["progress_payout"])
  )


def test_presentation3_row_fields_name_progress_without_gate_alias(
  presentation3_autoreset_records,
):
  record = presentation3_autoreset_records["V+M"]
  contract = eval_impulse._validate_presentation3_identity(
    **_presentation3_identity_kwargs("V+M")
  )
  guideline_fields = eval_impulse._guideline_seed_trace_fields(
    [record["trace"]], contract=contract, expected_episode_count=1
  )
  builder = getattr(eval_impulse, "_guideline_row_fields", None)
  assert callable(builder), "guideline row-field builder is missing"

  row_fields = builder(
    task=record["task"],
    checkpoint_path="model_199.pt",
    contract=contract,
    sampled_rec={"guideline_fields": guideline_fields},
  )

  assert row_fields["r_waypoint_progress_present"] is True
  assert row_fields["r_waypoint_progress_weight"] == 8.0
  assert row_fields["waypoint_progress_reward_present"] is True
  assert "actual_waypoint_progress_return_total_sampled" in row_fields
  assert "r_gate_present" not in row_fields
  assert "actual_gate_return_total_sampled" not in row_fields
  assert set(row_fields) <= set(eval_impulse.FIELDNAMES)


def _synthetic_guideline_trace(source, *, arm="C0"):
  trace = copy.deepcopy(source)
  head = torch.tensor(trace["physical"]["head_position_m"], dtype=torch.float64)
  entry = head[0].clone()
  nail = entry.clone()
  nail[2] -= 0.1
  direction = nail - entry
  progress = (
    ((head - entry) @ direction) / torch.dot(direction, direction)
  ).clamp(0.0, 1.0)
  closest = entry + progress[:, None] * direction
  errors = torch.linalg.vector_norm(head - closest, dim=-1)
  count = len(trace["physical"]["contact"])
  trace.update(
    {
      "arm": arm,
      "task": GUIDELINE_TASKS[arm],
      "guideline_trace_contract_version": 1,
      "impulse_limits_n_m_s": [1.64, 3.28, 1.64, 1.64, 1.64, 1.64],
      "guideline": {
        "entry_m": entry.tolist(),
        "nail_m": nail.tolist(),
        "next_gate": [0] * count,
        "perpendicular_error_m": errors.tolist(),
        "disarmed": list(trace["event_trace"]["tracker_started"]),
        "gate_reward_present": arm == "C-Gate",
        "gate_payout": [0.0] * len(trace["action_tape"]),
      },
    }
  )
  return trace


@pytest.mark.parametrize("arm", tuple(GUIDELINE_TASKS))
def test_guideline_collector_persists_episode_local_aligned_tracker_state_and_payout(
  guideline_autoreset_records, arm
):
  record = guideline_autoreset_records[arm]
  trace = record["first"]
  assert record["first_unchanged"] is True
  assert trace["guideline_trace_contract_version"] == 1
  guideline = trace["guideline"]
  assert set(guideline) == {
    "entry_m",
    "nail_m",
    "next_gate",
    "perpendicular_error_m",
    "disarmed",
    "gate_reward_present",
    "gate_payout",
  }
  count = len(trace["physical"]["contact"])
  assert len(guideline["entry_m"]) == len(guideline["nail_m"]) == 3
  assert guideline["entry_m"] == pytest.approx(
    trace["physical"]["head_position_m"][0]
  )
  assert len(guideline["next_gate"]) == count
  assert len(guideline["perpendicular_error_m"]) == count
  assert len(guideline["disarmed"]) == count
  assert len(guideline["gate_payout"]) == len(trace["action_tape"]) == 1
  assert guideline["gate_reward_present"] is (arm == "C-Gate")
  assert all(value >= 0.0 for value in guideline["gate_payout"])
  if arm == "C0":
    assert guideline["gate_payout"] == [0.0]
  assert all(
    later >= earlier
    for earlier, later in zip(
      guideline["next_gate"], guideline["next_gate"][1:]
    )
  )
  assert eval_impulse._validated_physical_trace_digest(
    trace, require_recorded_digest=True
  ) == trace["trace_digest"]


def test_guideline_collector_captures_real_gate_transition_disarm_and_payout():
  task = GUIDELINE_TASKS["C-Gate"]
  cfg = eval_impulse.load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  snapshot = eval_impulse._install_episode_hook(env)
  collector = eval_impulse._SampledTraceCollector(
    env,
    snapshot=snapshot,
    treatment="C-Gate",
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
  try:
    env.reset()
    tracker = getattr(env, eval_impulse._ENV_GUIDELINE_ATTR)
    first_strike = getattr(env, eval_impulse._ENV_FIRST_STRIKE_ATTR)
    env.sim.step()
    entry = tracker._head_position().detach().clone()
    nail = entry.clone()
    nail[:, 0] += 0.7
    tracker.initialized[:] = True
    tracker.entry.copy_(entry)
    tracker.nail.copy_(nail)
    tracker.previous_head.copy_(entry)
    env.metrics_manager.compute_substep()

    env.sim.step()
    crossed_head = entry.clone()
    crossed_head[:, 0] += 0.15
    head_site_id = tracker._head_site_ids[0]
    model_site_id = tracker._robot.data.indexing.site_ids[head_site_id]
    tracker._robot.data.data.site_xpos[:, model_site_id, :] = crossed_head
    env.metrics_manager.compute_substep()

    env.sim.step()
    tracker._robot.data.data.site_xpos[:, model_site_id, :] = crossed_head
    first_strike._state[:] = 2
    env.metrics_manager.compute_substep()

    before = env.reward_manager._episode_sums["r_gate"].detach().clone()
    env.reward_manager.compute(env.step_dt)
    actual_manager_contribution = (
      env.reward_manager._episode_sums["r_gate"] - before
    )
    assert float(actual_manager_contribution[0]) > 0.0

    env.reset_buf = torch.ones(1, dtype=torch.bool, device=env.device)
    env.reset_terminated = torch.zeros(
      1, dtype=torch.bool, device=env.device
    )
    env.metrics_manager.compute()
    assert len(collector.completed) == 1
    guideline = collector.completed[0]["guideline"]
    assert guideline["next_gate"] == [0, 1, 1]
    assert guideline["disarmed"] == [False, False, True]
    assert guideline["gate_payout"] == pytest.approx(
      actual_manager_contribution.tolist()
    )
  finally:
    env.close()


@pytest.mark.parametrize(
  "mutate",
  (
    lambda trace: trace["guideline"]["entry_m"].__setitem__(0, 0.25),
    lambda trace: trace["guideline"]["nail_m"].__setitem__(2, -0.25),
    lambda trace: trace["guideline"]["next_gate"].__setitem__(0, 1),
    lambda trace: trace["guideline"]["perpendicular_error_m"].__setitem__(0, 0.01),
    lambda trace: trace["guideline"]["disarmed"].__setitem__(0, True),
    lambda trace: trace["guideline"].__setitem__("gate_reward_present", True),
    lambda trace: trace["guideline"]["gate_payout"].__setitem__(0, 0.1),
    lambda trace: trace["impulse_limits_n_m_s"].__setitem__(0, 1.63),
  ),
)
def test_guideline_trace_digest_binds_every_additive_field(
  guideline_autoreset_records, mutate
):
  trace = _synthetic_guideline_trace(
    guideline_autoreset_records["C0"]["first"]
  )
  baseline = eval_impulse._physical_trace_digest(trace)
  mutate(trace)
  assert eval_impulse._physical_trace_digest(trace) != baseline


@pytest.mark.parametrize(
  "missing",
  (
    "entry_m",
    "nail_m",
    "next_gate",
    "perpendicular_error_m",
    "disarmed",
    "gate_reward_present",
    "gate_payout",
    "guideline_trace_contract_version",
    "impulse_limits_n_m_s",
  ),
)
def test_guideline_trace_validation_rejects_each_missing_channel(
  guideline_autoreset_records, missing
):
  trace = _synthetic_guideline_trace(
    guideline_autoreset_records["C0"]["first"]
  )
  if missing in trace:
    trace.pop(missing)
  else:
    trace["guideline"].pop(missing)
  with pytest.raises(ValueError, match="guideline"):
    eval_impulse._validated_physical_trace_digest(
      trace, require_recorded_digest=False
    )


@pytest.mark.parametrize(
  "mutate",
  (
    lambda trace: trace.__setitem__("guideline_trace_contract_version", 2),
    lambda trace: trace.__setitem__("guideline_trace_contract_version", 1.0),
    lambda trace: trace["guideline"]["entry_m"].__setitem__(0, float("nan")),
    lambda trace: trace["guideline"]["perpendicular_error_m"].__setitem__(0, -0.1),
    lambda trace: trace["guideline"]["next_gate"].__setitem__(0, 7),
    lambda trace: trace["guideline"]["next_gate"].__setitem__(0, 0.5),
    lambda trace: trace["guideline"]["next_gate"].__setitem__(
      slice(0, 3), [0, 1, 0]
    ),
    lambda trace: trace["guideline"]["disarmed"].__setitem__(
      slice(0, 3), [False, True, False]
    ),
    lambda trace: trace["guideline"]["next_gate"].pop(),
    lambda trace: trace["guideline"]["gate_payout"].append(0.0),
    lambda trace: trace["guideline"]["gate_payout"].__setitem__(0, 0.1),
    lambda trace: trace["guideline"].__setitem__("gate_reward_present", True),
  ),
)
def test_guideline_trace_validation_fails_closed_on_malformed_or_c0_inconsistent_values(
  guideline_autoreset_records, mutate
):
  trace = _synthetic_guideline_trace(
    guideline_autoreset_records["C0"]["first"]
  )
  mutate(trace)
  with pytest.raises(ValueError, match="guideline"):
    eval_impulse._validated_physical_trace_digest(
      trace, require_recorded_digest=False
    )


def test_guideline_trace_validation_rejects_cgate_absence(
  guideline_autoreset_records
):
  trace = _synthetic_guideline_trace(
    guideline_autoreset_records["C-Gate"]["first"], arm="C-Gate"
  )
  trace["guideline"]["gate_reward_present"] = False
  with pytest.raises(ValueError, match="guideline"):
    eval_impulse._validated_physical_trace_digest(
      trace, require_recorded_digest=False
    )


def test_guideline_seed_fields_use_pure_reducer_and_bind_reset_geometry(
  guideline_autoreset_records
):
  episodes = [
    _synthetic_guideline_trace(trace)
    for trace in (
      guideline_autoreset_records["C0"]["first"],
      guideline_autoreset_records["C0"]["second"],
    )
  ]
  contract = eval_impulse._validate_guideline_pilot_identity(
    **_guideline_identity_kwargs("C0")
  )

  fields = eval_impulse._guideline_seed_trace_fields(
    episodes,
    contract=contract,
    expected_episode_count=2,
  )

  assert fields["q90_terminal_descent_perpendicular_error_m_sampled"] == 0.050
  assert fields["all_six_gates_rate_sampled"] == 0.0
  assert fields["actual_gate_return_total_sampled"] == 0.0
  assert fields["gate_reward_present"] is False
  assert fields["reset_digest"] == episodes[0]["reset_state_digest"]
  assert len(fields["guideline_geometry_digest"]) == 64
  assert {
    "checkpoint_filename",
    "reset_position_range_rad",
    "windup_enabled",
    "impedance_mode",
    "r_gate_present",
    "r_gate_weight",
    "treatment_base_identity",
    "gate_reward_present",
    "reset_digest",
    "guideline_geometry_digest",
    "q90_terminal_descent_perpendicular_error_m_sampled",
    "all_six_gates_rate_sampled",
    "corridor_occupancy_mean_sampled",
    "backward_progress_count_mean_sampled",
    "actual_gate_return_total_sampled",
  } <= set(eval_impulse.FIELDNAMES)


def test_guideline_persistence_banks_contract_version(
  tmp_path, guideline_autoreset_records
):
  trace = _synthetic_guideline_trace(
    guideline_autoreset_records["C0"]["first"]
  )
  trace["trace_digest"] = eval_impulse._physical_trace_digest(trace)
  contract = eval_impulse._validate_guideline_pilot_identity(
    **_guideline_identity_kwargs("C0")
  )
  artifact = eval_impulse._persist_sampled_traces(
    out_dir=tmp_path,
    name="guideline-C0-seed0",
    sampled_rec={"control_steps": 1, "episodes": [trace]},
    task=GUIDELINE_TASKS["C0"],
    contract=contract,
    training_seed=0,
    reset_seed=GUIDELINE_RNG["reset_seed"],
    observation_seed=GUIDELINE_RNG["observation_seed"],
    action_seed=GUIDELINE_RNG["action_seed"],
    nail_geometry=trace["nail_geometry"],
    provenance={"checkpoint_sha256": "c" * 64},
    mean_rollout_invariants={},
  )
  with eval_impulse.np.load(artifact["path"], allow_pickle=False) as saved:
    payload = json.loads(eval_impulse.decode_payload_json(saved))
  assert payload["guideline_trace_contract_version"] == 1
  assert payload["episodes"][0]["guideline"] == trace["guideline"]


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
      lambda cfg: setattr(cfg, "scale_rewards_by_dt", False),
      "reward dt scaling",
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
    "waypoint_progress_state",
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
    ("waypoint_progress_state", "reader", object),
    ("waypoint_progress_state", "width", 1),
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


def test_native_guideline_contract_accepts_the_registered_four_observations():
    """The evaluator must accept the shipped guideline observation block as-is."""
    from src.tasks.hammer.mdp import waypoint_progress_state

    for arm, task in GUIDELINE_TASKS.items():
        cfg = eval_impulse.load_env_cfg(task, play=False)
        eval_impulse._validate_native_guideline_env_contract(cfg, task)
        for group_name in ("actor", "critic"):
            terms = cfg.observations[group_name].terms
            guideline = tuple(
                name for name, term in terms.items()
                if term.func is eval_impulse._guideline_observation
            )
            assert guideline == (
                "next_gate_vector",
                "completed_gate_fraction",
                "guideline_perpendicular_error",
                "waypoint_progress_state",
            ), arm
            assert tuple(terms)[-4:] == guideline, arm
            assert terms["waypoint_progress_state"].params == {
                "reader": waypoint_progress_state,
                "width": 2,
            }, arm


def test_native_guideline_contract_rejects_progress_state_out_of_order():
    """waypoint_progress_state must stay last; reordering is drift, not style."""
    task = GUIDELINE_TASKS["C0"]
    cfg = eval_impulse.load_env_cfg(task, play=False)
    terms = cfg.observations["actor"].terms
    moved = terms.pop("waypoint_progress_state")
    reordered = {"waypoint_progress_state": moved, **terms}
    cfg.observations["actor"].terms = reordered

    with pytest.raises(ValueError, match="ordering"):
        eval_impulse._validate_native_guideline_env_contract(cfg, task)


def test_native_guideline_contract_rejects_a_missing_progress_state():
    """Dropping the fourth observation must fail closed, not be tolerated."""
    task = GUIDELINE_TASKS["C0"]
    cfg = eval_impulse.load_env_cfg(task, play=False)
    del cfg.observations["actor"].terms["waypoint_progress_state"]

    with pytest.raises(ValueError, match="waypoint_progress_state"):
        eval_impulse._validate_native_guideline_env_contract(cfg, task)

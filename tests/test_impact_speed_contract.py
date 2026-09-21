"""Checkpoint, live startup, and deterministic-export contracts for impact speed."""

from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv

from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
from src.tasks.hammer.rl.runner import (
    HammerOnPolicyRunner,
    _get_hammer_metadata,
    _joint_observation_metadata,
    _summarize_vic_rollout_telemetry,
    _vic_controller_metadata,
    _JOINT_POSITION_CONTRACT_PATH,
)
from src.tasks.hammer.config.z1.joint_position_contract import load_joint_position_contract


def _contract():
    return importlib.import_module("src.tasks.hammer.rl.impact_speed_contract")


def _cfg():
    cfg = _impact_speed_env_cfg(play=True)
    cfg.scene.num_envs = 1
    # Arithmetic software-test inputs, not physically qualified speed bounds/cost.
    cfg.actions["strike_speed"].bounds = (0.2, 0.8)
    cfg.rewards["speed_goal"].weight = -0.4
    return cfg


def _runner_cfg():
    cfg = asdict(z1_impact_speed_ppo_runner_cfg())
    cfg["actor"]["hidden_dims"] = [16, 8]
    cfg["critic"]["hidden_dims"] = [16, 8]
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 1
    cfg["logger"] = "tensorboard"
    cfg["upload_model"] = False
    return cfg


@pytest.fixture(scope="module")
def live_env():
    env = ManagerBasedRlEnv(_cfg(), device="cpu")
    try:
        yield env
    finally:
        env.close()


def test_runner_rejects_swapped_binary_goal_and_pending_before_inference(live_env, monkeypatch):
    """A model sees only 42 grouped values: names must be checked before it runs."""
    manager = live_env.observation_manager
    for group in ("actor", "critic"):
        swapped = list(manager.active_terms[group])
        swapped[-2:] = reversed(swapped[-2:])
        monkeypatch.setitem(manager._group_obs_term_names, group, swapped)
    with pytest.raises(ValueError, match="observation"):
        HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg())


def test_runner_rejects_ordinary_empirical_normalizer(live_env):
    cfg = _runner_cfg()
    cfg["actor"]["class_name"] = "MLPModel"
    cfg["critic"]["class_name"] = "MLPModel"
    with pytest.raises(ValueError, match="normaliz|ImpactSpeedModel"):
        HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), cfg)


def test_metadata_captures_live_controller_speed_and_event_units(live_env):
    metadata = _contract().build_impact_speed_metadata(live_env, 1.0)
    assert metadata["schema_version"] == "impact_speed_v1"
    assert metadata["distribution_parameters"] == ["mean", "std", "activity_mask"]
    assert metadata["action_terms"] == ["joint_position", "joint_stiffness", "strike_speed"]
    assert metadata["action_term_dims"] == [6, 6, 1]
    assert metadata["action_dim"] == 13
    assert metadata["observation_names"][-2:] == ["impact_speed_goal", "impact_speed_pending"]
    assert metadata["observation_term_widths"] == [7, 7, 3, 3, 3, 3, 3, 1, 1, 3, 6, 1, 1]
    assert metadata["observation_widths"] == {"actor": 42, "critic": 42}
    assert metadata["observation_normalization"] == {
        "type": "physical40_empirical_fixed_tail_v1", "empirical_width": 40,
        "fixed_tail": ["impact_speed_goal", "impact_speed_pending"], "empirical_eps": 0.01,
    }
    assert metadata["speed_bounds_mps"] == [0.2, 0.8]
    assert metadata["speed_axis_world"] == [0.0, 0.0, -1.0]
    assert metadata["speed_mode"] == "policy"
    assert metadata["speed_cost"]["coefficient"] == 0.4
    assert metadata["speed_cost"]["units"] == "dimensionless_per_event"
    assert metadata["speed_cost"]["timeout_semantics"] == "reset_boundary_shaping"
    assert metadata["variable_impedance"]["C"] == 1.25
    assert metadata["action_observation_source"] == "joint_position"
    assert metadata["action_rate_source"] == "joint_position"
    assert metadata["safety"]["cat_soft"]["use_vel"] is True
    assert metadata["safety"]["cat_soft"]["imp_max_p"] == 0.2
    json.dumps(metadata, allow_nan=False)
    exported = _get_hammer_metadata(live_env, "test", raw_policy_clip=1.0)
    assert exported == {"run_path": "test", **metadata}


def test_external_metadata_validation_does_not_advance_goal_sampling_rng(live_env, monkeypatch):
    speed = live_env.action_manager.get_term("strike_speed")
    monkeypatch.setattr(speed.cfg, "mode", "external")
    monkeypatch.setattr(speed.state, "mode", "external")
    before = torch.random.get_rng_state().clone()
    metadata = _contract().build_impact_speed_metadata(live_env, 1.0)
    assert metadata["speed_mode"] == "external"
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("mutation,match", [
    ("dimensions", "dimensions"), ("order", "signature"), ("bounds", "bounds"),
    ("axis", "axis"), ("widths", "width"), ("gain", "C=1.25"),
    ("goal_function", "observation"), ("tail_scale", "observation"),
])
def test_metadata_rejects_live_contract_drift(live_env, monkeypatch, mutation, match):
    if mutation == "dimensions":
        speed_type = type(live_env.action_manager.get_term("strike_speed"))
        monkeypatch.setattr(speed_type, "action_dim", property(lambda self: 2))
    elif mutation == "order":
        monkeypatch.setattr(live_env.action_manager, "_term_names", ["joint_stiffness", "joint_position", "strike_speed"])
    elif mutation == "bounds":
        monkeypatch.setattr(live_env.action_manager.get_term("strike_speed").cfg, "bounds", None)
    elif mutation == "axis":
        monkeypatch.setattr(live_env._hammer_first_strike, "_axis", torch.tensor([0., 0., 1.]))
    elif mutation == "widths":
        widths = list(live_env.observation_manager.group_obs_term_dim["actor"])
        widths[-2:] = [(2,), (0,)]
        monkeypatch.setitem(live_env.observation_manager._group_obs_term_dim, "actor", widths)
    elif mutation == "gain":
        monkeypatch.setattr(live_env.action_manager.get_term("joint_stiffness"), "_C", 1.5)
    elif mutation == "goal_function":
        term = live_env.observation_manager.get_term_cfg("actor", "impact_speed_goal")
        monkeypatch.setattr(term, "func", lambda env: torch.zeros(1, 1))
    elif mutation == "tail_scale":
        term = live_env.observation_manager.get_term_cfg("actor", "impact_speed_pending")
        monkeypatch.setattr(term, "scale", 2.0)
    with pytest.raises(ValueError, match=match):
        _contract().build_impact_speed_metadata(live_env, 1.0)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 1), ("action_term_dims", [6, 6]),
    ("speed_bounds_mps", None), ("speed_axis_world", [0, 0, 1]),
    ("observation_names", ["wrong"]), ("observation_normalization", "empirical42"),
    ("variable_impedance", {"C": 1.5}), ("speed_mode", "external"),
    ("speed_cost", {"coefficient": 0.8}),
])
def test_resume_rejects_checkpoint_contract_mismatch(live_env, field, value):
    expected = _contract().build_impact_speed_metadata(live_env, 1.0)
    saved = deepcopy(expected)
    saved[field] = value
    with pytest.raises(ValueError, match="metadata"):
        _contract().validate_impact_speed_metadata(saved, expected)


def test_only_explicit_external_to_policy_transfer_allows_mode_difference(live_env):
    expected = _contract().build_impact_speed_metadata(live_env, 1.0)
    external = {**expected, "speed_mode": "external"}
    _contract().validate_impact_speed_metadata(external, expected, allow_phase_transfer=True)
    with pytest.raises(ValueError, match="metadata"):
        _contract().validate_impact_speed_metadata(expected, external, allow_phase_transfer=True)
    with pytest.raises(ValueError, match="metadata"):
        _contract().validate_impact_speed_metadata({}, expected)


def _storage(*, selected=True):
    actions = torch.zeros(2, 2, 13)
    actions[..., 6:12] = torch.tensor([-2., -1., 0., 0.5, 1., 2.])
    actions[..., 12] = torch.tensor([[-2., 100.], [0., -100.]])
    masks = torch.ones_like(actions)
    masks[..., 12] = torch.tensor([[1., 0.], [1., 0.]]) if selected else 0
    return SimpleNamespace(
        actions=actions, distribution_params=(actions / 2, torch.ones_like(actions), masks),
        num_transitions_per_env=2, num_envs=2, step=0,
    )


def test_selected_speed_uses_only_saved_active_raw_actions():
    telemetry = _contract().summarize_impact_speed_rollout(_storage(), (0.2, 0.8))
    assert telemetry["schema_version"] == "impact_speed_v1"
    assert telemetry["selection_count"] == 2
    assert telemetry["selected_speed_mps"] == pytest.approx({
        "mean": 0.35, "std": 0.15, "minimum": 0.2,
        "p05": 0.215, "median": 0.35, "p95": 0.485, "maximum": 0.5,
    })
    assert telemetry["sampled_action"]["clipped"]["mean"] == [-1., -1., 0., 0.5, 1., 1.]
    assert telemetry["gain_action_indices"] == [6, 7, 8, 9, 10, 11]
    assert len(telemetry["gaussian_exploration_std"]) == 6
    json.dumps(telemetry, allow_nan=False)


def test_zero_selection_has_null_speed_quantiles_but_retains_gain_telemetry():
    telemetry = _contract().summarize_impact_speed_rollout(_storage(selected=False), (0.2, 0.8))
    assert telemetry["selection_count"] == 0
    assert telemetry["selected_speed_mps"] == dict.fromkeys(("mean", "std", "minimum", "p05", "median", "p95", "maximum"))
    assert telemetry["sample_count"] == 4
    assert len(telemetry["sampled_action"]["raw"]["mean"]) == 6
    json.dumps(telemetry, allow_nan=False)


@pytest.mark.parametrize("mutation", ["nonfinite", "mask", "motor_mask", "mask_shape", "negative_std", "cursor"])
def test_bad_rollout_evidence_fails_closed(mutation):
    storage = _storage()
    if mutation == "nonfinite":
        storage.actions[0, 0, 12] = torch.nan
    elif mutation == "mask":
        storage.distribution_params[2][0, 0, 12] = 0.5
    elif mutation == "motor_mask":
        storage.distribution_params[2][0, 0, 2] = 0
    elif mutation == "mask_shape":
        storage.distribution_params = (*storage.distribution_params[:2], torch.ones(1, 13))
    elif mutation == "negative_std":
        storage.distribution_params[1][0, 0, 12] = -1
    else:
        storage.step = 1
    with pytest.raises(ValueError):
        _contract().summarize_impact_speed_rollout(storage, (0.2, 0.8))


def test_legacy_metadata_and_telemetry_still_reject_13d(live_env):
    with pytest.raises(ValueError, match="observation schema"):
        _joint_observation_metadata(live_env)
    with pytest.raises(ValueError, match="dimensions"):
        _vic_controller_metadata(live_env, live_env.scene["robot"], live_env.action_manager.get_term("joint_position"), load_joint_position_contract(_JOINT_POSITION_CONTRACT_PATH))
    storage = _storage()
    storage.distribution_params = storage.distribution_params[:2]
    with pytest.raises(ValueError, match="12D"):
        _summarize_vic_rollout_telemetry(storage, action_terms=("joint_position", "joint_stiffness"), raw_action_clip=1.0)


@pytest.fixture(scope="module")
def live_runner(live_env, tmp_path_factory):
    return HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg(), str(tmp_path_factory.mktemp("speed_runner")))


def test_missing_checkpoint_metadata_is_rejected_before_weights_load(live_runner, tmp_path):
    actor_before = {k: v.clone() for k, v in live_runner.alg.actor.state_dict().items()}
    checkpoint = tmp_path / "missing.pt"
    state = deepcopy(live_runner.alg.save())
    state.update(iter=9, infos={})
    for value in state["actor_state_dict"].values():
        value.zero_()
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="metadata"):
        live_runner.load(str(checkpoint))
    for name, value in live_runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name])


def test_external_qualification_transfer_preserves_weights_but_starts_fresh_optimizer(live_env, tmp_path, monkeypatch):
    speed = live_env.action_manager.get_term("strike_speed")
    with monkeypatch.context() as patch:
        patch.setattr(speed.cfg, "mode", "external")
        patch.setattr(speed.state, "mode", "external")
        source = HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg())
        source.alg.actor.obs_normalizer.update(torch.ones(8, 42))
        source.alg.critic.obs_normalizer.update(torch.ones(8, 42) * 2)
        for parameter in source.alg.actor.parameters():
            parameter.grad = torch.zeros_like(parameter)
        source.alg.optimizer.step()
        source.alg.optimizer.zero_grad()
        assert source.alg.optimizer.state
        for state in source.alg.optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].fill_(0.125)
        assert any(torch.count_nonzero(state["exp_avg"]) for state in source.alg.optimizer.state.values() if "exp_avg" in state)
        with torch.no_grad():
            source.alg.actor.mlp[-1].weight[12].fill_(0.2)
            source.alg.actor.mlp[-1].bias[12].fill_(0.7)
            source.alg.actor.distribution.std_param[12].fill_(0.4)
        payload = source.alg.save()
        payload.update(iter=17, infos={"impact_speed_contract": _contract().build_impact_speed_metadata(live_env, 1.0)})
        path = tmp_path / "qualified.pt"
        torch.save(payload, path)
    values = torch.zeros(2, 42)
    values[:, -1] = torch.tensor([0.0, 1.0])
    observations = TensorDict({"actor": values, "critic": values}, batch_size=[2])
    prior_control = source.alg.actor(observations)[:, :12].detach().clone()
    prior_values = source.alg.critic(observations).detach().clone()
    target = HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg())
    with pytest.raises(ValueError, match="metadata mismatch"):
        target.load(str(path))

    target.load_qualification_checkpoint(str(path))

    assert target.current_learning_iteration == 0
    assert not target.alg.optimizer.state
    torch.testing.assert_close(target.alg.actor(observations)[:, :12], prior_control)
    torch.testing.assert_close(target.alg.critic(observations), prior_values)
    torch.testing.assert_close(target.alg.actor.obs_normalizer.physical.mean, source.alg.actor.obs_normalizer.physical.mean)
    torch.testing.assert_close(target.alg.critic.obs_normalizer.physical.mean, source.alg.critic.obs_normalizer.physical.mean)
    torch.testing.assert_close(target.alg.actor.obs_normalizer(observations["actor"])[:, -1], values[:, -1])
    assert torch.count_nonzero(target.alg.actor.mlp[-1].weight[12]) == 0
    assert target.alg.actor.mlp[-1].bias[12].item() == 0
    assert target.alg.actor.distribution.std_param[12].item() == 1
    target.alg.actor(observations, stochastic_output=True)
    actions = torch.zeros(2, 13)
    actions[1, 12] = 1
    (-target.alg.actor.get_output_log_prob(actions).sum()).backward()
    assert target.alg.actor.mlp[-1].bias.grad[12].abs().item() > 0
    saved = tmp_path / "autonomous.pt"
    target.save(str(saved))
    infos = torch.load(saved, weights_only=False)["infos"]
    assert infos["impact_speed_contract"]["speed_mode"] == "policy"
    assert infos["impact_speed_phase_transition"] == {
        "from_mode": "external", "to_mode": "policy", "selector_initialization": "zero_mean_unit_std",
        "source_checkpoint": str(path.resolve()),
    }
    resumed = HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg())
    resumed.load(str(saved))
    saved_again = tmp_path / "resumed.pt"
    resumed.save(str(saved_again))
    assert torch.load(saved_again, weights_only=False)["infos"]["impact_speed_phase_transition"] == infos["impact_speed_phase_transition"]
    for model_name in ("actor", "critic"):
        invalid = deepcopy(payload)
        invalid[f"{model_name}_state_dict"]["mlp.0.weight"][:, -1] = 0.1
        invalid_path = tmp_path / f"qualified_learned_{model_name}_pending.pt"
        torch.save(invalid, invalid_path)
        candidate = HammerOnPolicyRunner(wrap_impact_speed_env(live_env, 1.0), _runner_cfg())
        with pytest.raises(ValueError, match="pending"):
            candidate.load_qualification_checkpoint(str(invalid_path))


def test_empty_real_checkpoint_still_contains_contract(live_runner, tmp_path, capsys):
    path = tmp_path / "empty.pt"
    old_infos = {
        "impact_speed_rollout_telemetry": {"stale": True},
        "impact_speed_phase_transition": {"stale": True},
        "caller": "retained",
    }
    live_runner.save(str(path), old_infos)
    assert "[WARN] ONNX export failed" not in capsys.readouterr().out
    infos = torch.load(path, weights_only=False)["infos"]
    assert infos["impact_speed_contract"]["schema_version"] == "impact_speed_v1"
    assert "impact_speed_rollout_telemetry" not in infos
    assert "impact_speed_phase_transition" not in infos
    assert infos["caller"] == "retained"
    assert old_infos["impact_speed_rollout_telemetry"] == {"stale": True}


def test_terminal_transition_evidence_survives_reset_without_persistent_outcomes(live_runner):
    env = live_runner.env
    env.reset()
    action = torch.zeros(1, 13)
    action[:, 12] = 1
    env.step(action)
    state = env.unwrapped.action_manager.get_term("strike_speed").state
    torch.testing.assert_close(state.goal_mps, torch.tensor([0.8]))
    env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length - 1
    obs, _, done, extras = env.step(action)
    outcome = extras["impact_speed_outcome"]
    assert done.item() == 1
    assert outcome["charged"].item()
    assert outcome["miss"].item()
    assert outcome["cost"].item() == 1
    assert outcome["goal_mps"].item() == pytest.approx(0.8)
    assert obs["actor"][0, -1].item() == 1
    assert state.goal_mps.item() == pytest.approx(0.5)
    _, _, _, next_extras = env.step(torch.zeros(1, 13))
    assert not next_extras["impact_speed_outcome"]["charged"].item()
    assert outcome["charged"].item()  # The saved snapshot, not reset-cleared state.


def test_real_checkpoint_roundtrip_and_short_export_feedback_rollout(live_runner, tmp_path, capsys):
    """Exercise real CPU rollout/update/save, PT resume and ONNX reference execution."""
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    runner = live_runner
    runner.env.reset()
    runner.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "speed.pt"
    runner.save(str(checkpoint), {"caller_info": "retained"})
    assert "[WARN] ONNX export failed" not in capsys.readouterr().out
    payload = torch.load(checkpoint, weights_only=False)
    contract = payload["infos"]["impact_speed_contract"]
    assert contract["schema_version"] == "impact_speed_v1"
    telemetry = payload["infos"]["impact_speed_rollout_telemetry"]
    assert telemetry["sample_count"] == 24
    assert telemetry["selection_count"] >= 1
    assert payload["infos"]["caller_info"] == "retained"
    actor_before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    # Normal resume constructs a fresh runner. Re-loading a just-trained instance
    # hits an upstream empirical-normalizer inference-tensor limitation.
    runner = HammerOnPolicyRunner(runner.env, _runner_cfg())
    runner.load(str(checkpoint))
    for name, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name])

    runner.export_policy_to_jit(str(tmp_path), "speed-policy.pt")
    jit = torch.jit.load(str(tmp_path / "speed-policy.pt"))
    _, _, onnx_path = runner._get_export_paths(str(checkpoint))
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    props = {item.key: item.value for item in model.metadata_props}
    assert props["schema_version"] == "impact_speed_v1"
    assert json.loads(props["observation_normalization"])["empirical_width"] == 40
    backend = ReferenceEvaluator(model)  # ONNX reference backend, NOT onnxruntime.
    policy = runner.get_inference_policy(device="cpu")
    env = runner.env
    selected_goal = None
    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(3):
            raw = obs["actor"]
            assert raw[0, -1].item() == (1.0 if step == 0 else 0.0)
            eager = policy(obs)
            torch.testing.assert_close(jit(raw), eager)
            onnx_output = torch.from_numpy(backend.run(None, {backend.input_names[0]: raw.numpy()})[0])
            torch.testing.assert_close(onnx_output, eager, rtol=1e-5, atol=1e-6)
            obs, _, _, _ = env.step(onnx_output)
            goal = env.unwrapped.action_manager.get_term("strike_speed").state.goal_mps.clone()
            if selected_goal is None:
                selected_goal = goal
                expected = 0.2 + (eager[:, 12].clamp(-1, 1) + 1) * 0.3
                torch.testing.assert_close(goal, expected)
            else:
                torch.testing.assert_close(goal, selected_goal)
            torch.testing.assert_close(obs["actor"][:, -2], (goal - 0.2) / 0.3 - 1)
        obs, _ = env.reset()
        assert obs["actor"][0, -1].item() == 1.0
        torch.testing.assert_close(jit(obs["actor"]), policy(obs))

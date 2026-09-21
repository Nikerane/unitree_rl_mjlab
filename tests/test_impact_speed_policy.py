"""Contract tests for the experimental 13-action speed policy."""

import math

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage

from src.tasks.hammer.rl.cat_ppo import CatPPO
from src.tasks.hammer.rl.cat_storage import CatRolloutStorage
from src.tasks.hammer.rl.impact_speed_policy import (
    ImpactSpeedGaussian,
    ImpactSpeedModel,
    ImpactSpeedNormalization,
    initialize_selector_for_autonomy,
)


def _obs(pending, width=42):
    values = torch.zeros(len(pending), width)
    values[:, -1] = torch.tensor(pending, dtype=torch.float32)
    return TensorDict({"policy": values}, batch_size=[len(pending)])


def _model(pending=(0.0, 1.0), output_dim=13, stochastic=True):
    obs = _obs(pending)
    distribution_cfg = {"class_name": "src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian"} if stochastic else None
    return ImpactSpeedModel(
        obs, {"actor": ["policy"], "critic": ["policy"]},
        "actor" if stochastic else "critic", output_dim,
        hidden_dims=(8, 8), obs_normalization=True, distribution_cfg=distribution_cfg,
    )


def test_pending_does_not_become_100_after_qualification():
    norm = ImpactSpeedNormalization()
    external = torch.zeros(16, 42)
    norm.update(external)
    autonomous = external[:1].clone()
    autonomous[:, -1] = 1
    assert norm(autonomous)[0, -1].item() == 1
    assert norm(autonomous)[0, -2].item() == 0


def test_normalizer_reload_preserves_physical_statistics_and_raw_tail():
    norm = ImpactSpeedNormalization()
    samples = torch.ones(8, 42)
    norm.update(samples)
    restored = ImpactSpeedNormalization()
    restored.load_state_dict(norm.state_dict())
    probe = torch.zeros(2, 42)
    probe[:, -2:] = torch.tensor([[0.5, 0.0], [-0.5, 1.0]])
    torch.testing.assert_close(restored(probe), norm(probe))
    torch.testing.assert_close(restored(probe)[:, -2:], probe[:, -2:])


def test_inactive_selector_contributes_nothing_to_log_prob_entropy_or_kl():
    dist = ImpactSpeedGaussian(13)
    dist.activity_mask = torch.tensor([[1.0] * 12 + [0.0]])
    dist.update(torch.zeros(1, 13))
    action = torch.zeros(1, 13)
    action[0, -1] = 100.0
    assert dist.log_prob(action).item() == pytest.approx(-12 * 0.5 * math.log(2 * math.pi))
    assert dist.entropy.item() == pytest.approx(12 * 0.5 * math.log(2 * math.pi * math.e))
    old = dist.params
    newer = (old[0].clone(), old[1].clone(), old[2].clone())
    newer[0][0, -1] = 100.0
    assert dist.kl_divergence(old, newer).item() == pytest.approx(0.0)


def test_pending_selector_is_active_and_its_kl_has_correct_direction():
    dist = ImpactSpeedGaussian(13)
    dist.activity_mask = torch.ones(1, 13)
    dist.update(torch.zeros(1, 13))
    old = dist.params
    action = torch.zeros(1, 13)
    action[0, -1] = 2
    assert dist.log_prob(action).item() == pytest.approx(-13 * 0.5 * math.log(2 * math.pi) - 2)
    new_mean = old[0].clone()
    new_mean[0, -1] = 2
    new_std = old[1].clone()
    new_std[..., -1] = 2.0
    newer = (new_mean, new_std, old[2].clone())
    assert dist.kl_divergence(old, newer).item() == pytest.approx(math.log(2) + 0.125)


def test_kl_rejects_disagreeing_recorded_masks():
    dist = ImpactSpeedGaussian(13)
    mean = torch.zeros(1, 13)
    std = torch.ones(13)
    old = (mean, std, torch.ones(1, 13))
    new = (mean, std, torch.tensor([[1.0] * 12 + [0.0]]))
    with pytest.raises(ValueError, match="masks disagree"):
        dist.kl_divergence(old, new)


def test_kl_rejects_nonbinary_recorded_masks_even_when_they_agree():
    dist = ImpactSpeedGaussian(13)
    mean = torch.zeros(1, 13)
    std = torch.ones(13)
    invalid = torch.tensor([[1.0] * 12 + [0.5]])
    with pytest.raises(ValueError, match="binary"):
        dist.kl_divergence((mean, std, invalid), (mean, std, invalid.clone()))


def test_model_masks_mixed_minibatch_from_raw_observations():
    actor = _model()
    obs = _obs([0.0, 1.0, 0.0])
    sampled = actor(obs, stochastic_output=True)
    assert sampled.shape == (3, 13)
    torch.testing.assert_close(actor.output_distribution_params[2][:, -1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.all(actor.output_distribution_params[2][:, :12] == 1)


def test_rollout_storage_roundtrips_masks_for_mixed_minibatch_and_no_selections():
    actor = _model()
    obs = _obs([0.0, 1.0, 0.0, 1.0])
    storage = RolloutStorage("rl", 4, 1, obs, [13], device="cpu")
    actions = actor(obs, stochastic_output=True)
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.actions = actions
    transition.rewards = torch.zeros(4)
    transition.dones = torch.zeros(4)
    transition.values = torch.zeros(4, 1)
    transition.actions_log_prob = actor.get_output_log_prob(actions)
    transition.distribution_params = actor.output_distribution_params
    storage.add_transition(transition)
    batch = next(storage.mini_batch_generator(1, 1))
    actor(batch.observations, stochastic_output=True)
    torch.testing.assert_close(actor.output_distribution_params[2], batch.old_distribution_params[2])
    assert actor.get_kl_divergence(batch.old_distribution_params, actor.output_distribution_params).shape == (4,)
    assert len(storage.distribution_params) == 3
    no_selection = _obs([0.0] * 4)
    no_selection_actions = actor(no_selection, stochastic_output=True)
    transition.observations = no_selection
    transition.actions = no_selection_actions
    transition.actions_log_prob = actor.get_output_log_prob(no_selection_actions)
    transition.distribution_params = actor.output_distribution_params
    empty_selection_storage = RolloutStorage("rl", 4, 1, no_selection, [13], device="cpu")
    empty_selection_storage.add_transition(transition)
    assert empty_selection_storage.distribution_params[2][:, :, -1].sum().item() == 0


def test_invalid_pending_flags_are_rejected_before_normalization():
    actor = _model()
    for invalid in (-0.1, 0.5, 2.0, float("nan")):
        with pytest.raises(ValueError, match="pending"):
            actor(_obs([invalid]), stochastic_output=True)


def test_clipping_does_not_replace_raw_action_in_likelihood():
    dist = ImpactSpeedGaussian(13)
    dist.activity_mask = torch.ones(1, 13)
    dist.update(torch.zeros(1, 13))
    raw = torch.zeros(1, 13)
    raw[0, -1] = 2.0
    clipped = raw.clamp(-1, 1)
    assert dist.log_prob(raw).item() == pytest.approx(dist.log_prob(clipped).item() - 1.5)


def test_inactive_selector_has_zero_gradient_but_active_selector_learns():
    actor = _model()
    obs = _obs([0.0, 1.0])
    actor(obs, stochastic_output=True)
    actions = torch.zeros(2, 13)
    actions[:, -1] = 2.0
    loss = -actor.get_output_log_prob(actions).sum()
    loss.backward()
    selector_bias_gradient = actor.mlp[-1].bias.grad[-1]
    assert selector_bias_gradient.abs().item() > 0
    actor.zero_grad()
    actor(_obs([0.0, 0.0]), stochastic_output=True)
    (-actor.get_output_log_prob(actions).sum()).backward()
    assert actor.mlp[-1].bias.grad[-1].item() == 0


def test_pending_initialization_has_no_actor_or_critic_output_jump():
    actor = _model()
    critic = _model(output_dim=1, stochastic=False)
    obs = _obs([0.0, 1.0])
    torch.testing.assert_close(actor(obs)[0, :12], actor(obs)[1, :12])
    torch.testing.assert_close(critic(obs)[0], critic(obs)[1])
    assert actor.mlp[0].weight[:, -1].requires_grad
    assert critic.mlp[0].weight[:, -1].requires_grad


def test_selector_transfer_changes_only_selector_row_and_std():
    actor = _model()
    critic = _model(output_dim=1, stochastic=False)
    actor.obs_normalizer.update(torch.ones(8, 42))
    critic.obs_normalizer.update(torch.ones(8, 42) * 2)
    with torch.no_grad():
        actor.mlp[-1].weight[12].fill_(0.25)
        actor.mlp[-1].bias[12].fill_(0.75)
        actor.distribution.std_param[12].fill_(0.4)
    observations = _obs([0.0, 1.0])
    before_mean = actor(observations).detach().clone()
    before_value = critic(observations).detach().clone()
    before_actor = {key: value.clone() for key, value in actor.state_dict().items()}
    before_critic = {key: value.clone() for key, value in critic.state_dict().items()}
    final_index = max(index for index, layer in enumerate(actor.mlp) if isinstance(layer, torch.nn.Linear))

    initialize_selector_for_autonomy(actor)

    torch.testing.assert_close(actor(observations)[:, :12], before_mean[:, :12])
    torch.testing.assert_close(critic(observations), before_value)
    for key, value in actor.state_dict().items():
        if key == f"mlp.{final_index}.weight":
            torch.testing.assert_close(value[:12], before_actor[key][:12])
        elif key == f"mlp.{final_index}.bias":
            torch.testing.assert_close(value[:12], before_actor[key][:12])
        elif key == "distribution.std_param":
            torch.testing.assert_close(value[:12], before_actor[key][:12])
        else:
            torch.testing.assert_close(value, before_actor[key])
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, before_critic[key])
    assert torch.count_nonzero(actor.mlp[-1].weight[12]) == 0
    assert actor.mlp[-1].bias[12].item() == 0
    assert actor.distribution.std_param[12].item() == 1
    assert actor(observations)[0, 12].item() == pytest.approx(0.0)
    actor(observations, stochastic_output=True)
    actions = torch.zeros(2, 13)
    actions[1, 12] = 1.0
    (-actor.get_output_log_prob(actions).sum()).backward()
    assert actor.mlp[-1].bias.grad[12].abs().item() > 0


def test_selector_transfer_rejects_learned_pending_column_without_mutation():
    actor = _model()
    with torch.no_grad():
        actor.mlp[0].weight[0, -1] = 0.1
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    with pytest.raises(ValueError, match="pending"):
        initialize_selector_for_autonomy(actor)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_selector_transfer_rejects_non_scalar_or_invalid_std_without_mutation():
    actor = _model()
    actor.distribution.std_type = "log"
    with pytest.raises(ValueError, match="scalar"):
        initialize_selector_for_autonomy(actor)
    actor.distribution.std_type = "scalar"
    for invalid in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="initial_std"):
            initialize_selector_for_autonomy(actor, initial_std=invalid)


def test_delayed_cat_return_updates_only_a_selected_first_step_selector():
    def run(first_pending):
        torch.manual_seed(27)
        features = torch.zeros(2, 42)
        features[:, -1] = float(first_pending)
        obs = TensorDict({"actor": features.clone(), "critic": features.clone()}, batch_size=[2])
        groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = ImpactSpeedModel(
            obs, groups, "actor", 13, hidden_dims=(8,), obs_normalization=True,
            distribution_cfg={"class_name": "src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian"},
        )
        critic = ImpactSpeedModel(obs, groups, "critic", 1, hidden_dims=(8,), obs_normalization=True)
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
        storage = CatRolloutStorage("rl", 2, 3, obs, [13], "cpu")
        algorithm = CatPPO(
            actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
            normalize_advantage_per_mini_batch=True,
        )
        with torch.no_grad():
            for step, delta in enumerate((0.2, 0.3, 0.4)):
                raw_action = algorithm.act(obs)
                next_features = features.clone()
                next_features[:, -2] = raw_action[:, -1].clamp(-1, 1) if step == 0 else obs["actor"][:, -2]
                next_features[:, -1] = 0
                next_obs = TensorDict({"actor": next_features.clone(), "critic": next_features.clone()}, batch_size=[2])
                algorithm.process_env_step(
                    next_obs, torch.full((2,), -0.5 if step == 2 else 0.0),
                    torch.full((2,), step == 2),
                    {"cat_delta": torch.full((2,), delta), "cat_r_pos": torch.zeros(2)},
                )
                obs = next_obs
            algorithm.compute_returns(obs)
        first_mask = storage.distribution_params[2][:, :, 12].clone()
        first_return = storage.returns[0, :, 0].clone()
        selector_before = actor.mlp[-1].weight[12].detach().clone()
        losses = algorithm.update()
        selector_after = actor.mlp[-1].weight[12].detach().clone()
        return first_mask, first_return, selector_before, selector_after, losses

    selected_mask, selected_return, before, after, losses = run(True)
    torch.testing.assert_close(selected_mask, torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    torch.testing.assert_close(selected_return, torch.full((2,), -0.5 * (0.99 * 0.95) ** 2 * 0.8 * 0.7))
    assert torch.count_nonzero(after - before).item() > 0
    assert all(math.isfinite(value) for value in losses.values())
    inactive_mask, _, inactive_before, inactive_after, _ = run(False)
    assert torch.count_nonzero(inactive_mask).item() == 0
    torch.testing.assert_close(inactive_after, inactive_before)


def test_inherited_deterministic_exports_keep_tail_fixed_scale():
    actor = _model()
    actor.obs_normalizer.update(torch.zeros(16, 42))
    obs = _obs([1.0])
    expected = actor(obs)
    torch.testing.assert_close(actor.as_jit()(obs["policy"]), expected)
    torch.testing.assert_close(actor.as_onnx(False)(obs["policy"]), expected)
    torch.testing.assert_close(torch.jit.script(actor.as_jit())(obs["policy"]), expected)


def test_actor_and_critic_require_exact_42_observations():
    for stochastic, output_dim in ((True, 13), (False, 1)):
        with pytest.raises(ValueError, match="42"):
            ImpactSpeedModel(
                _obs([0.0], width=41), {"actor": ["policy"], "critic": ["policy"]},
                "actor" if stochastic else "critic", output_dim,
                obs_normalization=True,
                distribution_cfg={"class_name": "src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedGaussian"} if stochastic else None,
            )

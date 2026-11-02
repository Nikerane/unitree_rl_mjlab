"""One accepted event's credit survives separation until a real reset boundary."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg

from tests.test_spline_speed_curriculum import environment


def credit_module():
    name = "src.tasks.hammer.mdp.spline_boundary_credit"
    assert importlib.util.find_spec(name) is not None, "boundary credit reward is missing"
    return importlib.import_module(name)


def setup_credit(fraction, batch=2):
    module = credit_module()
    env, head, tracker, params = environment(batch)
    env.reset_buf = torch.zeros(batch, dtype=torch.bool)
    env.reset_terminated = torch.zeros(batch, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(batch, dtype=torch.bool)
    tracker.elapsed_since_contact_s = torch.zeros(batch)
    params["onset_fraction"] = fraction
    cfg = RewardTermCfg(func=module.BoundarySplineStrikeReward, weight=1., params=params)
    term = cfg.func(cfg, env)
    return env, head, tracker, params, term


def accepted_onset(tracker):
    tracker.started[:] = True
    tracker.onset_head_w[:] = torch.tensor([0., 0., .1])


@pytest.mark.parametrize("fraction,want_onset,want_boundary", [(.2, .2, .8), (0., 0., 1.)])
def test_event_unit_is_conserved_through_separation_and_actual_boundary(fraction, want_onset, want_boundary):
    env, head, tracker, params, term = setup_credit(fraction)
    assert term(env, **params).eq(0).all()
    accepted_onset(tracker)
    onset = term(env, **params) * env.step_dt
    torch.testing.assert_close(onset, torch.full((2,), want_onset))
    # Rebound, contact loss and elapsed nominal 50 ms cannot drain the balance.
    head[:] = torch.tensor([.1, 0., .2])
    tracker.elapsed_since_contact_s[:] = .060
    for _ in range(3):
        assert term(env, **params).eq(0).all()
    env.reset_buf[:] = True
    boundary = term(env, **params) * env.step_dt
    torch.testing.assert_close(boundary, torch.full((2,), want_boundary))
    torch.testing.assert_close(onset + boundary, torch.ones(2))
    assert term(env, **params).eq(0).all()


@pytest.mark.parametrize("fraction", [.2, 0.])
@pytest.mark.parametrize("kind", ["failure", "timeout"])
def test_eligible_early_failure_or_timeout_pays_without_changing_done_flags(fraction, kind):
    env, _, tracker, params, term = setup_credit(fraction)
    accepted_onset(tracker)
    onset = term(env, **params) * env.step_dt
    tracker.elapsed_since_contact_s[:] = .005
    env.reset_buf[0] = True
    flags = env.reset_time_outs if kind == "timeout" else env.reset_terminated
    flags[0] = True
    terminated, timeouts = env.reset_terminated.clone(), env.reset_time_outs.clone()
    boundary = term(env, **params) * env.step_dt
    torch.testing.assert_close(onset + boundary, torch.tensor([1., fraction]))
    torch.testing.assert_close(env.reset_terminated, terminated)
    torch.testing.assert_close(env.reset_time_outs, timeouts)


@pytest.mark.parametrize("fraction", [.2, 0.])
def test_onset_on_same_transition_as_real_boundary_pays_total_once(fraction):
    env, _, tracker, params, term = setup_credit(fraction)
    accepted_onset(tracker)
    env.reset_buf[:] = True
    torch.testing.assert_close(term(env, **params) * env.step_dt, torch.ones(2))
    assert term(env, **params).eq(0).all()


@pytest.mark.parametrize("kind", ["no_contact", "off_spline", "off_endpoint", "upward", "stationary"])
def test_ineligible_first_contact_cannot_earn_onset_or_later_recontact_credit(kind):
    env, _, tracker, params, term = setup_credit(.2)
    accepted_onset(tracker)
    if kind == "no_contact":
        tracker.started[:] = False
    elif kind == "off_spline":
        tracker.onset_head_w[:, 0] = .02
    elif kind == "off_endpoint":
        tracker.onset_head_w[:, 2] = .15
    elif kind == "upward":
        tracker.v_precontact[:] = -1.
    else:
        tracker.v_precontact[:] = 0.
    assert term(env, **params).eq(0).all()
    # Even a later apparently valid contact must not rehabilitate a rejected first one.
    if kind != "no_contact":
        accepted_onset(tracker)
        tracker.v_precontact[:] = 1.
    env.reset_buf[:] = True
    assert term(env, **params).eq(0).all()


def test_asynchronous_resets_clear_only_reset_worlds_including_censored_balance():
    env, _, tracker, params, term = setup_credit(.2, batch=3)
    accepted_onset(tracker)
    term(env, **params)
    # Evaluator censoring is a caller reset, not a native step boundary: no payment.
    term.reset(torch.tensor([0]))
    tracker.started[0] = False
    env.reset_buf[1] = True
    torch.testing.assert_close(term(env, **params) * env.step_dt, torch.tensor([0., .8, 0.]))
    term.reset(torch.tensor([1]))
    env.reset_buf.zero_()
    tracker.started[1] = False
    tracker.started[0] = True
    torch.testing.assert_close(term(env, **params) * env.step_dt, torch.tensor([.2, 0., 0.]))
    env.reset_buf[2] = True
    torch.testing.assert_close(term(env, **params) * env.step_dt, torch.tensor([0., 0., .8]))
    term.reset(None)
    tracker.started.zero_()
    assert term(env, **params).eq(0).all()
    assert torch.isfinite(term.unpaid_credit).all()
    assert term.unpaid_credit.eq(0).all()


@pytest.mark.parametrize("fraction", [-.1, 1., float("nan"), float("inf")])
def test_unapproved_or_nonfinite_splits_are_rejected(fraction):
    with pytest.raises(ValueError, match="onset_fraction"):
        setup_credit(fraction)


@pytest.mark.parametrize("play", [False, True])
def test_four_arms_preserve_n_interface_and_only_change_intended_rewards(play):
    module = credit_module()
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from src.tasks.hammer.mdp.spline_learning_variants import TASKS
    from tests.test_spline_speed_variants import config_value
    baseline = load_env_cfg(TASKS["N"], play=play)
    baseline_rl = load_rl_cfg(TASKS["N"])
    for arm in ("C", "W", "S", "D"):
        cfg = load_env_cfg(TASKS[arm], play=play)
        assert config_value(load_rl_cfg(TASKS[arm])) == config_value(baseline_rl)
        if arm == "W":
            assert cfg.rewards["spline_path_cost"].weight == -1.
            cfg.rewards["spline_path_cost"].weight = -.2
        elif arm in ("S", "D"):
            reward = cfg.rewards["accepted_spline_strike"]
            assert reward.func is module.BoundarySplineStrikeReward
            assert reward.params["onset_fraction"] == (.2 if arm == "S" else 0.)
            cfg.rewards["accepted_spline_strike"] = deepcopy(baseline.rewards["accepted_spline_strike"])
        assert config_value(cfg) == config_value(baseline)


@pytest.mark.integration
def test_live_arm_contracts_reload_only_own_rewards_and_keep_n_observations(tmp_path):
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from src.tasks.hammer.mdp.sampled_pd500 import configure_sampled_pd500, install_sampled_pd500
    from src.tasks.hammer.mdp.spline_learning_variants import TASKS
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    checkpoints, contracts = {}, {}
    for arm in ("C", "W", "S", "D"):
        cfg = load_env_cfg(TASKS[arm], play=True)
        cfg.scene.num_envs = 2
        configure_sampled_pd500(cfg, gain_mode="variable")
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            install_sampled_pd500(env, gain_mode="variable")
            contract = build_impact_speed_metadata(env, 1.)
            contracts[arm] = contract
            assert contract["observation_widths"] == {"actor": 43, "critic": 43}
            assert contract["spline_learning_variant"]["variant"] == "N"
            if arm in ("S", "D"):
                timing = contract["reward_profile"]["credit_timing"]
                assert timing["onset_fraction"] == (.2 if arm == "S" else 0.)
                assert timing["boundary"] == "native_reset_buf_after_final_physics_before_reset"
                assert timing["observability"] == "unchanged_N_partial_observability"
            else:
                assert "credit_timing" not in contract["reward_profile"]
            rl = asdict(load_rl_cfg(TASKS[arm]))
            rl["actor"]["hidden_dims"] = rl["critic"]["hidden_dims"] = [16, 8]
            rl["upload_model"] = False
            runner = HammerOnPolicyRunner(wrap_impact_speed_env(env, 1.), rl)
            path = tmp_path / f"{arm}.pt"
            runner.save(str(path))
            checkpoints[arm] = path
            runner.load(str(path))
            for other, checkpoint in checkpoints.items():
                if other != arm:
                    with pytest.raises(ValueError, match="metadata mismatch"):
                        runner.load(str(checkpoint))
        finally:
            env.close()
    for arm in ("W", "S", "D"):
        # The complete execution contract changes only its reward-profile field.
        changed = {key for key in contracts["C"] if contracts["C"][key] != contracts[arm][key]}
        assert changed == {"reward_profile"}


@pytest.mark.integration
@pytest.mark.parametrize("arm,balance", [("S", .8), ("D", 1.)])
def test_native_timeout_pays_before_async_reset_with_final_substep_cat(arm, balance, monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY
    from src.tasks.hammer.mdp.sampled_pd500 import configure_sampled_pd500, install_sampled_pd500
    from src.tasks.hammer.mdp.spline_learning_variants import TASKS
    from src.tasks.hammer.rl.cat_ppo import CatPPO
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env

    cfg = load_env_cfg(TASKS[arm], play=True)
    cfg.scene.num_envs = 2
    configure_sampled_pd500(cfg, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        install_sampled_pd500(env, gain_mode="variable")
        wrapped = wrap_impact_speed_env(env, 1.)
        with torch.inference_mode():
            wrapped.reset()
            tracker = env._hammer_first_strike
            reward_cfg = env.reward_manager.get_term_cfg("accepted_spline_strike")
            term = reward_cfg.func
            # Stage a valid onset using the actual immutable event snapshot and
            # reward manager, with world 0 timing out on its next native step.
            tracker._state[:] = 2
            tracker._onset_head_w.copy_(env._strike_reference._nail)
            tracker._v_precontact.fill_(1.)
            env.reset_buf = env.termination_manager.compute()
            env.reward_manager.compute(env.step_dt)
            torch.testing.assert_close(term.unpaid_credit, torch.full((2,), balance))
            env.episode_length_buf[0] = env.max_episode_length - 1
            # Inject a test peak only at the final physics sample. Everything
            # from termination computation through reward/CaT/reset is native.
            original = env.metrics_manager.compute_substep
            calls = 0

            def final_peak():
                nonlocal calls
                original()
                calls += 1
                if calls == env.cfg.decimation:
                    env._hammer_substep_peak_qv.peak_qv_joint[0, 1] = 5.

            monkeypatch.setattr(env.metrics_manager, "compute_substep", final_peak)
            obs, rewards, done, extras = wrapped.step(torch.zeros(2, 13))
            assert done.tolist() == [1, 0]
            assert extras["time_outs"].tolist() == [True, False]
            assert env.reset_terminated.tolist() == [False, False]
            index = env.reward_manager.active_terms.index("accepted_spline_strike")
            paid = env.reward_manager._step_reward[:, index] * env.step_dt
            torch.testing.assert_close(paid, torch.tensor([balance, 0.]))
            assert extras[CAT_R_POS_KEY][0] >= balance - 1e-6
            assert extras[CAT_DELTA_KEY][0] > 0.
            scaled = CatPPO._cat_scale_reward(rewards, extras[CAT_DELTA_KEY], extras[CAT_R_POS_KEY])
            assert scaled[0] < rewards[0]
            torch.testing.assert_close(term.unpaid_credit, torch.tensor([0., balance]))
            assert obs["actor"][0, 40] == 0.
            assert obs["actor"][1, 40] > 0.
    finally:
        env.close()

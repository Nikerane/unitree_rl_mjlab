"""Bounded evaluator: paired evidence and transition-safe episode accounting."""

import importlib
from types import SimpleNamespace

import pytest
import torch


def evaluator():
    return importlib.import_module("scripts.evaluate_impact_speed")


def responses(speeds, hits=None):
    return {"episode_id": list(range(len(speeds))), "reset_hash": ["reset"] * len(speeds),
            "hit": hits or [True] * len(speeds), "speed": speeds}


def test_ignored_goal_does_not_pass_responsiveness():
    report = evaluator().compare_goal_responses(responses([.5] * 4), responses([.5] * 4), seed=7)
    assert report["paired_speed_difference"] == 0
    assert report["responsive"] is False


def test_more_hits_alone_cannot_pass_and_misses_are_unconditional():
    report = evaluator().compare_goal_responses(
        responses([.5, None, None, .5], [True, False, False, True]), responses([.5] * 4), seed=7)
    assert report["responsive"] is False
    assert report["common_valid_pairs"] == 2
    assert report["coverage"]["low"] == {"episodes": 4, "hits": 2, "available": 2, "misses": 2}
    assert report["coverage"]["high"]["hits"] == 4


def test_paired_bootstrap_positive_and_deterministic():
    low, high = responses([.2, .3, .4, .5]), responses([.4, .5, .6, .7])
    report = evaluator().compare_goal_responses(low, high, seed=7)
    assert report == evaluator().compare_goal_responses(low, high, seed=7)
    assert report["paired_speed_difference"] == pytest.approx(.2)
    assert report["ci95"][0] > 0
    assert report["responsive"] is True


@pytest.mark.parametrize("key,value", [("reset_hash", ["other"] * 4), ("episode_id", [3, 2, 1, 0])])
def test_mismatched_pairs_fail_closed(key, value):
    low, high = responses([.2] * 4), responses([.8] * 4)
    high[key] = value
    with pytest.raises(ValueError, match="paired"):
        evaluator().compare_goal_responses(low, high, seed=7)


def test_no_valid_pairs_not_a_zero_speed_achievement():
    rows = responses([None] * 3, [False] * 3)
    report = evaluator().compare_goal_responses(rows, rows, seed=7)
    assert report["paired_speed_difference"] is None
    assert report["ci95"] is None
    assert report["responsive"] is False


def transition(charged=(False, False), done=(False, False)):
    n = len(charged)
    outcome = {key: torch.zeros(n, dtype=torch.bool) for key in
               ("charged", "finalized", "valid_measured", "started", "miss", "incomplete", "done", "productive")}
    outcome.update(charged=torch.tensor(charged), done=torch.tensor(done),
                   goal_mps=torch.full((n,), .8), measured_mps=torch.full((n,), float("nan")),
                   cost=torch.zeros(n))
    physical = {"path_error_m": torch.full((n,), .003), "joint_target_error_rad": torch.full((n,), .2),
                "first_contact_time_s": torch.full((n,), .1), "started": torch.zeros(n, dtype=torch.bool),
                "delivered_impulse_ns": torch.full((n,), 2.), "joint_impulse_nms": torch.ones(n, 6),
                "cat_delta": torch.full((n,), .1), "impulse_exceeded": torch.zeros(n, 6, dtype=torch.bool),
                "kp": torch.full((n, 6), 40.), "kd": torch.full((n, 6), 4.),
                "timeout": torch.zeros(n, dtype=torch.bool)}
    return outcome, physical


def test_staggered_terminal_events_and_early_finalization_survive_reset():
    acc = evaluator().EpisodeAccumulator(2, gamma=.5, coefficient=.4)
    raw = torch.zeros(2, 13)
    out, physical = transition((True, False))
    for key in ("finalized", "valid_measured", "started", "productive"):
        out[key][0] = True
    out["measured_mps"][0] = .5
    out["cost"][0] = .25
    acc.update(out, physical, raw, torch.ones(2))
    for _ in range(5):
        out, physical = transition()
        acc.update(out, physical, raw, torch.zeros(2))
    out, physical = transition((False, True), (False, True))
    out["miss"][1] = True
    out["cost"][1] = 1
    physical["timeout"][1] = True
    acc.update(out, physical, raw, torch.zeros(2))
    out, physical = transition(done=(True, False))
    physical["joint_impulse_nms"][0] = 7
    acc.update(out, physical, raw, torch.tensor([0., 1.]))
    row = next(x for x in acc.rows if x["world"] == 0)
    assert row["episode"] == 0 and row["measured_mps"] == .5
    assert row["charge_count"] == 1 and row["discounted_cost"] == pytest.approx(.1)
    assert row["tracking_rms_mm"] == pytest.approx(3.)
    assert row["joint_impulse_peak_nms"] == [7.] * 6
    assert acc.rows[0]["timeout"] is True
    assert acc.rows[0]["discounted_cost"] == pytest.approx(.4 * .5**6)
    assert acc.episodes == [1, 1]


def test_double_charge_rejected_and_no_selection_is_null():
    acc = evaluator().EpisodeAccumulator(2, gamma=.99, coefficient=.4)
    out, physical = transition((True, False))
    out["miss"][0] = True
    acc.update(out, physical, torch.zeros(2, 13), torch.zeros(2))
    with pytest.raises(ValueError, match="twice"):
        acc.update(out, physical, torch.zeros(2, 13), torch.zeros(2))
    row = acc.finish()[1]
    assert row["selection_count"] == 0 and row["selected_raw"] is None
    assert row["measured_mps"] is None


def test_route_strata_report_requested_and_achieved_speed_separately():
    rows = []
    for world, sign, goal, measured in (
        (0, -1, 1.6, 1.1), (1, -1, 1.8, None), (2, 0, 1.9, 1.3), (3, 1, 2.1, 1.5),
    ):
        rows.append({
            "world": world, "episode": 0, "route_sign": sign,
            "hit": measured is not None, "miss": measured is None,
            "selection_count": 1, "goal_mps": goal, "selected_goal_mps": goal - .1,
            "measured_mps": measured,
            "measurement": {
                "common_window": {"window_complete": measured is not None,
                                   "axial_positive_ns": .2 if measured is not None else 0.},
                "loads": {"episode": {"exceeded": [False, False, False, False, False, False]}},
            },
        })
    report = evaluator().summarize_route_strata(rows, expected_route_signs=(-1, 0, 1))
    assert report["route_counts"] == {"-1": 2, "0": 1, "1": 1}
    left = report["routes"]["-1"]
    assert left["attempts"] == 2 and left["hits"] == 1 and left["misses"] == 1
    assert left["requested_goal_mps"]["mean"] == pytest.approx(1.7)
    assert left["selected_goal_mps"]["mean"] == pytest.approx(1.6)
    assert left["achieved_speed_mps"] == {"mean": pytest.approx(1.1), "count": 1}
    assert left["complete_common_windows"] == 1


def test_route_strata_reject_missing_route_identity_or_uncovered_start():
    row = {"world": 0, "episode": 0, "route_sign": -1, "selection_count": 0,
           "goal_mps": 1.5, "measured_mps": None, "hit": False, "miss": True,
           "measurement": {"common_window": {"window_complete": False,
                                                "axial_positive_ns": 0.},
                           "loads": {"episode": {"exceeded": [False] * 6}}}}
    with pytest.raises(ValueError, match="route"):
        evaluator().summarize_route_strata([row], expected_route_signs=(-1, 0, 1))
    row["route_sign"] = None
    with pytest.raises(ValueError, match="route"):
        evaluator().summarize_route_strata([row], expected_route_signs=(-1,))


@pytest.mark.parametrize(("mode", "expected"), [("external", 0), ("policy", 1)])
def test_first_episode_selector_count_matches_speed_mode(mode, expected):
    rows = [{"world": world, "episode": 0, "selection_count": expected}
            for world in range(3)]
    report = evaluator().validate_selector_counts(rows, mode=mode)
    assert report == {"first_episode_count": 3, "expected_per_episode": expected,
                      "episodes_with_expected_count": 3, "total_selections": 3 * expected}
    rows[1]["selection_count"] = 1 - expected
    with pytest.raises(ValueError, match="selector"):
        evaluator().validate_selector_counts(rows, mode=mode)


def test_accumulator_keeps_each_episode_route_from_its_pre_reset_snapshot():
    acc = evaluator().EpisodeAccumulator(1, gamma=.99, coefficient=.4)
    raw = torch.zeros(1, 13)
    out, physical = transition((False,), (True,))
    physical["route_sign"] = torch.tensor([-1])
    acc.update(out, physical, raw, torch.zeros(1))
    out, physical = transition((False,), (False,))
    physical["route_sign"] = torch.tensor([1])
    acc.update(out, physical, raw, torch.zeros(1))
    assert [row["route_sign"] for row in acc.finish()] == [-1, 1]


def test_accumulator_rejects_non_route_sign_values():
    acc = evaluator().EpisodeAccumulator(1, gamma=.99, coefficient=.4)
    out, physical = transition((False,))
    physical["route_sign"] = torch.tensor([.5])
    with pytest.raises(ValueError, match="route sign"):
        acc.update(out, physical, torch.zeros(1, 13), torch.zeros(1))


def test_short_three_route_policy_mode_steps_latch_speed_once_per_episode():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env

    cfg = _impact_speed_env_cfg(play=False, route_mode="three_starts")
    cfg.scene.num_envs = 1
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        wrapped = wrap_impact_speed_env(env, 1.)
        wrapped.reset()
        speed = env.action_manager.get_term("strike_speed").state
        assert torch.equal(speed.pending, torch.ones(1))
        pre_reset_signs = env._strike_reference.route_signs().clone()
        action = torch.zeros(1, 13)
        action[:, 12] = 1.
        env.episode_length_buf[:] = env.max_episode_length - 1
        with evaluator().PreResetSnapshot(env) as capture:
            _, _, done, extras = wrapped.step(action)
        assert done.item()
        torch.testing.assert_close(capture.latest["route_sign"], pre_reset_signs)
        torch.testing.assert_close(extras["impact_speed_outcome"]["goal_mps"], torch.tensor([.8]))
        assert torch.equal(speed.pending, torch.ones(1))

        action[:, 12] = -1.
        wrapped.step(action)
        torch.testing.assert_close(speed.goal_mps, torch.tensor([.2]))
        assert torch.equal(speed.pending, torch.zeros(1))
        action[:, 12] = 1.
        wrapped.step(action)
        torch.testing.assert_close(speed.goal_mps, torch.tensor([.2]))
        assert torch.equal(speed.pending, torch.zeros(1))
        assert cfg.events["sample_strike_route_signs"].params.get("fixed_route_sign") is None
        assert int(env._strike_reference.route_signs()[0]) in {-1, 0, 1}
    finally:
        env.close()


def test_snapshot_is_after_metrics_but_before_reset_and_is_restored():
    env = SimpleNamespace(value=torch.tensor([0.]))
    def compute():
        env.value.fill_(7)
    env.metrics_manager = SimpleNamespace(compute=compute)
    with evaluator().PreResetSnapshot(env, reader=lambda e: {"value": e.value}) as capture:
        env.metrics_manager.compute()
        env.value.zero_()  # native reset
        assert capture.latest["value"].item() == 7
    assert env.metrics_manager.compute is compute


def test_intervention_changes_only_goal_and_refreshes_both_observations():
    from src.tasks.hammer.mdp.impact_speed import ImpactSpeedState
    state = ImpactSpeedState(1, "cpu", (.2, .8), "policy")
    state.select(torch.zeros(1, 1))
    tracker = SimpleNamespace(started=torch.tensor([False]))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(state=state)),
                          _hammer_first_strike=tracker)
    # Real observation adapter behavior, with raw goal/pending in both groups.
    env.observation_manager = SimpleNamespace(compute_group=lambda group, **kw: state.observation())
    observations = evaluator().override_latched_goal(env, .8)
    assert observations["actor"].tolist() == [[1., 0.]]
    assert observations["critic"].tolist() == [[1., 0.]]
    assert not state.cost_consumed.any()
    tracker.started.fill_(True)
    with pytest.raises(ValueError, match="contact"):
        evaluator().override_latched_goal(env, .2)


def test_cli_requires_checkpoint_except_smoke_and_checks_goals(tmp_path):
    base = ["--task", "task", "--speed-bounds", ".2", ".8", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "1", "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "policy"])
    assert evaluator().parse_args(base + ["--mode", "smoke"]).checkpoint is None
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "external", "--checkpoint", "p.pt", "--goals", ".1", ".5", ".8"])
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "smoke", "--sampled-pd500-gain-mode", "variable"])
    args = evaluator().parse_args(base + ["--mode", "smoke", "--sampled-pd500", "--sampled-pd500-gain-mode", "variable"])
    assert args.sampled_pd500 and args.sampled_pd500_gain_mode == "variable"
    assert args.measurement_protocol == "native"
    assert evaluator().parse_args(base + ["--mode", "smoke", "--measurement-protocol", "common-window"]).measurement_protocol == "common-window"


def test_sampled_physical_snapshot_reports_applied_calibrated_gains():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.sampled_pd500 import configure_sampled_pd500, install_sampled_pd500
    cfg = _impact_speed_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    configure_sampled_pd500(cfg, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        controller = install_sampled_pd500(env, gain_mode="variable")
        from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
        wrapped = wrap_impact_speed_env(env, 1.)
        wrapped.reset()
        action = torch.zeros(1, 13)
        action[:, 6:12] = 1.
        wrapped.step(action)
        snapshot = evaluator().physical_snapshot(env)
        torch.testing.assert_close(snapshot["kp"], controller.applied_kp)
        torch.testing.assert_close(snapshot["kd"], controller.applied_kd)
        torch.testing.assert_close(snapshot["kp"][0], torch.tensor([320., 480., 480., 320., 240., 160.]))
    finally:
        env.close()


def test_native_smoke_64_transitions_no_learning_and_real_export_parity(tmp_path, monkeypatch):
    import json
    from mjlab.tasks import registry
    original = registry.load_env_cfg
    def configured_task(task, *, play):
        assert play is False, "play disables finite timeouts and changes reset distribution"
        return original(task, play=play)
    monkeypatch.setattr(registry, "load_env_cfg", configured_task)
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--speed-bounds", ".2", ".8",
        "--speed-cost", ".4", "--seed", "7", "--num-envs", "1", "--mode", "smoke",
        "--output", str(tmp_path / "native")])
    summary = evaluator().run(args)
    assert summary["transitions"] == 64
    assert summary["learning_updates"] == 0
    assert summary["conditions"][0]["contract"]["safety"]["cat_soft"]["imp_max_p"] == .2
    assert summary["conditions"][0]["parity"] == {"pt": "passed", "onnx": "passed: ReferenceEvaluator", "observations": 3}
    assert summary["conditions"][0]["selection_count"] >= 1
    assert summary["conditions"][0]["contract"]["observation_widths"] == {"actor": 42, "critic": 42}
    rows = [json.loads(line) for line in (args.output / "episodes.jsonl").read_text().splitlines()]
    assert rows and rows[0]["steps"] > 0
    assert rows[0]["tracking_rms_mm"] >= 0
    assert len(rows[0]["kp_mean"]) == 6
    assert "common_window" in rows[0]["measurement"]
    loads = rows[0]["measurement"]["loads"]
    assert loads["signal"] == summary["conditions"][0]["measurement"]["load_signal"]
    assert loads["common_window_semantics"] == (
        "peak of max(current rolling Lambda, current control-step peak) during the onset window; "
        "includes pre-onset ring history and any retained pre-onset control-step peak")
    assert loads["exceedance_duration_semantics"] == (
        "physics-sample duration above cap for the authoritative peak-held signal, "
        "not instantaneous rolling-Lambda exceedance duration")
    assert summary["conditions"][0]["measurement"]["site_state_offset_physics_samples"] == -1
    with pytest.raises(FileExistsError):
        evaluator().run(args)


def test_variable_sampled_smoke_installs_controller_before_runner(tmp_path):
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--speed-bounds", ".2", ".8",
        "--speed-cost", ".4", "--seed", "7", "--num-envs", "1", "--mode", "smoke",
        "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
        "--output", str(tmp_path / "variable")])
    summary = evaluator().run(args)
    contract = summary["conditions"][0]["contract"]
    assert contract["action_type"] == "joint_position_variable_impedance_sampled_pd"
    assert contract["controller_identity"] == "sampled_vic500_v1"
    assert summary["conditions"][0]["parity"]["pt"] == "passed"


def test_native_terminal_physical_snapshot_survives_reset():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    cfg = _impact_speed_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        wrapped = wrap_impact_speed_env(env, 1.)
        env.episode_length_buf[:] = env.max_episode_length - 1
        action = torch.zeros(1, 13)
        action[:, 6:12] = 1.
        with evaluator().PreResetSnapshot(env) as capture:
            _, _, done, extras = wrapped.step(action)
        assert done.item() == 1
        assert capture.latest["timeout"].item() is True
        nominal = env.action_manager.get_term("joint_stiffness").nominal_kp
        torch.testing.assert_close(capture.latest["kp"][0], nominal * 1.25)
        assert extras["impact_speed_outcome"]["charged"].item() is True
        assert env.action_manager.get_term("strike_speed").state.pending.item() == 1.
        assert torch.isfinite(capture.latest["joint_target_error_rad"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("protocol", ["native", "common-window"])
def test_native_paired_interventions_recompute_actions_and_gains(tmp_path, monkeypatch, protocol):
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    cfg = _impact_speed_env_cfg(play=False)
    cfg.scene.num_envs = 2
    cfg.seed = 7
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    checkpoint = tmp_path / "untrained-fixture.pt"
    try:
        agent_cfg = asdict(z1_impact_speed_ppo_runner_cfg())
        agent_cfg["upload_model"] = False
        agent_cfg["logger"] = "tensorboard"
        runner = HammerOnPolicyRunner(wrap_impact_speed_env(env, 1.), agent_cfg)
        runner.save(str(checkpoint))
    finally:
        env.close()
    original = registry.load_env_cfg
    def short_test_episode(task, *, play):
        cfg = original(task, play=play)
        cfg.episode_length_s = .06  # Only a test fixture; evaluator must not change native timeout.
        return cfg
    monkeypatch.setattr(registry, "load_env_cfg", short_test_episode)
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--checkpoint", str(checkpoint),
        "--speed-bounds", ".2", ".8", "--speed-cost", ".4", "--seed", "7", "--num-envs", "2",
        "--mode", "intervention", "--goals", ".2", ".5", ".8", "--output", str(tmp_path / "paired"),
        "--measurement-protocol", protocol])
    summary = evaluator().run(args)
    assert summary["transitions"] == 18
    assert summary["timeout_count"] == 6
    conditions = summary["conditions"]
    assert len({c["first_action_hash"] for c in conditions}) == 1
    assert conditions[0]["reset_hashes"] == conditions[2]["reset_hashes"]
    contrast = summary["comparisons"][-1]
    assert contrast["second_motor_action_rms_difference"] > 0
    assert contrast["second_kp_rms_difference"] > 0
    assert contrast["responsive"] is False  # Command change is not achieved-speed evidence.
    assert contrast["coverage"]["low"]["misses"] == 2

"""Bounded evaluator: paired evidence and transition-safe episode accounting."""

import importlib
from types import SimpleNamespace

import pytest
import torch


def evaluator():
    return importlib.import_module("evaluate_impact_speed")


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


def test_paired_speed_error_excludes_asymmetric_misses_but_reports_coverage():
    paired = []
    outcomes = ((True, True), (True, False), (False, True))
    for condition in (0, 1):
        rows = []
        for world, pair_hits in enumerate(outcomes):
            hit = pair_hits[condition]
            speed = (.5 + .1 * condition) if hit else None
            rows.append({
                "condition": condition, "world": world, "episode": 0,
                "episode_id": f"{world}:0", "reset_hash": f"reset-{world}",
                "route_sign": (-1, 0, 1)[world], "hit": hit, "miss": not hit,
                "complete": True, "goal_mps": .6 + .1 * condition,
                "measured_mps": speed,
                "measurement": {
                    "common_window": {"window_complete": True, "axial_positive_ns": .3},
                    "precontact_path": {"coverage": 1., "observed_bin_rms_m": .002},
                    "loads": {"episode": {
                        "peak_nms": [.1] * 6, "exceeded": [False] * 6,
                        "exceedance_duration_s": [0.] * 6, "observed_samples": 10,
                    }},
                },
            })
        paired.append(rows)

    report = evaluator().compare_paired_outcomes(*paired, seed=11)

    assert report["paired_metrics"]["terminal_speed_mps"]["common_pairs"] == 1
    assert report["paired_metrics"]["absolute_speed_error_mps"]["common_pairs"] == 1
    assert report["coverage"]["low_hits"] == report["coverage"]["high_hits"] == 2
    assert report["coverage"]["low_misses"] == report["coverage"]["high_misses"] == 1
    assert report["coverage"]["low_valid_contact_speeds"] == 2
    assert report["coverage"]["high_valid_contact_speeds"] == 2
    assert report["coverage"]["absolute_speed_error_common_pairs"] == 1
    assert "conditional" in report["coverage"]["speed_error_semantics"]


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


def test_paired_outcome_contrast_covers_impact_tracking_and_each_joint_load():
    rows = []
    for condition, goal, speed, impulse, path_mm, path_coverage, peak, exceeded, complete, samples, depth_peak in (
        (0, .6, .5, .3, 3., 1., [.2, .2, .2, .2, .2, .01], [False] * 6, True, 12, .010),
        (0, .6, .6, .4, 4., .5, [.2, .3, .2, .2, .2, .01], [False, True, False, False, False, False], False, 3, .004),
        (1, .8, .7, .4, 2., 1., [.3, .2, .2, .2, .2, .01], [True, False, False, False, False, False], True, 12, .012),
        (1, .8, .8, .5, 3., 1., [.2, .4, .2, .2, .2, .01], [False, True, False, False, False, False], False, 3, .006),
    ):
        world = len(rows) % 2
        rows.append({
            "condition": condition, "world": world, "episode": 0,
            "episode_id": f"{world}:0", "reset_hash": f"reset-{world}",
            "route_sign": -1 if world == 0 else 1, "hit": True, "goal_mps": goal,
            "measured_mps": speed, "complete": complete,
            "nail_depth_peak_m": depth_peak,
            "measurement": {
                "common_window": {"window_complete": True, "axial_positive_ns": impulse},
                "precontact_path": {"observed_bin_rms_m": path_mm / 1000,
                                    "coverage": path_coverage},
                "loads": {"episode": {
                    "peak_nms": peak, "exceeded": exceeded,
                    "exceedance_duration_s": [0.] * 6,
                    "observed_samples": samples,
                }},
            },
        })

    report = evaluator().compare_paired_outcomes(
        [row for row in rows if row["condition"] == 0],
        [row for row in rows if row["condition"] == 1], seed=9,
    )

    assert report["paired_episode_count"] == 2
    assert report["paired_metrics"]["positive_axial_impulse_50ms_ns"]["mean_difference_high_minus_low"] == pytest.approx(.1)
    assert report["paired_metrics"]["absolute_speed_error_mps"]["mean_difference_high_minus_low"] == pytest.approx(0.)
    assert report["paired_metrics"]["precontact_spline_rms_mm"]["mean_difference_high_minus_low"] == pytest.approx(-1.)
    assert report["paired_metrics"]["precontact_spline_rms_mm"]["common_pairs"] == 1
    assert report["paired_metrics"]["peak_sampled_nail_depth_mm"]["common_pairs"] == 1
    assert report["paired_metrics"]["peak_sampled_nail_depth_mm"]["mean_difference_high_minus_low"] == pytest.approx(2.)
    assert report["paired_metrics"]["path_coverage_fraction"]["mean_difference_high_minus_low"] == pytest.approx(.25)
    assert report["coverage"]["full_path_pairs"] == 1
    assert report["coverage"]["incomplete_path_pairs"] == 1
    assert report["coverage"]["complete_paired_attempts"] == 1
    assert report["coverage"]["censored_paired_attempts"] == 1
    assert report["coverage"]["speed_error_semantics"].startswith("absolute error conditional")
    assert report["per_joint_loads"][0]["added_violation_pairs"] == 1
    assert report["per_joint_loads"][0]["removed_violation_pairs"] == 0
    assert report["per_joint_loads"][0]["violation_rate_difference_high_minus_low"] == pytest.approx(1.)
    assert report["per_joint_loads"][1]["high_observed_exceedance_attempts"] == 1
    assert report["per_joint_loads"][1]["high_exceedance_attempts"] == 0
    assert report["by_route"]["R-"]["paired_episode_count"] == 1


def transition(charged=(False, False), done=(False, False)):
    n = len(charged)
    outcome = {key: torch.zeros(n, dtype=torch.bool) for key in
               ("charged", "finalized", "valid_measured", "started", "miss", "incomplete", "done", "productive")}
    outcome.update(charged=torch.tensor(charged), done=torch.tensor(done),
                   goal_mps=torch.full((n,), .8), measured_mps=torch.full((n,), float("nan")),
                   cost=torch.zeros(n))
    physical = {"path_error_m": torch.full((n,), .003), "joint_target_error_rad": torch.full((n,), .2),
                "nail_depth_m": torch.zeros(n),
                "first_contact_time_s": torch.full((n,), .1), "started": torch.zeros(n, dtype=torch.bool),
                "delivered_impulse_ns": torch.full((n,), 2.), "joint_impulse_nms": torch.ones(n, 6),
                "cat_delta": torch.full((n,), .1), "impulse_exceeded": torch.zeros(n, 6, dtype=torch.bool),
                "kp": torch.full((n, 6), 40.), "kd": torch.full((n, 6), 4.),
                "timeout": torch.zeros(n, dtype=torch.bool)}
    return outcome, physical


def test_episode_accumulator_reports_sampled_depth_and_keeps_censored_peak():
    acc = evaluator().EpisodeAccumulator(2, gamma=.99, coefficient=.4)
    raw = torch.zeros(2, 13)

    for depths, done in (((0., 0.), (False, False)),
                         ((.010, .003), (False, False)),
                         ((.006, .002), (True, True))):
        outcome, physical = transition(done=done)
        physical["nail_depth_m"] = torch.tensor(depths)
        if done[1]:
            outcome["charged"][1] = True
            outcome["miss"][1] = True
        acc.update(outcome, physical, raw, torch.zeros(2))

    rows = sorted(acc.rows, key=lambda row: row["world"])
    assert rows[0]["nail_depth_peak_m"] == pytest.approx(.010)
    assert rows[0]["nail_depth_peak_m"] == pytest.approx(.010)
    assert rows[1]["nail_depth_peak_m"] == pytest.approx(.003)
    assert rows[1]["miss"] is True

    censored = evaluator().EpisodeAccumulator(1, gamma=.99, coefficient=.4)
    outcome, physical = transition((False,), done=(False,))
    physical["nail_depth_m"] = torch.tensor([.020])
    censored.update(outcome, physical, torch.zeros(1, 13), torch.zeros(1))
    partial = censored.finish()[0]
    assert partial["nail_depth_peak_m"] == pytest.approx(.020)


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


def test_contact_safety_record_is_attached_to_completed_first_episode():
    acc = evaluator().EpisodeAccumulator(1, gamma=.5, coefficient=.4)
    outcome, physical = transition((False,), done=(True,))
    timing = {"schema_version": "contact_safety_timing_v1", "sample_count": 12}

    acc.update(
        outcome, physical, torch.zeros(1, 13), torch.zeros(1),
        evaluation=[{"termination_reasons": "nail_driven"}],
        contact_safety_records=[timing],
    )

    row = acc.finish()[0]
    assert row["contact_safety_timing"] == {
        "schema_version": "contact_safety_timing_v1",
        "sample_count": 12,
        "episode_steps": 1,
        "termination_reasons": "nail_driven",
    }
    assert row["complete"] is True


def test_contact_safety_timing_summary_reports_joint_peaks_and_phase_counts():
    names = [f"joint{i}" for i in range(1, 7)]

    def record(velocity_peak, velocity_crossing, velocity_offset,
               load_peak, load_crossing, load_offset):
        return {
            "schema_version": "contact_safety_timing_v1", "sample_count": 20,
            "episode_steps": 1, "termination_reasons": "nail_driven",
            "accepted_contact_sample": 5, "first_raw_contact_sample": 4,
            "joint_velocity": {
                "joint_names": names, "limit_rad_s": 3.1415,
                "peak_abs_rad_s": velocity_peak,
                "first_crossing_sample": velocity_crossing,
                "relative_to_accepted_contact_s": velocity_offset,
            },
            "joint_load": {
                "joint_names": names, "signal": "windowed-substep-impulse",
                "caps_nms": [.369, .246, .738, .369, .246, .0164],
                "peak_nms": load_peak,
                "first_crossing_sample": load_crossing,
                "relative_to_accepted_contact_s": load_offset,
            },
            "torque_saturation": {
                "first_crossing_sample": [None] * 6,
                "relative_to_accepted_contact_s": [None] * 6,
            },
        }

    rows = [
        {"world": 0, "episode": 0, "complete": True,
         "contact_safety_timing": record(
             [3.2, 2., 1., 1., 1., 1.], [2, None, None, None, None, None],
             [-.00075, None, None, None, None, None], [.1, .3, .2, .2, .2, .01],
             [None, 7, None, None, None, None], [None, .0005, None, None, None, None],
         )},
        {"world": 1, "episode": 0, "complete": True,
         "contact_safety_timing": record(
             [2.8, 2., 1., 1., 1., 1.], [None] * 6, [None] * 6,
             [.1, .2, .2, .2, .2, .01], [None] * 6, [None] * 6,
         )},
    ]

    summary = evaluator().summarize_contact_safety_timing(rows, expected_episode_count=2)

    assert summary["completed_episode_count"] == 2
    assert summary["episodes_with_accepted_contact"] == 2
    assert summary["joint_velocity"]["maximum_observed_peak_abs_rad_s"] == [3.2, 2., 1., 1., 1., 1.]
    assert summary["joint_velocity"]["episodes_with_first_crossing"]["before_accepted_contact"][0] == 1
    assert summary["joint_load"]["episodes_with_first_crossing"]["at_or_after_accepted_contact"][1] == 1
    with pytest.raises(ValueError, match="missing"):
        evaluator().summarize_contact_safety_timing([dict(rows[0], contact_safety_timing=None)])


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
    for world, sign, goal, measured, depth_peak, complete in (
        (0, -1, 1.6, 1.1, .008, True), (1, -1, 1.8, None, .004, True),
        (2, 0, 1.9, 1.3, .012, True), (3, 1, 2.1, 1.5, .020, False),
    ):
        rows.append({
            "world": world, "episode": 0, "route_sign": sign,
            "hit": measured is not None, "miss": measured is None,
            "complete": complete, "nail_depth_peak_m": depth_peak,
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
    assert left["peak_sampled_nail_depth_mm"] == {"mean": pytest.approx(6.), "count": 2}
    assert left["complete_depth_attempts"] == 2
    assert left["censored_depth_attempts"] == 0
    assert report["peak_sampled_nail_depth_mm"] == {"mean": pytest.approx(8.), "count": 3}
    assert report["complete_depth_attempts"] == 3
    assert report["censored_depth_attempts"] == 1
    assert report["censored_observed_peak_depth_mm"] == {"mean": pytest.approx(20.), "count": 1}
    assert "not retained/final depth" in report["nail_depth_measurement_semantics"]
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


def test_preaction_fixed_goal_consumes_selector_and_refreshes_observations():
    from src.tasks.hammer.mdp.impact_speed import ImpactSpeedState
    state = ImpactSpeedState(2, "cpu", (.2, .8), "policy")
    tracker = SimpleNamespace(started=torch.tensor([False, False]))
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(state=state)),
        _hammer_first_strike=tracker,
        num_envs=2,
    )
    observations = {
        "actor": torch.cat((torch.arange(80, dtype=torch.float32).reshape(2, 40), state.observation()), dim=1),
        "critic": torch.cat((torch.arange(80, 160, dtype=torch.float32).reshape(2, 40), state.observation()), dim=1),
    }
    before = {group: value.clone() for group, value in observations.items()}
    rng_before = torch.get_rng_state().clone()

    updated = evaluator().set_goal_before_first_action(env, .8, observations)

    for group in ("actor", "critic"):
        torch.testing.assert_close(updated[group][:, :40], before[group][:, :40])
        torch.testing.assert_close(updated[group][:, 40], torch.ones(2))
        torch.testing.assert_close(updated[group][:, 41], torch.zeros(2))
        torch.testing.assert_close(observations[group], before[group])
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert not state.pending.any()
    state.pending.fill_(1.)
    with pytest.raises(ValueError, match="outside speed bounds"):
        evaluator().set_goal_before_first_action(env, .9, observations)


def test_preaction_fixed_goal_validates_both_groups_before_mutating_state():
    from src.tasks.hammer.mdp.impact_speed import ImpactSpeedState
    state = ImpactSpeedState(1, "cpu", (.2, .8), "policy")
    tracker = SimpleNamespace(started=torch.tensor([False]))
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(state=state)),
        _hammer_first_strike=tracker, num_envs=1,
    )
    observations = {"actor": torch.zeros(1, 42), "critic": torch.zeros(1, 41)}
    initial_goal, initial_pending = state.goal_mps.clone(), state.pending.clone()

    with pytest.raises(ValueError, match="42D"):
        evaluator().set_goal_before_first_action(env, .8, observations)

    torch.testing.assert_close(state.goal_mps, initial_goal)
    torch.testing.assert_close(state.pending, initial_pending)


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


def test_post_terminal_shadow_cli_requires_policy_common_window_and_full_trace(tmp_path):
    base = ["--task", "task", "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "2", "--checkpoint", "policy.pt",
            "--output", str(tmp_path / "shadow"), "--mode", "policy",
            "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
            "--measurement-protocol", "common-window", "--trace-contact"]
    args = evaluator().parse_args(base + ["--post-terminal-shadow"])
    assert args.post_terminal_shadow is True
    assert args.shadow_post_onset_horizon_s == .250
    assert args.shadow_hard_cap_from_reset_s == 1.0
    with pytest.raises(SystemExit):
        evaluator().parse_args([arg for arg in base if arg != "--trace-contact"]
                               + ["--post-terminal-shadow"])
    native_protocol = list(base)
    native_protocol[native_protocol.index("common-window")] = "native"
    with pytest.raises(SystemExit):
        evaluator().parse_args(native_protocol + ["--post-terminal-shadow"])


def test_post_terminal_shadow_cli_accepts_configured_window_and_validates_timing(tmp_path):
    base = ["--task", "task", "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "2", "--checkpoint", "policy.pt",
            "--output", str(tmp_path / "shadow"), "--mode", "policy",
            "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
            "--measurement-protocol", "common-window", "--trace-contact",
            "--post-terminal-shadow"]
    args = evaluator().parse_args(base + [
        "--shadow-post-onset-horizon-s", ".5",
        "--shadow-hard-cap-from-reset-s", "1.0",
    ])
    assert args.shadow_post_onset_horizon_s == .5
    assert args.shadow_hard_cap_from_reset_s == 1.0

    for invalid_args in (
        ["--shadow-post-onset-horizon-s", "0"],
        ["--shadow-post-onset-horizon-s", "nan"],
        ["--shadow-hard-cap-from-reset-s", "inf"],
        ["--shadow-post-onset-horizon-s", ".5",
         "--shadow-hard-cap-from-reset-s", ".5"],
    ):
        with pytest.raises(SystemExit):
            evaluator().parse_args(base + invalid_args)


def test_post_terminal_shadow_timing_overrides_require_shadow_mode(tmp_path):
    base = ["--task", "task", "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "2", "--checkpoint", "policy.pt",
            "--output", str(tmp_path / "shadow"), "--mode", "policy",
            "--shadow-post-onset-horizon-s", ".5"]
    with pytest.raises(SystemExit):
        evaluator().parse_args(base)


def test_post_terminal_shadow_cli_accepts_one_external_target(tmp_path):
    base = ["--task", "task", "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "2", "--checkpoint", "external.pt",
            "--output", str(tmp_path / "external-shadow"), "--mode", "external",
            "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
            "--measurement-protocol", "common-window", "--trace-contact",
            "--post-terminal-shadow"]
    args = evaluator().parse_args(base + ["--goals", "1.63"])
    assert args.mode == "external"
    assert args.goals == [1.63]
    assert args.post_terminal_shadow is True

    with pytest.raises(SystemExit):
        evaluator().parse_args(base)
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--goals", "1.5", "1.85", "2.2"])


def test_post_terminal_shadow_suppresses_only_nail_success_and_restores_function():
    PostTerminalShadowProtocol = evaluator().PostTerminalShadowProtocol

    native = torch.tensor([True, False])
    calls = []
    def nail_success(env, **params):
        return native.clone()
    def latch_success(values):
        calls.append(values.clone())
        return values
    original = nail_success
    cfg = SimpleNamespace(func=original)
    env = SimpleNamespace(
        termination_manager=SimpleNamespace(_term_names=["nail_driven"], _term_cfgs=[cfg]),
        _strike_evaluation=SimpleNamespace(success_termination=latch_success),
    )

    with PostTerminalShadowProtocol(env):
        shadow_done = cfg.func(env)
        assert shadow_done.tolist() == [False, False]
        assert len(calls) == 1
        assert calls[0].tolist() == [True, False]

    assert cfg.func is original


def test_cat_dose_audit_cli_is_one_policy_native_replay_only(tmp_path):
    base = ["--task", "task", "--speed-bounds", "1.5", "2.2", "--speed-cost", ".4",
            "--seed", "7", "--num-envs", "2", "--checkpoint", "policy.pt",
            "--output", str(tmp_path / "dose")]
    args = evaluator().parse_args(base + ["--mode", "policy", "--cat-dose-audit"])
    assert args.cat_dose_audit is True
    assert args.cat_dose_target_max_p == .75
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "paired", "--goals", "1.5", "1.8", "2.2",
                                       "--cat-dose-audit"])
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "policy", "--measurement-protocol", "common-window",
                                       "--cat-dose-audit"])
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--mode", "policy", "--cat-dose-audit",
                                       "--cat-dose-target-max-p", ".5"])


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
        speed_state = env.action_manager.get_term("strike_speed").state
        torch.testing.assert_close(snapshot["desired_speed_mps"], speed_state.goal_mps)
        assert "geometric_path_error_m" in snapshot
        assert snapshot["nail_depth_m"].shape == (1,)
        assert snapshot["nail_depth_m"].item() == pytest.approx(0.)
        torch.testing.assert_close(snapshot["kp"][0], torch.tensor([320., 480., 480., 320., 240., 160.]))
    finally:
        env.close()


def test_cat_dose_snapshot_reads_the_actual_post_update_velocity_normalizer():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env
    from scripts.cat_dose_audit import counterfactual_components

    cfg = _impact_speed_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        wrapped = wrap_impact_speed_env(env, 1.)
        wrapped.reset()
        wrapped.step(torch.zeros(2, 13))
        hook = next(term.func for name, term in zip(
            env.metrics_manager.active_terms, env.metrics_manager._term_cfgs
        ) if name == "cat_soft")
        normalizer_before = hook._cat.running_maxes["joint_velocity_excess"].clone()
        snapshot = evaluator().physical_snapshot(env, include_cat_dose_audit=True)

        torch.testing.assert_close(
            snapshot["cat_velocity_margin_per_joint"],
            hook._cat.raw_constraints["joint_velocity_excess"],
        )
        torch.testing.assert_close(
            snapshot["cat_velocity_normalizer_per_joint"],
            normalizer_before.expand_as(snapshot["cat_velocity_margin_per_joint"]),
        )
        impulse_margin = hook._cat.raw_constraints["joint_impulse_excess"]
        impulse_max, impulse_world = impulse_margin.clamp_min(0.).max(dim=0)
        torch.testing.assert_close(snapshot["cat_impulse_margin_per_joint"], impulse_margin)
        torch.testing.assert_close(snapshot["cat_impulse_cmax_per_joint"], hook._imp_cmax.squeeze(0))
        assert torch.equal(snapshot["cat_impulse_seeded_per_joint"], hook._imp_seeded.squeeze(0))
        torch.testing.assert_close(snapshot["cat_impulse_batch_max_margin_per_joint"], impulse_max)
        assert torch.equal(snapshot["cat_impulse_batch_max_world_index_per_joint"], impulse_world)
        assert torch.equal(snapshot["cat_impulse_batch_max_violated_per_joint"], impulse_max > 0.)
        counterfactual_components(
            snapshot["cat_velocity_margin_per_joint"],
            snapshot["cat_velocity_normalizer_per_joint"],
            snapshot["delta_impulse_per_joint"],
            baseline_max_p=.5, target_max_p=.75, min_p=0.,
            observed_velocity_per_joint=snapshot["delta_velocity_per_joint"],
            observed_combined_delta=snapshot["cat_delta"],
        )
        torch.testing.assert_close(hook._cat.running_maxes["joint_velocity_excess"], normalizer_before)
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


def test_contact_safety_summary_cli_requires_paired_sampled_common_window(tmp_path):
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed", "--speed-bounds", "1.5", "2.2",
        "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--speed-cost", ".8493676", "--seed", "20260926", "--num-envs", "2",
        "--mode", "paired", "--goals", "1.5", "1.85", "2.2",
        "--sampled-pd500", "--measurement-protocol", "common-window",
        "--contact-safety-summary", "--output", str(tmp_path / "timing"),
    ])

    assert args.contact_safety_summary is True


def test_contact_safety_summary_cli_accepts_single_policy_common_window(tmp_path):
    base = [
        "--task", "Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed",
        "--speed-bounds", "1.5", "2.2", "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--speed-cost", ".8493676", "--seed", "20260926", "--num-envs", "2",
        "--mode", "policy", "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
        "--contact-safety-summary", "--output", str(tmp_path / "policy-timing"),
    ]

    args = evaluator().parse_args(base + ["--measurement-protocol", "common-window"])

    assert args.mode == "policy"
    assert args.measurement_protocol == "common-window"
    assert args.contact_safety_summary is True
    with pytest.raises(SystemExit):
        evaluator().parse_args(base + ["--measurement-protocol", "native"])


def test_single_policy_common_window_records_nail_depth_and_joint_safety(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller, install_impact_speed_controller,
        wrap_impact_speed_env,
    )
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    cfg = _impact_speed_env_cfg(play=False)
    cfg.scene.num_envs = 1
    cfg.seed = 7
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    configure_impact_speed_controller(cfg, sampled_pd500=True, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    checkpoint = tmp_path / "single-policy-fixture.pt"
    try:
        install_impact_speed_controller(env, sampled_pd500=True, gain_mode="variable")
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
        cfg.episode_length_s = .06
        return cfg

    monkeypatch.setattr(registry, "load_env_cfg", short_test_episode)
    output = tmp_path / "single-policy"
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--checkpoint", str(checkpoint),
        "--speed-bounds", ".2", ".8", "--speed-cost", ".4", "--seed", "7", "--num-envs", "1",
        "--mode", "policy", "--output", str(output), "--sampled-pd500",
        "--sampled-pd500-gain-mode", "variable", "--measurement-protocol", "common-window",
        "--contact-safety-summary",
    ])

    summary = evaluator().run(args)

    assert summary["mode"] == "policy"
    assert summary["learning_updates"] == 0
    assert summary["contact_safety_summary_enabled"] is True
    assert summary["comparisons"] == []
    assert len(summary["conditions"]) == 1
    condition = summary["conditions"][0]
    timing = condition["contact_safety_timing"]
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    first = [row for row in rows if row["episode"] == 0]
    assert timing["schema_version"] == "contact_safety_timing_v1"
    assert timing["first_episode_count"] == timing["timed_episode_count"] == 1
    assert timing["completed_episode_count"] == 1
    assert len(first) == 1
    record = first[0]["contact_safety_timing"]
    assert record["sample_count"] == record["episode_steps"] * 80
    assert record["sample_count"] == first[0]["measurement"]["loads"]["episode"]["observed_samples"]
    assert len(timing["joint_velocity"]["maximum_observed_peak_abs_rad_s"]) == 6
    assert len(timing["joint_load"]["maximum_observed_peak_nms"]) == 6
    assert len(timing["joint_load"]["caps_nms"]) == 6
    assert first[0]["selection_count"] == 1
    assert first[0]["nail_depth_peak_m"] is not None
    assert condition["measurement"]["protocol"] == "common-window"


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
        assert capture.latest["nail_depth_m"].item() == pytest.approx(0.)
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


def test_paired_mode_matches_autonomous_policy_and_preaction_fixed_targets(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.config.z1.rl_cfg import z1_impact_speed_ppo_runner_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller, install_impact_speed_controller,
        wrap_impact_speed_env)
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

    cfg = _impact_speed_env_cfg(play=False)
    cfg.scene.num_envs = 2
    cfg.seed = 7
    cfg.actions["strike_speed"].bounds = (.2, .8)
    cfg.rewards["speed_goal"].weight = -.4
    configure_impact_speed_controller(cfg, sampled_pd500=True, gain_mode="variable")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    checkpoint = tmp_path / "paired-policy-fixture.pt"
    try:
        agent_cfg = asdict(z1_impact_speed_ppo_runner_cfg())
        agent_cfg["upload_model"] = False
        agent_cfg["logger"] = "tensorboard"
        install_impact_speed_controller(env, sampled_pd500=True, gain_mode="variable")
        runner = HammerOnPolicyRunner(wrap_impact_speed_env(env, 1.), agent_cfg)
        runner.save(str(checkpoint))
    finally:
        env.close()

    original = registry.load_env_cfg

    def short_test_episode(task, *, play):
        cfg = original(task, play=play)
        cfg.episode_length_s = .06
        return cfg

    monkeypatch.setattr(registry, "load_env_cfg", short_test_episode)
    output = tmp_path / "paired-policy"
    args = evaluator().parse_args([
        "--task", "Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed", "--checkpoint", str(checkpoint),
        "--speed-bounds", ".2", ".8", "--speed-cost", ".4", "--seed", "7", "--num-envs", "2",
        "--mode", "paired", "--goals", ".2", ".5", ".8", "--output", str(output),
        "--sampled-pd500", "--sampled-pd500-gain-mode", "variable",
        "--measurement-protocol", "common-window", "--contact-safety-summary",
    ])
    summary = evaluator().run(args)

    conditions = summary["conditions"]
    assert summary["learning_updates"] == 0
    assert summary["contact_safety_summary_enabled"] is True
    assert summary["initial_state_pairing"] == "verified_policy_vs_preaction_fixed_targets"
    assert len(conditions) == 4
    assert all(condition["contact_safety_timing"]["timed_episode_count"] == 2
               for condition in conditions)
    assert all(len(condition["contact_safety_timing"]["observer_source_sha256"]) == 64
               for condition in conditions)
    assert conditions[0]["goal_mps"] is None
    assert [condition["goal_mps"] for condition in conditions[1:]] == [.2, .5, .8]
    assert len({tuple(condition["reset_hashes"]) for condition in conditions}) == 1
    assert len({tuple(condition["first_episode_route_signs"]) for condition in conditions}) == 1
    assert len({condition["torch_post_reset_rng_hash"] for condition in conditions}) == 1
    assert len({condition["cuda_post_reset_rng_hash"] for condition in conditions}) == 1
    assert len({condition["observation_physical_prefix_hash"] for condition in conditions}) == 1
    assert all(condition["contract"] == conditions[0]["contract"] for condition in conditions)
    assert conditions[0]["selector_validation"]["total_selections"] == 2
    assert [condition["selector_validation"]["total_selections"] for condition in conditions[1:]] == [0, 0, 0]
    assert len(summary["comparisons"]) == 6
    first_outcome_contrast = summary["comparisons"][0]["outcome_contrast"]
    assert first_outcome_contrast["paired_episode_count"] == 2
    labels = {-1: "R-", 0: "R0", 1: "R+"}
    assert set(first_outcome_contrast["by_route"]) == {
        labels[sign] for sign in conditions[0]["first_episode_route_signs"]
    }
    assert len(first_outcome_contrast["per_joint_loads"]) == 6

    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    first = [row for row in rows if row["episode"] == 0]
    assert len(first) == 8
    for condition, goal in enumerate((None, .2, .5, .8)):
        arm = [row for row in first if row["condition"] == condition]
        assert len(arm) == 2
        assert [row["goal_mps"] for row in arm] == pytest.approx([goal] * 2) if goal is not None else all(
            row["selection_count"] == 1 and row["selected_goal_mps"] is not None for row in arm
        )
        if goal is not None:
            assert all(row["selection_count"] == 0 and row["selected_goal_mps"] is None for row in arm)

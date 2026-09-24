"""Hand-derived physics-cadence evidence, independent of reward-time caches."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch


def _reference_metric():
    from src.tasks.hammer.mdp.references import SingleStrikeReference
    from src.tasks.hammer.mdp.strike_evaluation import StrikeEvaluationMetric
    ref = SingleStrikeReference(1, "cpu", overshoot=.02, path_shape="quintic")
    ref.update(torch.tensor([[.48, 0., .25]]), torch.tensor([[.5, 0., .1]]), torch.tensor([0]))
    head = torch.zeros(1, 1, 3)
    tracker = SimpleNamespace(started=torch.tensor([False]), _axis=torch.tensor([0., 0., -1.]),
        _site_ids=[0], _robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=head)),
        _contact_sensor=SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1))),
        _impulse_sensor=SimpleNamespace(data=SimpleNamespace(force=torch.zeros(1, 1, 3))))
    env = SimpleNamespace(num_envs=1, device="cpu", physics_dt=.00025,
        cfg=SimpleNamespace(metrics={"cat_soft": SimpleNamespace(params={"imp_limit": [1.] * 6})}),
        _strike_reference=ref, _hammer_first_strike=tracker,
        _hammer_substep_impulse=SimpleNamespace(impulse=torch.zeros(1, 6)))
    metric = StrikeEvaluationMetric(None, env)
    return env, metric, head, ref


def test_real_reference_metric_dense_approach_covers_all_bins_without_followthrough():
    env, metric, head, ref = _reference_metric()
    for approach in torch.linspace(0., .999, 1000):
        head[:, 0] = ref.waypoint((approach * (.15 / .17)).reshape(1))
        metric(env)
    path = metric.collector.records()[0]["precontact_path"]
    assert path["max_m"] < 1e-6
    assert path["coverage"] == 1.
    assert path["skipped_bins"] == []
    assert path["approach_height_extent"] == pytest.approx([0., .999], abs=1e-6)
    assert ref.peek().item() == 0.


def test_real_reference_metric_sparse_rebound_offcurve_preserves_phase():
    env, metric, head, ref = _reference_metric()
    for approach, lateral in ((.01, 0.), (.99, 0.), (.26, .06)):
        head[:, 0] = ref.waypoint(torch.tensor([approach * .15 / .17]))
        head[:, 0, 1] += lateral
        metric(env)
    path = metric.collector.records()[0]["precontact_path"]
    assert path["coverage"] == 3 / 16
    assert path["skipped_bins"] == [i for i in range(16) if i not in (0, 4, 15)]
    assert path["bins"][4]["max_m"] == pytest.approx(.06)
    assert path["approach_height_extent"] == pytest.approx([.01, .99], abs=1e-6)
    assert ref.peek().item() == 0.
    phase, _, error = ref.diagnose_current_height(head[:, 0])
    assert phase.item() == pytest.approx(.26 * .15 / .17)
    assert error.item() == pytest.approx(.06)


def collector(n=1):
    module = importlib.import_module("src.tasks.hammer.mdp.strike_evaluation")
    return module.StrikeEvaluationCollector(n, "cpu", .00025, [1.] * 6)


def sample(c, *, started=True, contact=True, force=(3., 4., -10.), load=None, phase=.5, error=.003):
    n = c.num_envs
    def vector(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype).expand(n)
    c.sample(started=vector(started, torch.bool), contact=vector(contact, torch.bool),
             force_world=torch.tensor(force).expand(n, 3),
             axis=torch.tensor([0., 0., -1.]),
             load=torch.zeros(n, 6) if load is None else torch.tensor(load).expand(n, 6),
             approach_height=vector(phase), error_m=vector(error))


def test_inclusive_200_sample_integrals_freeze_and_contact_gaps_count():
    c = collector()
    for _ in range(199):
        sample(c)
    r = c.records()[0]
    assert r["common_window"]["window_samples"] == 199
    assert not r["common_window"]["window_complete"]
    sample(c)
    r = c.records()[0]["common_window"]
    assert r["window_samples"] == 200 and r["window_complete"]
    assert r["axial_positive_ns"] == pytest.approx(10. * 200 * .00025)
    assert r["axial_signed_ns"] == pytest.approx(.5)
    assert r["transverse_magnitude_ns"] == pytest.approx(.25)
    sample(c, force=(0., 0., -1000.))
    assert c.records()[0]["common_window"] == r
    c.reset(None)
    sample(c, force=(0., 0., 10.))
    for _ in range(199):
        sample(c, contact=False)
    r = c.records()[0]["common_window"]
    assert r["window_complete"] and r["axial_positive_ns"] == 0
    assert r["axial_signed_ns"] == pytest.approx(-.0025)


def test_async_subset_reset_censoring_no_reset_state_leak():
    c = collector(2)
    sample(c, started=[True, False])
    sample(c, started=[True, True])
    evidence = c.records(["failure", "timeout"])
    assert [r["common_window"]["window_samples"] for r in evidence] == [2, 1]
    assert [r["common_window"]["censor_reason"] for r in evidence] == ["failure", "timeout"]
    c.reset(torch.tensor([0]))
    sample(c, started=[False, True])
    assert [r["common_window"]["window_samples"] for r in c.records()] == [0, 2]
    assert evidence[0]["common_window"]["axial_positive_ns"] == pytest.approx(.005)


def test_hovering_miss_has_low_coverage_not_a_good_path_completion():
    c = collector()
    for _ in range(100):
        sample(c, started=False, phase=.01, error=.001)
    r = c.records(["timeout"])[0]
    assert r["miss"]
    path = r["precontact_path"]
    assert path["sample_count"] == 100
    assert path["rms_m"] == pytest.approx(.001)
    assert path["coverage"] == 1 / 16
    assert path["skipped_bins"] == list(range(1, 16))
    assert path["bins"][1]["rms_m"] is None
    assert path["approach_height_extent"] == pytest.approx([.01, .01])
    sample(c, phase=.99, error=.1)
    assert c.records()[0]["precontact_path"] == path
    assert c.records()[0]["postcontact_path"]["rms_m"] == pytest.approx(.1)


def test_live_load_spike_all_joints_and_pre_onset_ring_history():
    c = collector()
    sample(c, started=False, load=[3., 0., 0., 0., 0., 0.])
    sample(c, load=[2., 2., 3., 4., 5., 6.])
    sample(c, load=[0.] * 6)
    r = c.records()[0]["loads"]
    assert r["episode"]["peak_nms"] == [3., 2., 3., 4., 5., 6.]
    assert r["common_window"]["peak_nms"] == [2., 2., 3., 4., 5., 6.]
    assert r["common_window"]["exceedance_samples"] == [1] * 6
    assert r["episode"]["exceedance_samples"] == [2, 1, 1, 1, 1, 1]
    assert r["common_window"]["utilization"] == [2., 2., 3., 4., 5., 6.]


@pytest.mark.parametrize("success_sample", [1, 200])
def test_success_is_latched_until_complete_even_after_rebound(success_sample):
    c = collector()
    for step in range(1, 201):
        sample(c)
        done = c.success_termination(torch.tensor([step == success_sample]))
        assert done.item() == (step == 200)
    assert c.records()[0]["native_success_sample"] == success_sample


def test_final_sample_success_and_failure_remain_distinct():
    c = collector()
    sample(c)
    assert not c.success_termination(torch.tensor([True])).item()
    r = c.records(["failure+success"])[0]
    assert r["native_success_sample"] == 1
    assert r["common_window"]["censor_reason"] == "failure+success"


def test_pre_reset_capture_carries_immutable_collector_record():
    from scripts.evaluate_impact_speed import PreResetSnapshot
    c = collector()
    sample(c)
    tracker = SimpleNamespace(delivered=torch.tensor([.1]), delivered_transverse=torch.tensor([.2]),
                              finalized=torch.tensor([True]), reason=torch.tensor([1]))
    env = SimpleNamespace(_strike_evaluation=c, _hammer_first_strike=tracker, num_envs=1,
        termination_manager=SimpleNamespace(_term_dones={"time_out": torch.tensor([True])}),
        metrics_manager=SimpleNamespace(compute=lambda: None))
    with PreResetSnapshot(env, reader=lambda e: {}) as capture:
        env.metrics_manager.compute()
        c.reset(None)
        tracker.delivered.zero_()
        record = capture.latest_evaluation[0]
    assert record["common_window"]["window_samples"] == 1
    assert record["common_window"]["censor_reason"] == "time_out"
    assert record["native"]["delivered_impulse_ns"] == pytest.approx(.1)


def test_native_collection_preserves_full_transition_and_rng():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation, MeasurementProtocol
    from contextlib import nullcontext
    traces = []
    for enabled in (False, True):
        cfg = _impact_speed_env_cfg(play=False)
        cfg.scene.num_envs, cfg.seed = 1, 23
        cfg.actions["strike_speed"].bounds = (.2, .8)
        cfg.rewards["speed_goal"].weight = -.4
        if enabled:
            configure_strike_evaluation(cfg)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            env.seed(23)
            env.reset()
            trace = []
            context = MeasurementProtocol(env, "native") if enabled else nullcontext()
            with context:
                for _ in range(3):
                    obs, rew, done, timeout, _ = env.step(torch.zeros(1, 13))
                    trace.append([v.clone() for v in (obs["actor"], obs["critic"], rew, done, timeout,
                        env.action_manager.action, torch.get_rng_state())])
            traces.append(trace)
        finally:
            env.close()
    for bare, observed in zip(*traces):
        for a, b in zip(bare, observed):
            assert torch.equal(a, b)


def test_real_metric_order_reads_live_accumulator_before_cat_policy_read():
    from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
    from tests.test_first_strike_event import _tracker
    from tests.test_impulse_bound import _acc
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation
    tracker, strike_step = _tracker(quality_instrumentation=False)
    accumulator, feed = _acc(1, window=25)
    # A sensor contact before the two-off-sample arming sequence is not accepted.
    trace = iter([(True, 1000.), (False, 0.), (False, 0.), (True, 0.), (True, 0.)])
    state = {}
    def first(env):
        state["contact"], state["load"] = next(trace)
        return strike_step(head_z=.1, depth=0., contact_on=state["contact"], downward_force=10.)
    def impulse(env):
        return feed(torch.full((1, 6), state["load"]), torch.tensor([[state["contact"]]]))
    cfg = SimpleNamespace(metrics={
        "first_strike": MetricsTermCfg(func=first, per_substep=True),
        "substep_impulse": MetricsTermCfg(func=impulse, per_substep=True),
        "cat_soft": MetricsTermCfg(func=lambda e: torch.zeros(1), params={"imp_limit": [1.] * 6})})
    configure_strike_evaluation(cfg)
    env = SimpleNamespace(num_envs=1, device="cpu", physics_dt=.002, cfg=cfg,
        _hammer_first_strike=tracker, _hammer_substep_impulse=accumulator,
        _strike_reference=SimpleNamespace(approach_height_coordinate=lambda h: torch.zeros(1), diagnose_current_height=lambda h:
            (torch.zeros(1), h, torch.zeros(1))))
    manager = MetricsManager(cfg.metrics, env)
    manager.compute_substep()
    c = env._strike_evaluation
    assert c.records()[0]["loads"]["episode"]["exceedance_samples"] == [1] * 6
    assert manager._step_count.item() == 0  # No policy-rate/CaT read has run.
    for _ in range(4):
        manager.compute_substep()
    r = c.records()[0]
    assert r["common_window"]["window_samples"] == 2
    assert r["loads"]["common_window"]["peak_nms"] == [2.] * 6
    assert r["common_window"]["axial_positive_ns"] == pytest.approx(.04)


@pytest.mark.parametrize("reason", ["failure", "time_out"])
@pytest.mark.parametrize("native_success", [False, True])
def test_protocol_preserves_failure_timeout_and_simultaneous_success(reason, native_success):
    from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
    from src.tasks.hammer.mdp.strike_evaluation import MeasurementProtocol
    c = collector()
    sample(c)
    env = SimpleNamespace(num_envs=1, device="cpu", _strike_evaluation=c)
    success = lambda e: torch.tensor([native_success])
    cfg = {"nail_driven": TerminationTermCfg(func=success),
           reason: TerminationTermCfg(func=lambda e: torch.tensor([True]), time_out=reason == "time_out")}
    env.termination_manager = TerminationManager(cfg, env)
    with MeasurementProtocol(env, "common-window"):
        assert env.termination_manager.compute().item()
        assert getattr(env.termination_manager, "time_outs" if reason == "time_out" else "terminated").item()
        assert c.records()[0]["native_success_sample"] == (1 if native_success else None)
    assert env.termination_manager._term_cfgs[0].func is success


def test_observed_bin_aggregate_does_not_weight_dwell_time_as_progress():
    c = collector()
    sample(c, started=False, phase=0., error=.03)
    for _ in range(3):
        sample(c, started=False, phase=1., error=.01)
    p = c.records()[0]["precontact_path"]
    assert p["rms_m"] == pytest.approx(math.sqrt(.0003))
    assert p["observed_bin_rms_m"] == pytest.approx(math.sqrt(.0005))
    assert p["observed_bin_max_m"] == pytest.approx(.03)


def test_summary_denominator_includes_no_contact_and_failed_attempts():
    module = importlib.import_module("src.tasks.hammer.mdp.strike_evaluation")
    c = collector()
    sample(c, started=False, load=[2.] * 6)
    rows = c.records(["failure"])
    c.reset(None)
    for _ in range(200):
        sample(c)
    rows += c.records(["nail_driven"])
    summary = module.summarize_strike_evaluations(rows)
    assert summary["attempts"] == 2
    assert summary["misses"] == 1
    assert summary["complete_windows"] == 1
    assert summary["censored_windows"] == 1
    assert summary["episode_joint_exceedance_attempts"] == [1] * 6
    assert summary["common_window_joint_exceedance_attempts"] == [0] * 6

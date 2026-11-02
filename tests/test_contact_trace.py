from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.contact_trace import ContactTrace


def test_contact_trace_records_raw_and_clamped_nail_slide_state():
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(
            qpos=torch.tensor([[.04], [-.003]]),
            qvel=torch.tensor([[.7], [-.2]]),
        )),
    )
    trace = ContactTrace.__new__(ContactTrace)
    trace.env = env
    trace.nail_qpos_id = 0
    trace.nail_dof_id = 0
    trace.active = torch.tensor([True, False])
    trace._rows = {}
    trace.native_success_sample = torch.zeros(2, dtype=torch.long)
    trace.shadow_window = None

    trace._sample_nail_slide()
    values = trace.tensors()

    assert values["nail_slide_position_m"][0].tolist() == pytest.approx([.04, 0.])
    assert values["nail_slide_velocity_mps"][0].tolist() == pytest.approx([.7, 0.])
    assert values["clamped_nail_depth_m"][0].tolist() == pytest.approx([.032, 0.])


def test_native_success_marker_is_first_episode_pre_reset_evidence():
    env = SimpleNamespace(
        _sampled_pd500=SimpleNamespace(), _hammer_first_strike=SimpleNamespace(),
        num_envs=2, device='cpu', physics_dt=.00025, max_episode_length=10,
        cfg=SimpleNamespace(decimation=80),
        sim=SimpleNamespace(mj_model=SimpleNamespace(
            joint=lambda name: SimpleNamespace(id=0),
            jnt_qposadr=[0], jnt_dofadr=[0],
        )),
    )
    trace = ContactTrace(env)
    trace.finish_step(torch.tensor([True, False]), [
        {'native_success_sample': 37}, {'native_success_sample': None},
    ])
    trace.finish_step(torch.tensor([False, True]), [
        {'native_success_sample': 2}, {'native_success_sample': 91},
    ])
    assert trace.tensors()['native_success_sample'].tolist() == [37, 91]
    assert trace.active.tolist() == [False, False]


def test_post_onset_trace_window_stops_at_horizon_and_censors_unhit_worlds():
    from scripts.contact_trace import PostOnsetTraceWindow

    window = PostOnsetTraceWindow(
        num_envs=2, device="cpu", physics_dt_s=.01,
        post_onset_horizon_s=.02, hard_cap_s=.05,
    )
    active = torch.ones(2, dtype=torch.bool)
    for sample in range(1, 6):
        accepted_onset = torch.tensor([sample == 2, False])
        sample_count = torch.full((2,), sample, dtype=torch.long)
        active = window.advance(active, sample_count, accepted_onset)
        if sample < 4:
            assert active[0]
        if sample < 5:
            assert active[1]

    assert window.onset_sample.tolist() == [2, 0]
    assert window.complete.tolist() == [True, False]
    assert window.censored.tolist() == [False, True]
    assert active.tolist() == [False, False]


def test_post_onset_trace_window_marks_early_native_done_as_censored():
    from scripts.contact_trace import PostOnsetTraceWindow

    window = PostOnsetTraceWindow(
        num_envs=1, device="cpu", physics_dt_s=.01,
        post_onset_horizon_s=.02, hard_cap_s=.05,
    )
    active = window.advance(
        torch.tensor([True]), torch.tensor([1]), torch.tensor([True]),
    )
    active = window.mark_done(active, torch.tensor([True]))

    assert not active.any()
    assert not window.complete.any()
    assert window.censored.all()


def _safety_summary_env(num_envs=1):
    joint_names = tuple(f"joint{i}" for i in range(1, 7))
    qpos = torch.tensor([[.1, .2, .3, .4, .5, .6, 0.]]).repeat(num_envs, 1)
    controller = SimpleNamespace(
        gain_mode="variable",
        gains=SimpleNamespace(joint_names=joint_names),
        ids=torch.arange(6),
        vids=torch.arange(7),
        qids=torch.arange(7),
        requested_joint_position=qpos[:, :6].clone(),
        applied_joint_position=qpos[:, :6].clone(),
        # Gain snapshots are emitted in the six-joint canonical action order;
        # torque/saturation arrays below remain in native seven-actuator order.
        applied_kp=torch.full((num_envs, 6), 100.),
        applied_kd=torch.full((num_envs, 6), 6.),
        requested_torque=torch.zeros(num_envs, 7),
        torque=torch.zeros(num_envs, 7),
        saturated=torch.zeros(num_envs, 7, dtype=torch.bool),
    )
    tracker = SimpleNamespace(
        started=torch.zeros(num_envs, dtype=torch.bool),
        _contact_sensor=SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1)),
        ),
    )
    accumulator = SimpleNamespace(impulse=torch.zeros(num_envs, 6))
    manager = SimpleNamespace(compute_substep=lambda: "metrics")
    env = SimpleNamespace(
        _sampled_pd500=controller,
        _hammer_first_strike=tracker,
        _hammer_substep_impulse=accumulator,
        cfg=SimpleNamespace(metrics={"cat_soft": SimpleNamespace(params={
            "limit": 3.1415, "imp_limit": [1.] * 6,
        })}),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=torch.zeros(num_envs, 7))),
        metrics_manager=manager,
        num_envs=num_envs,
        device="cpu",
        physics_dt=.00025,
    )
    return env


def test_contact_safety_summary_records_first_per_joint_crossing_relative_to_both_contacts():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env()
    summary = ContactSafetySummary(env)
    manager = env.metrics_manager

    env.sim.data.qvel[0, :6] = torch.tensor([3.1415, 3.2, 0., 0., 0., 0.])
    env._hammer_substep_impulse.impulse[0, 2] = 1.1
    env._sampled_pd500.requested_torque[0, :6] = torch.tensor([10., 11., 12., 13., 14., 15.])
    env._sampled_pd500.torque[0, :6] = torch.tensor([10., 11., 11., 13., 14., 15.])
    env._sampled_pd500.saturated[0, 2] = True
    with summary:
        manager.compute_substep()

        env.sim.data.qvel[0, 0] = 3.15
        env._hammer_first_strike._contact_sensor.data.found[0, 0] = 1.
        manager.compute_substep()

        env.sim.data.qvel[0, 3] = 3.2
        env._hammer_substep_impulse.impulse[0, 1] = 1.2
        env._hammer_first_strike.started[0] = True
        manager.compute_substep()

        env._sampled_pd500.saturated[0, 1] = True
        manager.compute_substep()
        records = summary.finish_step(torch.tensor([True]))

    record = records[0]
    assert record["schema_version"] == "contact_safety_timing_v1"
    assert record["sample_count"] == 4
    assert record["accepted_contact_sample"] == 3
    assert record["first_raw_contact_sample"] == 2
    velocity = record["joint_velocity"]
    assert velocity["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    assert velocity["limit_rad_s"] == pytest.approx(3.1415)
    assert velocity["peak_abs_rad_s"][:4] == pytest.approx([3.15, 3.2, 0., 3.2])
    assert velocity["exceedance_samples"][:4] == [3, 4, 0, 2]
    assert velocity["exceedance_duration_s"][:4] == pytest.approx([.00075, .001, 0., .0005])
    assert velocity["first_crossing_sample"][:4] == [2, 1, None, 3]
    assert velocity["relative_to_accepted_contact_s"][:4] == pytest.approx(
        [-.00025, -.0005, None, 0.], nan_ok=True,
    )
    assert velocity["relative_to_first_raw_contact_s"][:4] == pytest.approx(
        [0., -.00025, None, .00025], nan_ok=True,
    )

    load = record["joint_load"]
    assert load["first_crossing_sample"][:3] == [None, 3, 1]
    assert load["exceedance_duration_s"][:3] == pytest.approx([0., .0005, .001])
    assert load["relative_to_accepted_contact_s"][:3] == pytest.approx(
        [None, 0., -.0005], nan_ok=True,
    )
    context = load["context_at_first_crossing"][2]
    assert context["joint_load_nms"] == pytest.approx(1.1)
    assert {key: value for key, value in context.items() if key != "joint_load_nms"} == {
        "joint_velocity_rad_s": 0.,
        "requested_torque_nm": 12.,
        "applied_torque_nm": 11.,
        "torque_saturated": True,
        "kp": 100.,
        "kd": 6.,
    }
    saturation = record["torque_saturation"]
    assert saturation["first_crossing_sample"][:3] == [None, 4, 1]
    assert saturation["relative_to_accepted_contact_s"][:3] == pytest.approx(
        [None, .00025, -.0005], nan_ok=True,
    )


def test_contact_safety_summary_keeps_gain_snapshots_in_canonical_joint_order():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env()
    controller = env._sampled_pd500
    controller.ids = torch.tensor([0, 5, 1, 2, 3, 4])
    controller.vids = torch.tensor([0, 2, 3, 4, 5, 1, 6])
    controller.qids = controller.vids.clone()
    controller.applied_kp = torch.tensor([[10., 20., 30., 40., 50., 60.]])
    controller.applied_kd = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    # With the native actuator order [J1,J3,J4,J5,J6,J2,gripper], this
    # represents a J2 velocity crossing after canonical joint-order selection.
    env.sim.data.qvel[0, 1] = 3.2
    summary = ContactSafetySummary(env)

    with summary:
        env.metrics_manager.compute_substep()
        record = summary.finish_step(torch.tensor([True]))[0]

    context = record["joint_velocity"]["context_at_first_crossing"][1]
    assert context["kp"] == pytest.approx(20.)
    assert context["kd"] == pytest.approx(2.)


def test_summary_distinguishes_requested_onset_from_ramped_delivery_and_retains_late_jump():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env()
    controller = env._sampled_pd500
    reset = env.sim.data.qpos[:, :6].clone()
    summary = ContactSafetySummary(env)
    controller.requested_joint_position += torch.tensor([[.2, .1, 0., 0., 0., 0.]])
    with summary:
        for _ in range(8):
            env.metrics_manager.compute_substep()
        controller.applied_joint_position += torch.tensor([[.004, .002, 0., 0., 0., 0.]])
        env.sim.data.qpos[0, 0] += .001
        for _ in range(392):
            env.metrics_manager.compute_substep()
        controller.requested_joint_position[0, 0] = reset[0, 0] + .8
        controller.applied_joint_position = reset + torch.tensor([[.8, .1, 0., 0., 0., 0.]])
        env.sim.data.qpos[0, 0] = reset[0, 0] + .01
        env.metrics_manager.compute_substep()
        record = summary.finish_step(torch.tensor([True]))[0]

    target = record["target_delivery"]
    assert record["sample_count"] == 401
    assert target["reset_joint_position_rad"] == pytest.approx([.1, .2, .3, .4, .5, .6])
    assert target["first_requested_delta_from_reset_rad"] == pytest.approx([.2, .1, 0., 0., 0., 0.])
    assert target["first_applied_delta_from_reset_rad"] == pytest.approx([0.] * 6)
    assert target["max_abs_requested_jump_rad"] == pytest.approx([.6, .1, 0., 0., 0., 0.])
    assert target["max_abs_applied_jump_rad"] == pytest.approx([.796, .098, 0., 0., 0., 0.])
    assert target["requested_jump_count"] == [2, 1, 0, 0, 0, 0]
    assert target["applied_jump_count"] == [2, 2, 0, 0, 0, 0]
    assert target["max_abs_applied_tracking_error_rad"] == pytest.approx([.79, .1, 0., 0., 0., 0.])


def test_summary_target_snapshots_are_canonical_and_inactive_world_isolated_after_reset():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env(2)
    controller = env._sampled_pd500
    controller.ids = torch.tensor([0, 5, 1, 2, 3, 4])
    controller.qids = torch.tensor([0, 2, 3, 4, 5, 1, 6])
    summary = ContactSafetySummary(env)
    controller.requested_joint_position[0, 1] += .2
    controller.applied_joint_position[0, 1] += .05
    with summary:
        env.metrics_manager.compute_substep()
        first = summary.finish_step(torch.tensor([True, False]))[0]
        # A reset and subsequent commands in finished world0 must not overwrite
        # its original anchor or enter world1's first-attempt measurements.
        env.sim.data.qpos[0] = 99.
        controller.requested_joint_position[0] = 101.
        controller.applied_joint_position[0] = 100.
        controller.requested_joint_position[1, 2] += .3
        controller.applied_joint_position[1, 2] += .1
        env.metrics_manager.compute_substep()
        second = summary.finish_step(torch.tensor([False, True]))[1]

    assert summary._record(0) == first
    assert first["target_delivery"]["max_abs_requested_jump_rad"] == pytest.approx([0., .2, 0., 0., 0., 0.])
    assert first["target_delivery"]["max_abs_applied_jump_rad"] == pytest.approx([0., .05, 0., 0., 0., 0.])
    assert second["target_delivery"]["requested_jump_count"] == [0, 0, 1, 0, 0, 0]
    assert second["target_delivery"]["applied_jump_count"] == [0, 0, 1, 0, 0, 0]
    assert second["target_delivery"]["first_requested_delta_from_reset_rad"] == pytest.approx([0.] * 6)


@pytest.mark.parametrize("field", ["requested_joint_position", "applied_joint_position"])
def test_summary_rejects_missing_or_wrong_width_target_snapshot(field):
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env()
    summary = ContactSafetySummary(env)
    setattr(env._sampled_pd500, field, torch.zeros(1, 7))
    with pytest.raises(ValueError, match="target"):
        summary._sample()
    setattr(env._sampled_pd500, field, None)
    with pytest.raises(RuntimeError, match="target"):
        summary._sample()


def test_summary_splits_100ms_boundary_and_keeps_late_violations_after_early_crossing():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env()
    summary = ContactSafetySummary(env)
    with summary:
        for _ in range(399):
            env.metrics_manager.compute_substep()
        env.sim.data.qvel[0, 3] = 4.
        env._hammer_substep_impulse.impulse[0, 1] = 1.2
        env.metrics_manager.compute_substep()  # sample400 = exactly100ms
        env.sim.data.qvel[0, 3] = 5.
        env._hammer_substep_impulse.impulse[0, 1] = 1.5
        env.metrics_manager.compute_substep()  # sample401 is late
        env.metrics_manager.compute_substep()
        record = summary.finish_step(torch.tensor([True]))[0]

    velocity = record["joint_velocity"]
    load = record["joint_load"]
    assert velocity["first_crossing_sample"][3] == 400
    assert velocity["early_100ms"]["exceedance_samples"][3] == 1
    assert velocity["late_after_100ms"]["exceedance_samples"][3] == 2
    assert velocity["early_100ms"]["peak_abs_rad_s"][3] == pytest.approx(4.)
    assert velocity["late_after_100ms"]["peak_abs_rad_s"][3] == pytest.approx(5.)
    assert load["early_100ms"]["exceedance_samples"][1] == 1
    assert load["late_after_100ms"]["exceedance_samples"][1] == 2
    assert load["early_100ms"]["peak_nms"][1] == pytest.approx(1.2)
    assert load["late_after_100ms"]["peak_nms"][1] == pytest.approx(1.5)


def test_contact_safety_summary_stops_after_first_reset_without_losing_completed_record():
    from scripts.contact_trace import ContactSafetySummary

    env = _safety_summary_env(2)
    summary = ContactSafetySummary(env)
    with summary:
        env.sim.data.qvel[0, 0] = 3.2
        env._hammer_first_strike.started[0] = True
        env._hammer_substep_impulse.impulse[1, 1] = 1.2
        env.metrics_manager.compute_substep()
        completed = summary.finish_step(torch.tensor([True, False]))

        env.sim.data.qvel[:, 0] = 4.
        env._hammer_substep_impulse.impulse[:, 1] = 2.
        env.metrics_manager.compute_substep()
        second = summary.finish_step(torch.tensor([False, True]))

    assert completed[0]["sample_count"] == 1
    assert completed[1] is None
    assert second[0] is None
    assert second[1]["sample_count"] == 2
    assert second[1]["joint_load"]["peak_nms"][1] == pytest.approx(2.)


def test_trace_cli_requires_sampled_control_and_saves_condition_artifact(tmp_path):
    import evaluate_impact_speed as evaluator

    base = [
        '--task', 'Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed',
        '--speed-bounds', '1.5', '2.2', '--speed-cost', '.8493676',
        '--seed', '7', '--num-envs', '1', '--mode', 'smoke',
        '--output', str(tmp_path / 'trace'),
    ]
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + ['--trace-contact'])
    args = evaluator.parse_args(base + [
        '--sampled-pd500', '--sampled-pd500-gain-mode', 'variable', '--trace-contact',
    ])
    summary = evaluator.run(args)
    assert summary['conditions'][0]['contact_trace'] == 'contact-trace.pt'
    trace_dir = args.output / 'condition-0'
    data = torch.load(trace_dir / 'contact-trace.pt', weights_only=True)
    assert data['valid'].shape == (64 * 80, 1)
    assert data['sample_time_s'][0, 0].item() == pytest.approx(.00025)
    assert data['native_success_sample'].shape == (1,)
    assert data['rolling_joint_load_nms'].shape == (64 * 80, 1, 6)
    assert data['control_step_peak_load_nms'].shape == (64 * 80, 1, 6)
    assert data['off_contact_baseline_nm'].shape == (64 * 80, 1, 6)
    assert data['qfrc_constraint_nm'].shape == (64 * 80, 1, 6)
    assert data['raw_contact'].dtype == torch.bool
    assert data['nail_slide_position_m'].shape == (64 * 80, 1)
    assert data['nail_slide_velocity_mps'].shape == (64 * 80, 1)
    assert data['clamped_nail_depth_m'].shape == (64 * 80, 1)
    assert (trace_dir / 'contact-trace.json').is_file()


@pytest.mark.parametrize('mode', ['fixed', 'variable'])
def test_live_trace_preserves_dynamics_and_stops_at_first_reset(tmp_path, mode):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller, install_impact_speed_controller,
        wrap_impact_speed_env)
    from evaluate_impact_speed import PreResetSnapshot
    from scripts.contact_trace import ContactSafetySummary

    def rollout(record):
        cfg = _impact_speed_env_cfg(play=False)
        cfg.scene.num_envs = 2
        cfg.seed = 41
        cfg.actions['strike_speed'].bounds = (1.5, 2.2)
        cfg.actions['strike_speed'].mode = 'external'
        cfg.rewards['speed_goal'].weight = -.8493676
        configure_impact_speed_controller(cfg, sampled_pd500=True, gain_mode=mode)
        configure_strike_evaluation(cfg)
        env = ManagerBasedRlEnv(cfg, device='cpu')
        try:
            install_impact_speed_controller(env, sampled_pd500=True, gain_mode=mode)
            wrapped = wrap_impact_speed_env(env, 1.)
            wrapped.reset()
            original = env.metrics_manager.compute_substep
            trace = ContactTrace(env) if record else None
            safety = ContactSafetySummary(env) if record else None
            safety_records = [None, None]
            results = []
            with (torch.inference_mode(), PreResetSnapshot(env) as capture,
                  trace if trace is not None else nullcontext(),
                  safety if safety is not None else nullcontext()):
                for step in range(3):
                    if step == 0:
                        env.episode_length_buf[0] = env.max_episode_length - 1
                    action = torch.zeros(2, 13)
                    action[:, 6:12] = .5
                    obs, reward, done, _ = wrapped.step(action)
                    if trace:
                        trace.finish_step(done, capture.latest_evaluation)
                    if safety:
                        for world, item in enumerate(safety.finish_step(done)):
                            if item is not None:
                                safety_records[world] = item
                    results.extend([obs['actor'].clone(), obs['critic'].clone(),
                                    reward.clone(), done.clone(), env.sim.data.qpos.clone(),
                                    env.sim.data.qvel.clone(), env._sampled_pd500.torque.clone()])
            restored = env.metrics_manager.compute_substep
            assert restored.__func__ is original.__func__
            assert restored.__self__ is original.__self__
            if trace:
                trace.save(tmp_path / mode)
                data = torch.load(tmp_path / mode / 'contact-trace.pt', weights_only=True)
                assert data['valid'][:, 0].sum() == 80
                assert data['valid'][:, 1].sum() == 240
                assert data['kp'].shape == (240, 2, 6)
                assert torch.isfinite(data['kp']).all()
                assert data['sample_time_s'][0, 0].item() == pytest.approx(.00025)
                assert data['saturated'].dtype == torch.bool
                assert data['rolling_joint_load_nms'].shape == (240, 2, 6)
                assert data['control_step_peak_load_nms'].shape == (240, 2, 6)
                assert data['off_contact_baseline_nm'].shape == (240, 2, 6)
                assert data['qfrc_constraint_nm'].shape == (240, 2, 6)
                assert data['raw_contact'].dtype == torch.bool
                assert data['native_success_sample'].shape == (2,)
                assert data['joint_load_nms'].shape == (240, 2, 6)
                assert data['requested_torque_nm'].shape == (240, 2, 6)
                assert data['clipped_torque_nm'].shape == (240, 2, 6)
                assert data['joint_position_rad'].shape == (240, 2, 6)
                assert data['desired_joint_position_rad'].shape == (240, 2, 6)
                assert data['joint_position_error_rad'].shape == (240, 2, 6)
                assert data['hammer_head_position_w_m'].shape == (240, 2, 3)
                assert data['spline_reference_position_w_m'].shape == (240, 2, 3)
                assert data['spline_phase'].shape == (240, 2)
                assert data['spline_error_m'].shape == (240, 2)
                assert data['route_sign'].shape == (240, 2)
                assert data['nail_slide_position_m'].shape == (240, 2)
                assert data['nail_slide_velocity_mps'].shape == (240, 2)
                assert data['clamped_nail_depth_m'].shape == (240, 2)
                assert data['accepted_onset'].dtype == torch.bool
                assert data['sample_time_s'][79, 0].item() == pytest.approx(.02)
                assert not data['valid'][80:, 0].any()
                assert not data['kp'][80:, 0].any()
                metadata = json.loads((tmp_path / mode / 'contact-trace.json').read_text())
                assert metadata['site_state_offset_physics_samples'] == -1
                assert 'native post-integration nail_slide qpos/qvel' in metadata['nail_slide_state_semantics']
                assert metadata['controller_gain_mode'] == mode
                with pytest.raises(FileExistsError):
                    trace.save(tmp_path / mode)
                with pytest.raises(RuntimeError, match='observer failure'):
                    with trace:
                        raise RuntimeError('observer failure')
                restored = env.metrics_manager.compute_substep
                assert restored.__func__ is original.__func__
                assert restored.__self__ is original.__self__
                assert safety_records[0]["sample_count"] == 80
                assert safety_records[1] is None
            return results
        finally:
            env.close()

    without, with_trace = rollout(False), rollout(True)
    for a, b in zip(without, with_trace):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

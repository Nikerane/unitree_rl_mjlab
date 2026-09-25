from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.contact_trace import ContactTrace


def test_native_success_marker_is_first_episode_pre_reset_evidence():
    env = SimpleNamespace(
        _sampled_pd500=SimpleNamespace(), _hammer_first_strike=SimpleNamespace(),
        num_envs=2, device='cpu', physics_dt=.00025, max_episode_length=10,
        cfg=SimpleNamespace(decimation=80),
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


def test_trace_cli_requires_sampled_control_and_saves_condition_artifact(tmp_path):
    from scripts import evaluate_impact_speed as evaluator

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
    assert (trace_dir / 'contact-trace.json').is_file()


@pytest.mark.parametrize('mode', ['fixed', 'variable'])
def test_live_trace_preserves_dynamics_and_stops_at_first_reset(tmp_path, mode):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.config.z1 import _impact_speed_env_cfg
    from src.tasks.hammer.mdp.strike_evaluation import configure_strike_evaluation
    from src.tasks.hammer.rl.impact_speed_wrapper import (
        configure_impact_speed_controller, install_impact_speed_controller,
        wrap_impact_speed_env)
    from scripts.evaluate_impact_speed import PreResetSnapshot

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
            results = []
            with torch.inference_mode(), PreResetSnapshot(env) as capture, trace if record else nullcontext():
                for step in range(3):
                    if step == 0:
                        env.episode_length_buf[0] = env.max_episode_length - 1
                    action = torch.zeros(2, 13)
                    action[:, 6:12] = .5
                    obs, reward, done, _ = wrapped.step(action)
                    if trace:
                        trace.finish_step(done, capture.latest_evaluation)
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
                assert data['accepted_onset'].dtype == torch.bool
                assert data['sample_time_s'][79, 0].item() == pytest.approx(.02)
                assert not data['valid'][80:, 0].any()
                assert not data['kp'][80:, 0].any()
                metadata = json.loads((tmp_path / mode / 'contact-trace.json').read_text())
                assert metadata['site_state_offset_physics_samples'] == -1
                assert metadata['controller_gain_mode'] == mode
                with pytest.raises(FileExistsError):
                    trace.save(tmp_path / mode)
                with pytest.raises(RuntimeError, match='observer failure'):
                    with trace:
                        raise RuntimeError('observer failure')
                restored = env.metrics_manager.compute_substep
                assert restored.__func__ is original.__func__
                assert restored.__self__ is original.__self__
            return results
        finally:
            env.close()

    without, with_trace = rollout(False), rollout(True)
    for a, b in zip(without, with_trace):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

import pytest

from scripts.vega_impact_campaign import evaluation_command, run_spec, training_config


def test_paired_matrix_is_bounded():
    runs = [run_spec(i) for i in range(6)]
    assert [(r['gain_mode'], r['seed']) for r in runs] == [
        ('fixed', 10), ('variable', 10), ('fixed', 11),
        ('variable', 11), ('fixed', 12), ('variable', 12)]
    assert all(r['iterations'] == 50 and r['worlds'] == 1024 for r in runs)
    for invalid in (-1, 6):
        with pytest.raises(ValueError):
            run_spec(invalid)


@pytest.mark.parametrize('index', range(6))
def test_training_contract(index):
    spec = run_spec(index)
    cfg = training_config(spec)
    assert cfg.sampled_pd500 and cfg.sampled_pd500_gain_mode == spec['gain_mode']
    assert cfg.speed_mode == 'external' and cfg.speed_bounds == (1.5, 2.2)
    assert cfg.speed_cost == .8493676
    assert cfg.agent.seed == spec['seed'] and cfg.agent.max_iterations == 50
    assert cfg.env.scene.num_envs == 1024
    assert cfg.agent.logger == 'tensorboard'


@pytest.mark.parametrize('smoke', (True, False))
def test_evaluation_command_is_accepted(tmp_path, smoke):
    from scripts.evaluate_impact_speed import parse_args
    command = evaluation_command(tmp_path / 'model.pt', tmp_path / 'eval',
                                 'variable', 'common-window', smoke)
    args = parse_args(command[2:])
    assert args.num_envs == (8 if smoke else 128)
    assert args.goals == (None if smoke else [1.5, 1.85, 2.2])

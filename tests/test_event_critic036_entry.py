"""CPU config/tensor checks only; never instantiate an environment or run PPO."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch


def api():
    return importlib.import_module('scripts.z1_event_critic036')


@pytest.fixture(autouse=True)
def forbid_physics_and_loading(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*args, **kwargs):
        raise AssertionError('real environment/checkpoint loading forbidden')
    monkeypatch.setattr(ManagerBasedRlEnv, '__init__', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)


def test_exact_four_recipe_major_groups_three_common_new_seeds():
    m = api()
    recipes = [('split', 'vic_baseline'), ('200', 'vic_baseline'),
               ('split', 'vic_position_low'), ('50', 'vic_baseline')]
    for i in range(12):
        cell = m.cell_for_index(i)
        assert (cell['payout'], cell['setting']) == recipes[i // 3]
        assert cell['seed'] == (2026110201, 2026110202, 2026110203)[i % 3]
        assert cell['critic_mode'] == 'event'
        assert (cell['updates'], cell['worlds'], cell['rollout_steps']) == (500, 1024, 24)
    assert m.SAVE_INDICES == (0, 24, 49, 99, 249, 499)
    from scripts import z1_critic_noise_training021 as inherited
    assert (m.UPDATES,m.WORLDS,m.STEPS,m.SAVE_INDICES) == (inherited.UPDATES,inherited.WORLDS,inherited.STEPS,inherited.SAVE_INDICES)
    for bad in (True, -1, 12, '0', 0., None):
        with pytest.raises(ValueError):
            m.cell_for_index(bad)


def test_configs_reuse_factories_and_preserve_plant_ppo_rewards():
    m = api()
    from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_env_cfg, event_task_id
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_rl_cfg
    from tests.test_spline_speed_variants import config_value
    for i in range(12):
        cell = m.cell_for_index(i)
        task, cfg = m.build_train_config(i)
        expected = event_env_cfg(play=False, payout=cell['payout'], setting=cell['setting'])
        expected.scene.num_envs = 1024
        expected.seed = cell['seed']
        assert task == event_task_id(payout=cell['payout'], setting=cell['setting'])
        assert config_value(asdict(cfg.env)) == config_value(asdict(expected))
        agent = pilot_rl_cfg(setting=cell['setting'], critic_mode='event')
        agent.seed = cell['seed']
        agent.max_iterations = 500
        agent.resume = False
        agent.save_interval = 1
        agent.logger = 'tensorboard'
        assert asdict(cfg.agent) == asdict(agent)
        assert cfg.sampled_pd500 and cfg.sampled_pd500_gain_mode == 'variable'
        assert cfg.speed_bounds == (1., 1.5) and cfg.speed_cost == .8493676
        assert cfg.speed_mode == 'policy' and cfg.qualification_checkpoint is None
        assert cfg.enable_nan_guard


def test_cli_calls_existing_trainer_once_inside_gates(monkeypatch, tmp_path):
    m = api()
    from scripts import train
    events = []
    task, cfg = object(), object()
    monkeypatch.setattr(m, 'build_train_config', lambda i: (task, cfg) if i == 8 else None)
    def context(name):
        @contextmanager
        def gate(index, output):
            assert index == 8 and output == tmp_path / 'run'
            events.append(name + ' enter')
            yield []
            events.append(name + ' exit')
        return gate
    monkeypatch.setattr(m, 'initial_state_gate', context('initial'))
    monkeypatch.setattr(m, 'selected_checkpoint_saves', context('saves'))
    def run_train(actual_task, actual_cfg, output):
        assert actual_task is task and actual_cfg is cfg
        assert output == tmp_path / 'run' / 'train'
        events.append('train')
    monkeypatch.setattr(train, 'run_train', run_train)
    m.main(['--index', '8', '--output', str(tmp_path / 'run')])
    assert events == ['initial enter', 'saves enter', 'train', 'saves exit', 'initial exit']
    with pytest.raises(FileExistsError):
        m.main(['--index', '8', '--output', str(tmp_path / 'run')])


def test_initial_gate_explicitly_disables_trace_and_historical_pairing(monkeypatch, tmp_path):
    m = api()
    seen = []
    monkeypatch.setattr(m, '_initial_gate', lambda *a, **kw: seen.append((a, kw)) or 'gate')
    for index in (0, 3, 6, 9):
        assert m.initial_state_gate(index, tmp_path) == 'gate'
        args, kwargs = seen[-1]
        assert args == (m.cell_for_index(index), tmp_path)
        assert kwargs['enable_trace'] is False and kwargs['reference_initial_state'] is None
        assert kwargs['recipe'] is m._live_recipe
        assert kwargs['marker_key'] == ('spline_split_event036' if index in (0, 6) else 'spline_exploration_pilot')


def test_initial_tensor_evidence_and_same_seed_model_matching(monkeypatch, tmp_path):
    m = api()
    from tests.test_z1_exploration_pilot_training import runner
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    monkeypatch.setattr(HammerOnPolicyRunner, 'learn', lambda *args: 'intercepted')
    rows = []
    for i in range(12):
        cell = m.cell_for_index(i)
        r = runner(1 if cell['setting'] == 'vic_position_low' else 0, 'event')
        r.cfg.update(seed=cell['seed'], resume=False)
        r.env.unwrapped.cfg.seed = cell['seed']
        marker = dict(critic_width=47)
        key = 'spline_split_event036' if cell['payout'] == 'split' else 'spline_exploration_pilot'
        monkeypatch.setattr(m, '_live_recipe', lambda r, c, key=key: {key: marker})
        def forbidden(*args, **kwargs):
            raise AssertionError('initialization must not run forward or observations')
        r.env.get_observations = r.env.unwrapped.get_observations = forbidden
        monkeypatch.setattr(r.alg.actor, 'forward', forbidden)
        monkeypatch.setattr(r.alg.critic, 'forward', forbidden)
        with m.initial_state_gate(i, None) as evidence:
            assert HammerOnPolicyRunner.learn(r, 500) == 'intercepted'
        row = evidence[0]
        rows.append(row)
        assert row['critic_width'] == 47
        assert row['model_initialization_seed'] == row['collection_seed'] == cell['seed']
        assert row['initial_observations']['actor_shape'] == [1024, 43]
        assert row['initial_observations']['critic_shape'] == [1024, 47]
        assert row['initial_std'] == pytest.approx([.1]*6+[.3]*6+[1.] if i // 3 == 2 else [.3]*12+[1.])
        assert {'actor', 'critic', *m.COMMON_INITIAL_HASHES} <= row['hashes'].keys()
        first = next(layer for layer in r.alg.critic.mlp if isinstance(layer, torch.nn.Linear))
        assert torch.count_nonzero(first.weight[:, 43:]) == 0
        if i >= 3:
            for name in m.COMMON_INITIAL_HASHES:
                assert row['hashes'][name] == rows[i % 3]['hashes'][name]


@pytest.mark.parametrize('index',[0,3,6,9])
def test_real_021_initial_gate_output_feeds_study_record_reader(monkeypatch,tmp_path,index):
    m = api()
    from tests.test_z1_exploration_pilot_training import runner
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    cell = m.cell_for_index(index)
    r = runner(1 if cell['setting'] == 'vic_position_low' else 0,'event')
    r.cfg.update(seed=cell['seed'],resume=False)
    r.env.unwrapped.cfg.seed = cell['seed']
    metadata = evaluation_record(index)['impact_speed_contract']
    monkeypatch.setattr(m,'_live_recipe',lambda *a: metadata)
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *a: None)
    with m.initial_state_gate(index,tmp_path) as rows:
        HammerOnPolicyRunner.learn(r,500)
    assert len(rows) == 1
    for label in (99,499):
        checkpoint = tmp_path/f'model_{label}.pt'; checkpoint.write_bytes(b'no checkpoint deserialization')
        record = eval_api().record_for_checkpoint(index,checkpoint,tmp_path/'initial-state.json')
        assert record['impact_speed_contract'] == metadata and record['common_step_counter'] == (label+1)*24


def eval_api():
    return importlib.import_module('scripts.z1_event_critic036_eval')


def evaluation_record(index=0, label=99):
    m = api()
    from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_task_id
    cell = m.cell_for_index(index)
    if cell['payout'] == 'split':
        from src.tasks.hammer.rl.spline_split_event036_contract import split_event_identity
        marker = split_event_identity(cell['setting'])
        contract = {'spline_split_event036': marker}
    else:
        from src.tasks.hammer.rl.spline_exploration_pilot_contract import _identity
        from src.tasks.hammer.rl.spline_credit_critic_contract import experiment_identity
        experiment = experiment_identity(cell['credit_deadline_ms'], 'event')
        contract = {'spline_exploration_pilot': _identity(cell['setting'], experiment),
                    'credit_critic_experiment': experiment}
    contract.update(schema_version='impact_speed_v1', observation_widths={'actor':43, 'critic':47},
                    physics_dt_s=.00025, control_decimation=80, speed_bounds_mps=[1.,1.5],
                    speed_mode='policy', action_dim=13,
                    action_type='joint_position_variable_impedance_sampled_pd', controller_identity='sampled_vic500_v1')
    return dict(schema='z1_event_critic036_policy_v1', policy_id=f'cell-{index:02d}-model-{label}',
                training_cell=cell, checkpoint_label=label, completed_updates=label+1,
                common_step_counter=(label+1)*24,
                task=event_task_id(payout=cell['payout'], setting=cell['setting']),
                checkpoint_path=f'/test/policy-{index}/train/model_{label}.pt', checkpoint_sha256='a'*64,
                initial_state_sha256='b'*64, impact_speed_contract=contract)


def test_evaluation_new_identity_both_clocks_and_exact_bank():
    m = eval_api()
    assert m.BANK_SHA == 'b0baebc9dee774c89091a16ed7ba6ce5bfb3df4e4a639b6faabf56ec9da43f22'
    assert m.EVALUATION_SEED == 2026103001
    for index in range(12):
        for label in (99, 499):
            record = evaluation_record(index, label)
            assert m.validate_record(record) == record
            for field, value in (('checkpoint_label', 98), ('common_step_counter', 0), ('training_cell', m.__dict__.get('bad', {}))):
                bad = deepcopy(record)
                bad[field] = value
                with pytest.raises(ValueError):
                    m.validate_record(bad)


def test_evaluation_full_checkpoint_tensor_validation_before_frontend():
    m = eval_api()
    from scripts.z1_thirty_evaluation023.records import expected_state_shapes
    record = evaluation_record(3, 499)
    payload = dict(iter=499, infos=dict(env_state=dict(common_step_counter=12000),
                                     impact_speed_contract=record['impact_speed_contract']),
                   actor_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(43, True).items()},
                   critic_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(47).items()})
    assert m.validate_payload(payload, record)['status'] == 'PASS_PAYLOAD_ONLY'
    negative = deepcopy(payload)
    negative['actor_state_dict']['distribution.std_param'].fill_(-.25)
    audited = m.validate_payload(negative, record)
    assert audited['raw_gaussian_std_parameter'] == [-.25]*13
    assert audited['effective_gaussian_std_after_clamp'] == pytest.approx([1e-6]*13)
    for fault in ('iter', 'counter', 'metadata', 'critic_width', 'nonfinite', 'actor_missing', 'scheduler'):
        bad = deepcopy(payload)
        if fault == 'iter': bad['iter'] = 99
        elif fault == 'counter': bad['infos']['env_state']['common_step_counter'] = 2400
        elif fault == 'metadata': bad['infos']['impact_speed_contract']['speed_mode'] = 'external'
        elif fault == 'critic_width': bad['critic_state_dict']['mlp.0.weight'] = torch.ones(256,43)
        elif fault == 'nonfinite': bad['critic_state_dict']['mlp.0.weight'][0,0] = float('nan')
        elif fault == 'actor_missing': del bad['actor_state_dict']['distribution.std_param']
        else: bad['infos']['cat_pressure_treatment'] = {}
        with pytest.raises(ValueError):
            m.validate_payload(bad, record)


def test_live_split_recipe_checks_exact_cat_worlds_clock_seed_and_models(monkeypatch):
    m = api()
    from tests.test_split_event036 import event_env
    from tests.test_spline_exploration_pilot_contract import models
    from src.tasks.hammer.rl import impact_speed_contract
    for index in (0, 6):
        cell = m.cell_for_index(index)
        env = event_env(cell['setting'])
        # The synthetic managers hold three initial tensor rows; this recipe
        # checks the selected world count without simulating manager buffers.
        env.num_envs = 1024
        env.cfg.seed = cell['seed']
        a, c = models(cell['setting'], 'event')
        from src.tasks.hammer.rl.spline_split_event036_contract import split_event_identity
        metadata = {'spline_split_event036': split_event_identity(cell['setting'])}
        monkeypatch.setattr(impact_speed_contract, 'build_impact_speed_metadata', lambda *a: metadata)
        r = SimpleNamespace(env=SimpleNamespace(unwrapped=env, clip_actions=1.), alg=SimpleNamespace(actor=a,critic=c),
                            cfg={'seed':cell['seed'], 'num_steps_per_env':24})
        assert m._live_recipe(r, cell) == metadata
        hook = env.metrics_manager.cfg['cat_soft'].func
        for name, value in (('_max_p', .4), ('_imp_max_p', .1), ('_velocity_pressure', object()),
                            ('_use_vel', False), ('_use_impulse', False)):
            old = getattr(hook, name)
            setattr(hook, name, value)
            with pytest.raises(ValueError): m._live_recipe(r, cell)
            setattr(hook, name, old)
        for key, value in (('seed', cell['seed']+1), ('num_steps_per_env',23)):
            old = r.cfg[key]
            r.cfg[key] = value
            with pytest.raises(ValueError): m._live_recipe(r, cell)
            r.cfg[key] = old


def test_evaluation_binding_reuses_bank_lifecycle_but_checks_dynamic_counter(monkeypatch):
    m = eval_api()
    from src.tasks.hammer.rl import impact_speed_contract
    calls = []
    monkeypatch.setattr(impact_speed_contract, 'validate_impact_speed_models', lambda *a: calls.append('models'))
    class Qualified:
        def bind(self, *args): calls.append('qualified')
    class LegacyPilot(Qualified):
        def bind(self, *args): raise AssertionError('legacy record/counter2400 bind must be bypassed')
    pilot = SimpleNamespace(make_adapter_class=lambda base: LegacyPilot)
    adapter = m.make_adapter_class(Qualified, pilot)()
    for index, label in ((0,99),(3,499),(6,499),(9,99)):
        record = evaluation_record(index,label)
        adapter.pilot_record = record
        env = SimpleNamespace(num_envs=32,device='cpu',common_step_counter=(label+1)*24)
        runner = SimpleNamespace(alg=SimpleNamespace(actor=object(),critic=object()))
        adapter.bind(env, runner, record['impact_speed_contract'], record['impact_speed_contract'])
        assert calls[-2:] == ['models','qualified']
        env.common_step_counter = 0
        with pytest.raises(ValueError): adapter.bind(env, runner, record['impact_speed_contract'], record['impact_speed_contract'])


def test_evaluation_cli_delegates_one_checkpoint_without_running_rollout(monkeypatch, tmp_path):
    m = eval_api()
    seen = []
    monkeypatch.setattr(m, 'evaluate', lambda *args, **kwargs: seen.append((args,kwargs)) or 'result')
    result = m.main(['--index','9','--checkpoint','/test/model_499.pt','--initial-state','/test/initial-state.json',
                     '--output',str(tmp_path/'eval'),'--deadline-epoch','1234'])
    assert result == 'result' and len(seen) == 1
    args, kwargs = seen[0]
    assert args == (9,Path('/test/model_499.pt'),Path('/test/initial-state.json'),tmp_path/'eval')
    assert kwargs == dict(deadline_epoch=1234.,max_output_bytes=128*1024**2)


def test_preserved_frontend_import_bank_config_and_constructor_only(tmp_path):
    m = eval_api()
    pilot, qualified, evaluator, bank, pins = m._load_frontend()
    assert m._sha(m.BANK) == m.BANK_SHA
    record = evaluation_record(6,499)
    from scripts.z1_thirty_evaluation023.integration import adapt_family_configuration
    adapt_family_configuration(evaluator, m.VENDOR/'pilot-compat001/qualified007/current_evaluator_candidate.py',
                               m._selected_configuration(record))
    assert evaluator.pilot_condition_adaptation_counts == {'balanced_route_expected_signs':1}
    assert evaluator.thirty_configuration_adaptation_counts == {'training_factory_config':1}
    factory = evaluator._thirty_config
    m.cap_technical_steps(evaluator)
    assert evaluator._thirty_config is factory
    assert '_thirty_config' in evaluator._condition.__code__.co_names
    assert 'expected_route_signs' in evaluator._condition.__code__.co_names
    assert evaluator.pilot_condition_adaptation_counts == {'balanced_route_expected_signs':1}
    assert evaluator.thirty_configuration_adaptation_counts == {'training_factory_config':1}
    context = dict(policy=record, reset_bank=dict(path=str(m.BANK),sha256=m.BANK_SHA),
                   output_root=str(tmp_path), deadline_epoch=1., max_output_bytes=128*1024**2)
    adapter = m.make_adapter_class(qualified.HorizonAdapter,pilot)(context,tmp_path,bank)
    assert adapter.pilot_record is record
    assert adapter.intent['policy']['native_window_s'] == .2
    assert adapter.pilot_balanced_bank and adapter.expected_route_signs == (-1,0,1)
    assert adapter.started_counts == {} and adapter.observer is None
    # These are pure factory loads; do not bind, enter, reset, or call _condition.
    env_cfg, agent_cfg = evaluator._thirty_config(record['task'])
    assert env_cfg.scene.num_envs == 1024 and env_cfg.seed == 2026110201
    assert env_cfg.episode_length_s == 20. and agent_cfg.max_iterations == 500


def test_invalid_checkpoint_fails_before_frontend_or_output_creation(monkeypatch, tmp_path):
    m = eval_api()
    record = evaluation_record(0,99)
    monkeypatch.setattr(m, 'record_for_checkpoint', lambda *args: record)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    events = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda *a: events.append('threads'))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *a: events.append('interop'))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: events.append('load') or {'iter':499})
    def forbidden(): raise AssertionError('frontend reached before full checkpoint validation')
    monkeypatch.setattr(m, '_load_frontend', forbidden)
    with pytest.raises(ValueError, match='actual checkpoint label differs'):
        m.evaluate(0,'/test/model_99.pt','/test/initial-state.json',tmp_path/'eval',deadline_epoch=m.time.time()+10)
    assert not (tmp_path/'eval').exists()
    assert events == ['threads','interop','load']


def technical_record(index=0):
    row = evaluation_record(index, 99)
    row.update(schema='z1_event_critic036_technical_policy_v1', policy_id=f'technical-cell-{index:02d}-model-0',
               checkpoint_label=0, completed_updates=1, common_step_counter=24,
               checkpoint_path=f'/test/technical-{index}/train/model_0.pt')
    row['training_cell']['updates'] = 1
    return row


def test_technical_identity_never_admitted_as_a_study_checkpoint():
    m = eval_api()
    for index in (0, 3):
        row = technical_record(index)
        assert m.validate_record(row, technical=True) == row
        with pytest.raises(ValueError): m.validate_record(row)
        with pytest.raises(ValueError): m.validate_record(evaluation_record(index), technical=True)
        for key, val in (('checkpoint_label',99),('common_step_counter',2400),('completed_updates',100)):
            bad = deepcopy(row); bad[key] = val
            with pytest.raises(ValueError): m.validate_record(bad, technical=True)


def test_technical_record_uses_actual_model0_initial_evidence_and_payload(tmp_path):
    import json
    m = eval_api()
    from scripts.z1_thirty_evaluation023.records import expected_state_shapes
    row = technical_record(3)
    checkpoint = tmp_path/'model_0.pt'
    checkpoint.write_bytes(b'actual technical model0')
    initial = dict(schema='z1_event_critic036_technical_initial_v1', cell=row['training_cell'],
                   phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',actual_counter=0,critic_width=47,
                   model_initialization_seed=2026110201,collection_seed=2026110201,
                   hashes={name:'a'*64 for name in ('actor','critic',*api().COMMON_INITIAL_HASHES)},
                   initial_observations={'actor_shape':[1024,43],'critic_shape':[1024,47]},
                   contract=row['impact_speed_contract'])
    initial_path = tmp_path/'initial-state.json'
    initial_path.write_text(json.dumps(initial))
    actual = m.record_for_checkpoint(3,checkpoint,initial_path,technical=True)
    assert actual['checkpoint_label'] == 0 and actual['common_step_counter'] == 24
    with pytest.raises(ValueError): m.record_for_checkpoint(3,checkpoint,initial_path)
    payload = dict(iter=0,infos={'env_state':{'common_step_counter':24},'impact_speed_contract':actual['impact_speed_contract']},
                   actor_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(43,True).items()},
                   critic_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(47).items()})
    assert m.validate_payload(payload,actual,technical=True)['common_step_counter'] == 24
    with pytest.raises(ValueError): m.validate_payload(payload,actual)


def test_technical_loop_cap_is_exactly_three_steps_without_running_a_frontend():
    import ast
    m = eval_api()
    source = '''def _condition(args, goal, index):
    steps = 1013
    seen = []
    for step in range(steps):
        seen.append(step)
    return seen
'''
    frontend = SimpleNamespace(thirty_condition_source=source,
        thirty_configuration_adaptation_counts={'training_factory_config':1},
        pilot_condition_adaptation_counts={'balanced_route_expected_signs':1})
    m.cap_technical_steps(frontend)
    assert frontend._condition(None,None,0) == [0,1,2]
    assert frontend.technical_step_cap == 3
    with pytest.raises(ValueError): m.cap_technical_steps(frontend)
    with pytest.raises(ValueError): m.cap_technical_steps(SimpleNamespace(thirty_condition_source='def _condition(a,b,c): return []'))
    with pytest.raises(ValueError): m.cap_technical_steps(SimpleNamespace(thirty_condition_source=source+'\nEXTRA = 1\n',
        thirty_configuration_adaptation_counts={'training_factory_config':1},
        pilot_condition_adaptation_counts={'balanced_route_expected_signs':1}))


@pytest.mark.parametrize('inference_buffers',[False,True])
def test_technical_check_one_mocked_update_zero_and_full_reload(monkeypatch, tmp_path, inference_buffers):
    m = importlib.import_module('scripts.z1_event_critic036_check')
    from scripts import train
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from tests.test_z1_exploration_pilot_training import runner
    r = runner(0,'event')
    r.cfg.update(seed=2026110201,resume=False)
    r.env.unwrapped.cfg.seed = 2026110201
    cell = api().cell_for_index(3)
    payload = {}
    calls = []
    metadata = {'spline_exploration_pilot':{'critic_width':47}}
    monkeypatch.setattr(api(), '_live_recipe', lambda *args: metadata)
    def learn(actual, n, init_at_random_ep_len=False):
        assert actual is r and n == 1 and init_at_random_ep_len
        r.env.unwrapped.common_step_counter = 24
        if inference_buffers:
            # PPO rollout normalization can replace registered buffers inside
            # inference_mode. Initialization itself precedes this transition.
            with torch.inference_mode():
                for model in (r.alg.actor,r.alg.critic):
                    for module in model.modules():
                        for name,buffer in module._buffers.items():
                            if buffer is not None: module._buffers[name] = buffer.clone()
            assert all(buffer.is_inference() for model in (r.alg.actor,r.alg.critic) for buffer in model.buffers())
        payload.update(iter=0,infos={'impact_speed_contract':metadata,'env_state':{'common_step_counter':24}},
                       actor_state_dict=deepcopy(r.alg.actor.state_dict()),critic_state_dict=deepcopy(r.alg.critic.state_dict()))
        (tmp_path/'check'/'train').mkdir()
        (tmp_path/'check'/'train'/'model_0.pt').write_bytes(b'actual technical model0')
        calls.append('learn')
    def load(actual, path, *, load_cfg, strict):
        assert actual is r and load_cfg == {'actor':True,'critic':True} and strict
        assert torch.is_inference_mode_enabled()
        assert all(torch.count_nonzero(p) == 0 for model in (r.alg.actor,r.alg.critic) for p in model.parameters())
        assert r.env.unwrapped.common_step_counter == 17
        assert all((buffer == 17).all() for model in (r.alg.actor,r.alg.critic) for buffer in model.buffers())
        r.alg.actor.load_state_dict(payload['actor_state_dict'])
        r.alg.critic.load_state_dict(payload['critic_state_dict'])
        r.env.unwrapped.common_step_counter = payload['infos']['env_state']['common_step_counter']
        calls.append('reload')
        return payload['infos']
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    monkeypatch.setattr(HammerOnPolicyRunner,'load',load)
    r.load = lambda path, **kwargs: HammerOnPolicyRunner.load(r,path,**kwargs)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs: payload)
    def run_train(task,cfg,output):
        assert cfg.env.scene.num_envs == 1024 and cfg.agent.max_iterations == 1 and cfg.agent.save_interval == 1
        assert output == tmp_path/'check'/'train'
        calls.append('train')
        HammerOnPolicyRunner.learn(r,1,True)
    monkeypatch.setattr(train,'run_train',run_train)
    result = m.run_check(3,tmp_path/'check')
    assert calls == ['train','learn','reload']
    assert result['completed_updates'] == 1 and result['common_step_counter'] == 24
    assert result['status'] == 'PASS_UPDATE_SAVE_FULL_RELOAD_ONLY'
    assert result['loaded_model_components'] == ['actor','critic'] and result['study_trials'] == 0
    assert HammerOnPolicyRunner.learn is learn
    assert (tmp_path/'check'/'train'/'model_0.pt').read_bytes() == b'actual technical model0'
    with pytest.raises(FileExistsError): m.run_check(3,tmp_path/'check')
    assert calls == ['train','learn','reload']


def test_eval_module_disables_vendor_bytecode_even_without_python_B():
    import subprocess
    import sys
    code = '''import sys, types
sys.dont_write_bytecode = False
training = types.ModuleType('scripts.z1_event_critic036')
records = types.ModuleType('scripts.z1_thirty_evaluation023.records')
records.finite_tree = records.validate_state_shapes = lambda *a, **k: None
sys.modules[training.__name__] = training
sys.modules[records.__name__] = records
import scripts.z1_event_critic036_eval
assert sys.dont_write_bytecode is True
assert 'pilot_integration' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)


def test_numeric_audit_envelope_source_provenance_and_single_call(monkeypatch,tmp_path):
    import json
    m = eval_api()
    from scripts.z1_thirty_evaluation023 import qualified as q
    context = dict(schema='z1_event_critic036_measurement_context_v1',policy=evaluation_record(),
                   source_files=m.measurement_source_files(),deadline_epoch=m.time.time()+30,max_output_bytes=128*1024**2)
    assert len(context['source_files']) == 1
    source = context['source_files'][0]
    assert source['path'].endswith('/src/tasks/hammer/mdp/impulse_bound.py')
    assert source['sha256'] == m._sha(source['path'])
    reference = {'path':str(tmp_path/'mapping.json'),'sha256':'a'*64}
    worker = dict(status='RAW_REVIEW_REQUIRED',reports=[dict(case_id='pilot-bank32',report_sha256='b'*64,
                  observer_mapping=reference,native_onset_trace=reference,safety_endpoint_summary=reference)])
    calls = []
    dependency = object()
    def audit(actual_dependency,actual_context,output,bank):
        assert actual_dependency is dependency and actual_context is context and output == tmp_path
        assert json.loads((tmp_path/'worker-result.json').read_text()) == worker
        assert len(bank['rows']) == 32
        calls.append('audit')
        return {'denominators':{'all_first_episodes':32},'worlds':[]}
    reader = SimpleNamespace(dependencies=lambda: dependency,audit_measurements=audit)
    monkeypatch.setattr(q,'measurement_reader',lambda: reader)
    writer = SimpleNamespace(durable=lambda p,v: p.write_text(json.dumps(v)))
    receipt = m.write_numeric_audit(writer,context,tmp_path,worker)
    assert calls == ['audit'] and receipt['status'] == 'PASS_FILE_NUMERIC_AUDIT_ONLY'
    assert receipt['scientific_qualification'] is False and receipt['safety_certified'] is False
    other = tmp_path/'technical'; other.mkdir()
    receipt = m.write_numeric_audit(writer,context,other,worker,technical=True)
    assert calls == ['audit'] and receipt['status'] == 'SKIPPED_TECHNICAL_BINDING_ONLY'


def test_actual_file_reader_abi_loads_without_decoding_or_environment():
    import inspect
    from scripts.z1_thirty_evaluation023.qualified import measurement_reader
    reader = measurement_reader()
    assert tuple(inspect.signature(reader.audit_measurements).parameters) == ('w','intent','root','bank')
    source = Path(reader.__file__).read_text()
    assert "worker-result.json" in source and "intent['source_files']" in source
    assert 'native_window_s=.2' in source
    dependencies = reader.dependencies()
    assert callable(dependencies.audit_case_measurements) and callable(dependencies.read_trace)

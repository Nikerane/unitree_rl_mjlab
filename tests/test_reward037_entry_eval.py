"""Reward037 entry and preserved evaluator checks; no simulator or PPO execution."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def entry():
    return importlib.import_module('scripts.z1_reward_training037')


def evaluator():
    return importlib.import_module('scripts.z1_event_critic036_eval')


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    def forbidden(*args, **kwargs):
        raise AssertionError('these checks must not construct an environment or deserialize a checkpoint')
    monkeypatch.setattr(ManagerBasedRlEnv, '__init__', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)


def policy(index=0, label=99, *, technical=False):
    from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_task_id
    from src.tasks.hammer.rl.spline_reward037_contract import reward_identity
    cell = entry().cell_for_index(index)
    if technical:
        cell['updates'] = 1
    contract = dict(schema_version='impact_speed_v1', observation_widths={'actor':43, 'critic':47},
                    physics_dt_s=.00025, control_decimation=80, speed_bounds_mps=[1.,1.5],
                    speed_mode='policy', action_dim=13,
                    action_type='joint_position_variable_impedance_sampled_pd', controller_identity='sampled_vic500_v1',
                    spline_reward037=reward_identity(cell['setting'],height_dr=cell.get('height_dr',False)))
    return dict(schema='z1_reward037_technical_policy_v1' if technical else 'z1_reward037_policy_v1',
                policy_id=('technical-' if technical else '') + f'cell-{index:02d}-model-{label}',
                training_cell=cell, checkpoint_label=label, completed_updates=label+1,
                common_step_counter=(label+1)*24, task=reward_task_id(cell['setting'],height_dr=cell.get('height_dr',False)),
                checkpoint_path=f'/test/reward037-{index}/train/model_{label}.pt', checkpoint_sha256='a'*64,
                initial_state_sha256='b'*64, impact_speed_contract=contract)


def test_six_cells_select_only_all_reward_split_event_and_the_three_reused_seeds():
    m = entry()
    for index in range(6):
        cell = m.cell_for_index(index)
        assert cell == dict(index=index, payout='split', setting='vic_baseline' if index < 3 else 'vic_position_low',
                            seed=(2026110201,2026110202,2026110203)[index%3], critic_mode='event',
                            updates=500, worlds=1024, rollout_steps=24,
                            early_fraction=.5, early_deadline_ms=50, reserve_deadline_ms=200)
    for bad in (True,-1,8,'0',0.,None):
        with pytest.raises(ValueError): m.cell_for_index(bad)
    assert {99,499} <= set(m.SAVE_INDICES)


def test_height_pilots_are_two_additional_cells_without_changing_the_six_base_identities():
    from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_env_cfg,reward_task_id
    from dataclasses import asdict
    from tests.test_spline_speed_variants import config_value
    for index,setting in ((6,'vic_baseline'),(7,'vic_position_low')):
        cell=entry().cell_for_index(index)
        assert cell == dict(index=index,payout='split',setting=setting,seed=2026110201,
                            critic_mode='event',updates=500,worlds=1024,rollout_steps=24,
                            early_fraction=.5,early_deadline_ms=50,reserve_deadline_ms=200,height_dr=True)
        task,cfg=entry().build_train_config(index)
        assert task == reward_task_id(setting,height_dr=True) and task != reward_task_id(setting)
        expected=reward_env_cfg(play=False,setting=setting,height_dr=True)
        expected.scene.num_envs=1024;expected.seed=2026110201
        assert config_value(asdict(cfg.env)) == config_value(asdict(expected))
        assert cfg.agent.seed == 2026110201 and cfg.agent.max_iterations == 500
        assert evaluator().validate_record(policy(index),study='reward037')['training_cell']['height_dr'] is True
        bad=policy(index);bad['training_cell'].pop('height_dr')
        with pytest.raises(ValueError):evaluator().validate_record(bad,study='reward037')
        bad=policy(index);bad['task']=reward_task_id(setting)
        with pytest.raises(ValueError):evaluator().validate_record(bad,study='reward037')


def test_factories_are_fresh_and_use_new_reward_task_with_unchanged_ppo():
    from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_env_cfg, reward_task_id
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_rl_cfg
    from dataclasses import asdict
    from tests.test_spline_speed_variants import config_value
    for index in range(6):
        cell = entry().cell_for_index(index)
        task, cfg = entry().build_train_config(index)
        assert task == reward_task_id(cell['setting'])
        expected = reward_env_cfg(play=False,setting=cell['setting'])
        expected.seed = cell['seed']; expected.scene.num_envs = 1024
        assert config_value(asdict(cfg.env)) == config_value(asdict(expected))
        agent = pilot_rl_cfg(setting=cell['setting'],critic_mode='event')
        agent.seed=cell['seed']; agent.max_iterations=500; agent.resume=False; agent.save_interval=1; agent.logger='tensorboard'
        assert asdict(cfg.agent) == asdict(agent)
        assert cfg.sampled_pd500 and cfg.sampled_pd500_gain_mode == 'variable'
        assert cfg.speed_bounds == (1.,1.5) and cfg.speed_cost == .8493676 and cfg.speed_mode == 'policy'
        assert cfg.qualification_checkpoint is None and cfg.enable_nan_guard


def test_entry_calls_existing_trainer_once_with_both_existing_gates(monkeypatch,tmp_path):
    m = entry()
    from scripts import train
    events=[]
    @contextmanager
    def gate(index, output):
        assert index == 5 and output == tmp_path/'run'
        events.append('enter'); yield; events.append('exit')
    task, cfg = object(), object()
    monkeypatch.setattr(m,'build_train_config',lambda index:(task,cfg))
    monkeypatch.setattr(m,'initial_state_gate',gate)
    monkeypatch.setattr(m,'selected_checkpoint_saves',gate)
    def run(actual_task,actual_cfg,output):
        assert actual_task is task and actual_cfg is cfg and output == tmp_path/'run/train'
        events.append('train')
    monkeypatch.setattr(train,'run_train',run)
    m.main(['--index','5','--output',str(tmp_path/'run')])
    assert events == ['enter','enter','train','exit','exit']
    with pytest.raises(FileExistsError): m.main(['--index','5','--output',str(tmp_path/'run')])


def test_real_initial_gate_keeps_fresh_rng_std_and_zero_event_columns_without_claiming_historical_pairing(monkeypatch,tmp_path):
    m = entry()
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from tests.test_z1_exploration_pilot_training import runner
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',lambda *args:'passed')
    for index in (0,3):
        cell=m.cell_for_index(index); r=runner(index//3,'event')
        r.cfg.update(seed=cell['seed'],resume=False);r.env.unwrapped.cfg.seed=cell['seed']
        monkeypatch.setattr(m,'_live_recipe',lambda *args:{'spline_reward037':{'critic_width':47}})
        out=tmp_path/f'cell-{index}';out.mkdir()
        with m.initial_state_gate(index,out) as rows:
            assert HammerOnPolicyRunner.learn(r,500) == 'passed'
        row=rows[0]
        assert row['contract'] == {'spline_reward037':{'critic_width':47}}
        assert row['initial_std'] == pytest.approx([.3]*12+[1.] if index == 0 else [.1]*6+[.3]*6+[1.])
        assert row['initial_observations']['actor_shape'] == [1024,43]
        assert row['initial_observations']['critic_shape'] == [1024,47]
        first=next(layer for layer in r.alg.critic.mlp if isinstance(layer,torch.nn.Linear))
        assert torch.count_nonzero(first.weight[:,43:]) == 0
        assert json.loads((out/'initial-state.json').read_text())['cell'] == cell


def test_reward037_records_admit_only_selected_schema_cell_task_and_reward_identity():
    m = evaluator()
    assert m.BANK_SHA == 'b0baebc9dee774c89091a16ed7ba6ce5bfb3df4e4a639b6faabf56ec9da43f22'
    for index in range(6):
        for label in (99,499):
            row=policy(index,label)
            assert m.validate_record(row,study='reward037') is row
            with pytest.raises(ValueError):m.validate_record(row)
            for key,value in (('schema','z1_event_critic036_policy_v1'),('task','legacy-task'),('checkpoint_label',0),('common_step_counter',0)):
                bad=deepcopy(row);bad[key]=value
                with pytest.raises(ValueError):m.validate_record(bad,study='reward037')
            bad=deepcopy(row);bad['impact_speed_contract']['spline_split_event036']={}
            with pytest.raises(ValueError):m.validate_record(bad,study='reward037')
            bad=deepcopy(row);bad['impact_speed_contract']['spline_reward037']['setting']='fixed_nominal'
            with pytest.raises(ValueError):m.validate_record(bad,study='reward037')
    with pytest.raises(ValueError):m.validate_record(policy(),study='arbitrary_module')


def test_actual_initial_evidence_feeds_new_record_and_full_tensor_validation(tmp_path):
    m=evaluator();row=policy(3,499)
    initial=dict(cell=row['training_cell'],phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',actual_counter=0,critic_width=47,
                 model_initialization_seed=2026110201,collection_seed=2026110201,
                 hashes={name:'a'*64 for name in ('actor','critic',*entry().COMMON_INITIAL_HASHES)},
                 initial_observations={'actor_shape':[1024,43],'critic_shape':[1024,47]},contract=row['impact_speed_contract'])
    initial_path=tmp_path/'initial-state.json';initial_path.write_text(json.dumps(initial))
    checkpoint=tmp_path/'train/model_499.pt';checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'no deserialization')
    from tests.test_event_critic036_entry import write_checkpoint_receipt
    write_checkpoint_receipt(checkpoint,row,study='reward037')
    actual=m.record_for_checkpoint(3,checkpoint,initial_path,study='reward037')
    from scripts.z1_thirty_evaluation023.records import expected_state_shapes
    payload=dict(iter=499,infos={'env_state':{'common_step_counter':12000},'impact_speed_contract':actual['impact_speed_contract']},
                 actor_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(43,True).items()},
                 critic_state_dict={k:torch.ones(v) for k,v in expected_state_shapes(47).items()})
    assert m.validate_payload(payload,actual,study='reward037')['status'] == 'PASS_PAYLOAD_ONLY'
    bad=deepcopy(payload);bad['critic_state_dict']['mlp.0.weight']=torch.ones(256,43)
    with pytest.raises(ValueError):m.validate_payload(bad,actual,study='reward037')
    payload['infos']['impact_speed_contract']={}
    with pytest.raises(ValueError):m.validate_payload(payload,actual,study='reward037')


def test_technical_scope_stays_distinct_from_500_update_study():
    m=evaluator();row=policy(0,0,technical=True)
    assert m.validate_record(row,technical=True,study='reward037') is row
    with pytest.raises(ValueError):m.validate_record(row,study='reward037')
    with pytest.raises(ValueError):m.validate_record(policy(),technical=True,study='reward037')
    selected=m._selected_configuration(row,study='reward037')
    env,agent=selected(row['task'])
    assert agent.max_iterations == 1 and env.scene.num_envs == 1024
    with pytest.raises(ValueError):selected(row['task'])


def test_new_study_cli_delegates_to_shared_evaluator_and_technical_checker(monkeypatch,tmp_path):
    m=evaluator();calls=[]
    monkeypatch.setattr(m,'evaluate',lambda *a,**kw:calls.append((a,kw)) or 'evaluated')
    assert m.main(['--study','reward037','--index','0','--checkpoint','/test/model_99.pt',
                   '--initial-state','/test/initial-state.json','--output',str(tmp_path/'eval'),
                   '--deadline-epoch','1234']) == 'evaluated'
    assert calls[0][1] == dict(study='reward037',deadline_epoch=1234.,max_output_bytes=128*1024**2)
    check=importlib.import_module('scripts.z1_event_critic036_check')
    monkeypatch.setattr(check,'run_check',lambda *a,**kw:calls.append((a,kw)) or 'checked')
    assert check.main(['--study','reward037','--index','3','--output',str(tmp_path/'check')]) == 'checked'
    assert calls[-1] == ((3,tmp_path/'check'),{'study':'reward037'})


@pytest.fixture
def isolated_preserved_imports():
    before=dict(sys.modules);search_path=sys.path[:]
    yield
    vendor=evaluator().VENDOR.resolve()
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name not in before and path and Path(path).resolve().is_relative_to(vendor):
            sys.modules.pop(name,None)
    sys.path[:]=search_path


def test_preserved_frontend_accepts_reward037_factory_and_adapter_without_construction(tmp_path,isolated_preserved_imports):
    m=evaluator();row=policy(3,499)
    pilot,qualified,frontend,bank,pins=m._load_frontend()
    from scripts.z1_thirty_evaluation023.integration import adapt_family_configuration
    adapt_family_configuration(frontend,m.VENDOR/'pilot-compat001/qualified007/current_evaluator_candidate.py',
                               m._selected_configuration(row,study='reward037'))
    env,agent=frontend._thirty_config(row['task'])
    assert env.seed == 2026110201 and env.scene.num_envs == 1024 and agent.max_iterations == 500
    assert env.rewards['delivered_impulse'].weight > 0
    context=dict(policy=row,reset_bank={'path':str(m.BANK),'sha256':m.BANK_SHA},output_root=str(tmp_path),
                 deadline_epoch=1.,max_output_bytes=128*1024**2)
    adapter=m.make_adapter_class(qualified.HorizonAdapter,pilot,study='reward037')(context,tmp_path,bank)
    assert adapter.pilot_record is row and adapter.intent['policy']['native_window_s'] == .2
    assert adapter.expected_route_signs == (-1,0,1) and adapter.started_counts == {} and adapter.observer is None
    m.cap_technical_steps(frontend)
    assert frontend.technical_step_cap == 3 and frontend.technical_step_cap_counts == {'policy_step_loop_cap':1}


def test_reward037_technical_check_saves_and_reloads_both_models_with_own_schema(monkeypatch,tmp_path):
    check=importlib.import_module('scripts.z1_event_critic036_check')
    from scripts import train
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from tests.test_z1_exploration_pilot_training import runner
    r=runner(1,'event');r.cfg.update(seed=2026110201,resume=False);r.env.unwrapped.cfg.seed=2026110201
    metadata={'spline_reward037':{'critic_width':47}}
    monkeypatch.setattr(entry(),'_live_recipe',lambda *args:metadata)
    payload={};events=[]
    def learn(actual,n,init_at_random_ep_len=False):
        assert actual is r and n == 1
        r.env.unwrapped.common_step_counter=24
        payload.update(iter=0,infos={'impact_speed_contract':metadata,'env_state':{'common_step_counter':24}},
                       actor_state_dict=deepcopy(r.alg.actor.state_dict()),critic_state_dict=deepcopy(r.alg.critic.state_dict()))
        (tmp_path/'check/train').mkdir()
        (tmp_path/'check/train/model_0.pt').write_bytes(b'technical model0')
        events.append('one_update')
    def load(path,*,load_cfg,strict):
        assert load_cfg == {'actor':True,'critic':True} and strict and torch.is_inference_mode_enabled()
        assert r.env.unwrapped.common_step_counter == 17
        for name in ('actor','critic'):
            assert all(torch.count_nonzero(p) == 0 for p in getattr(r.alg,name).parameters())
            getattr(r.alg,name).load_state_dict(payload[name+'_state_dict'])
        r.env.unwrapped.common_step_counter=24;events.append('reload')
    r.load=load
    monkeypatch.setattr(HammerOnPolicyRunner,'learn',learn)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:payload)
    def train_once(task,cfg,output):
        assert cfg.agent.max_iterations == 1 and cfg.agent.save_interval == 1
        assert output == tmp_path/'check/train'
        HammerOnPolicyRunner.learn(r,1)
    monkeypatch.setattr(train,'run_train',train_once)
    result=check.run_check(3,tmp_path/'check',study='reward037')
    assert events == ['one_update','reload']
    assert result['schema'] == 'z1_reward037_technical_result_v1'
    assert result['study_trials'] == 0 and result['completed_updates'] == 1 and result['common_step_counter'] == 24
    initial=json.loads((tmp_path/'check/initial-state.json').read_text())
    assert initial['schema'] == 'z1_reward037_technical_initial_v1' and initial['cell']['updates'] == 1
    assert initial['contract'] == metadata and initial['initial_std'] == pytest.approx([.1]*6+[.3]*6+[1.])
    assert HammerOnPolicyRunner.learn is learn


def body_position_fixture(num_worlds=3):
    """Tensor-backed native DR input with nontrivial global body indexing."""
    from mjlab.envs import mdp
    from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_env_cfg
    event=deepcopy(reward_env_cfg(play=False,setting='vic_baseline',height_dr=True).events['randomize_board_nail_height037'])
    event.params['asset_cfg'].body_ids=[0,1]
    defaults=torch.zeros(9,3);defaults[3]=torch.tensor([.5,0.,.05]);defaults[7]=torch.tensor([.5,0.,.1])
    model=SimpleNamespace(body_pos=defaults.unsqueeze(0).expand(num_worlds,-1,-1).clone())
    asset=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([3,7],dtype=torch.int32)),body_names=['block','nail'],
                          data=SimpleNamespace(joint_pos=torch.zeros(num_worlds,1),site_pos_w=torch.zeros(num_worlds,1,3)))
    env=SimpleNamespace(num_envs=num_worlds,device='cpu',common_step_counter=0,scene={'nail_block':asset},
        sim=SimpleNamespace(model=model,get_default_field=lambda name:defaults,per_world_default_fields=set()),
        cfg=SimpleNamespace(events={'randomize_board_nail_height037':deepcopy(event)}),
        event_manager=SimpleNamespace(cfg={'randomize_board_nail_height037':event},get_term_cfg=lambda name:event))
    env._hammer_first_strike=SimpleNamespace(_nail_site_ids=[0])
    env._strike_reference=SimpleNamespace(_nail=torch.zeros(num_worlds,3),_target=torch.zeros(num_worlds,3),_anchored=torch.ones(num_worlds,dtype=torch.bool),overshoot=.02)
    def reset():
        mdp.dr.body_pos(env,None,**event.params)
        delta=model.body_pos[:,3,2]-defaults[3,2]
        asset.data.site_pos_w[:,0,:]=torch.stack((torch.full_like(delta,.5),torch.zeros_like(delta),.102+delta),-1)
        env._strike_reference._nail.copy_(asset.data.site_pos_w[:,0])
        env._strike_reference._target.copy_(env._strike_reference._nail)
        env._strike_reference._target[:,2]-=.02
    return env,event,reset


def test_nominal_height_overlay_uses_native_reset_restores_callback_and_preserves_rng():
    m=evaluator();env,event,reset=body_position_fixture();original=event.func
    original(env,None,**event.params)
    assert bool((env.sim.model.body_pos[:,3,2] != .05).any())
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    with m.nominal_height_reset(env) as evidence:
        assert event.func is not original and event.func.recompute == original.recompute
        assert event.params['ranges'] == (-.005,.005)
        event.func(env,None,**event.params)
        assert torch.equal(torch.get_rng_state(),before)
        assert torch.allclose(env.sim.model.body_pos[:,[3,7]],env.sim.get_default_field('body_pos')[[3,7]].expand(3,-1,-1))
        assert evidence['reset_calls'] == 1 and evidence['world_reset_counts'] == {'0':1,'1':1,'2':1}
    assert event.func is original and evidence['callback_restored'] is True
    assert evidence['evaluation_offset_m'] == 0. and evidence['height_robustness_evaluation'] is False
    with pytest.raises(RuntimeError):
        with m.nominal_height_reset(env):raise RuntimeError('test cleanup')
    assert event.func is original


@pytest.mark.parametrize('fault',[None,'independent','range','site','reference','nail_qpos'])
def test_technical_height_checker_reads_two_resets_and_rejects_geometry_drift(fault):
    check=importlib.import_module('scripts.z1_event_critic036_check')
    env,event,native_reset=body_position_fixture();calls=[]
    def reset():
        native_reset();calls.append(True)
        if fault == 'independent':env.sim.model.body_pos[:,7,2]+=.001
        elif fault == 'range':env.sim.model.body_pos[:,[3,7],2]+=.1
        elif fault == 'site':env.scene['nail_block'].data.site_pos_w[:,0,2]+=.001
        elif fault == 'reference':env._strike_reference._nail[:,2]+=.001
        elif fault == 'nail_qpos':env.scene['nail_block'].data.joint_pos.fill_(.001)
    runner=SimpleNamespace(env=SimpleNamespace(unwrapped=env,reset=reset))
    if fault:
        with pytest.raises(ValueError):check.height_reset_evidence(runner)
    else:
        row=check.height_reset_evidence(runner)
        assert len(calls) == 2 and row['status'] == 'PASS_TWO_NATIVE_HEIGHT_RESETS_ONLY'
        assert row['global_body_ids'] == [3,7] and len(row['resets']) == 2
        assert row['training_updates'] == 0 and row['physics_steps'] == 0


def test_height_receipt_covers_all1024_worlds_with_hashes_and_bounded_sample_under_existing_writer_cap(tmp_path):
    check=importlib.import_module('scripts.z1_event_critic036_check')
    from scripts.z1_credit_critic_training import _write_once,tensor_state_digest
    env,event,reset=body_position_fixture(1024)
    runner=SimpleNamespace(env=SimpleNamespace(unwrapped=env,reset=reset))
    row=check.height_reset_evidence(runner)
    path=tmp_path/'height-reset-technical.json'
    _write_once(path,row)
    assert path.stat().st_size < 64*1024
    assert json.loads(path.read_text()) == row
    assert row['verified_worlds'] == 1024
    for result in row['resets']:
        assert result['all_world_summary']['world_count'] == 1024
        assert result['sample_worlds']['world_ids'] == [0,1,2,3]
        assert len(result['sample_worlds']['height_offset_m']) == 4
        assert result['sample_worlds']['scope'] == 'EXPLICIT_SAMPLE_ONLY_NOT_ALL_WORLDS'
        assert result['all_world_summary']['max_shared_z_shift_error_m'] <= 1e-7
        assert result['all_world_summary']['max_nail_site_error_m'] <= 2e-6
        assert all(len(digest) == 64 for digest in result['all_world_tensor_sha256'].values())
    final=row['resets'][-1]['all_world_tensor_sha256']
    assert final['body_positions_m'] == tensor_state_digest({'body_positions_m':env.sim.model.body_pos[:,[3,7]]})
    assert final['nail_top_world_m'] == tensor_state_digest({'nail_top_world_m':env.scene['nail_block'].data.site_pos_w[:,0]})
    # A fault beyond the four displayed worlds still fails the all-world check.
    def bad_reset():
        reset();env._strike_reference._nail[1023,2]+=.001
    runner.env.reset=bad_reset
    with pytest.raises(ValueError,match='site/reference'):check.height_reset_evidence(runner)

"""Finite twelve-cell pilot adapter; no CLI, submission or admission authority."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re

from scripts.z1_credit_critic_training import (
    _directory, _write_once, _file_sha, tensor_state_digest,
    _normalizer_hash,
)
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_SETTINGS

PILOT_SEEDS = (2026102902, 2026102903, 2026102904)
UPDATES, WORLDS, STEPS = 100, 1024, 24
SAVE_INDICES = (0,24,49,99)
COMMON_INITIAL_HASHES = ('actor_shared_non_std','shared_critic43','actor_normalizer','critic_normalizer',
                        'collection_torch_rng','collection_python_rng','collection_numpy_rng','collection_cuda_rng')
_PATCHED, _REGISTERED_ANCHORS = set(), set()


def pilot_cell_for_index(index, *, credit_deadline_ms=200, critic_mode='current'):
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import _require_anchor
    _require_anchor(credit_deadline_ms,critic_mode)
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('pilot index must be integer0 through11')
    return dict(index=index,setting=PILOT_SETTINGS[index%4],seed=PILOT_SEEDS[index//4],
                credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode,
                updates=UPDATES,worlds=WORLDS,rollout_steps=STEPS)


def build_pilot_train_config(index, *, credit_deadline_ms=200, critic_mode='current'):
    cell=pilot_cell_for_index(index,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import register_pilot_tasks,pilot_task_id
    anchor=(credit_deadline_ms,critic_mode)
    if anchor not in _REGISTERED_ANCHORS:
        register_pilot_tasks(credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode)
        _REGISTERED_ANCHORS.add(anchor)
    task=pilot_task_id(setting=cell['setting'],credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode)
    cfg=replace(TrainConfig.from_task(task),sampled_pd500=True,
        sampled_pd500_gain_mode='fixed' if cell['setting']=='fixed_nominal' else 'variable',
        speed_bounds=(1.,1.5),speed_cost=.8493676,speed_mode='policy',
        qualification_checkpoint=None,enable_nan_guard=True)
    cfg.env.scene.num_envs=WORLDS;cfg.env.seed=cfg.agent.seed=cell['seed']
    cfg.agent.max_iterations=UPDATES;cfg.agent.resume=False
    cfg.agent.save_interval=1;cfg.agent.logger='tensorboard'
    return task,cfg


def _live_recipe(runner,cell):
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.spline_exploration_pilot_contract import validate_pilot_models
    from src.tasks.hammer.rl.spline_credit_critic_contract import experiment_identity
    env=runner.env.unwrapped;hook=env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p != .5 or hook._imp_max_p != .2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse recipe differs')
    metadata=build_impact_speed_metadata(env,runner.env.clip_actions)
    marker=metadata.get('spline_exploration_pilot')
    if (not isinstance(marker,dict) or marker.get('setting') != cell['setting']
            or marker.get('credit_critic_experiment') != experiment_identity(cell['credit_deadline_ms'],cell['critic_mode'])):
        raise ValueError('live pilot treatment/anchor differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs != WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('live worlds, rollout clock or paired seed differs')
    validate_pilot_models(runner.alg.actor,runner.alg.critic,metadata)
    return metadata


def _initial_observation_hashes(env,marker):
    import torch
    obs=getattr(env,'obs_buf',None)
    if obs is None or set(obs.keys()) != {'actor','critic'}:
        raise ValueError('existing initial observation buffer must contain actor and critic')
    for group,width in (('actor',43),('critic',marker['critic_width'])):
        value=obs[group]
        if (not isinstance(value,torch.Tensor) or value.shape!=(WORLDS,width)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('existing initial observations require finite correctly shaped tensors')
    return dict(source='existing_env_obs_buf_no_recompute',actor_shape=list(obs['actor'].shape),
        critic_shape=list(obs['critic'].shape),actor_sha256=tensor_state_digest({'observation':obs['actor']}),
        critic_shared43_sha256=tensor_state_digest({'observation':obs['critic'][:,:43]}),
        scope='saved observations only; not full physical-state identity')


def _reject_storage_aliases(models):
    import torch
    intervals=[]
    for model in models:
        for value in model.state_dict().values():
            if not isinstance(value,torch.Tensor) or value.layout != torch.strided or value.device.type=='meta':
                raise ValueError('matched initialization requires inspectable tensor storage')
            storage=value.untyped_storage()
            if storage.nbytes():intervals.append((str(value.device),storage.data_ptr(),storage.data_ptr()+storage.nbytes()))
    intervals.sort()
    for a,b in zip(intervals,intervals[1:]):
        if a[0]==b[0] and b[1]<a[2]:raise ValueError('matched initialization forbids aliased storage')


def _initialize(runner,cell):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    from src.tasks.hammer.rl.spline_exploration_pilot import PilotBaselineActor,initialize_matched_pilot_actor
    from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic,_check_model,_prepared_state
    device=torch.device(runner.device)
    if device.type not in ('cpu','cuda'):raise ValueError('initialization supports onlycpu/allocatedCUDA')
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cell['seed'])
        x=torch.zeros(1,43,device=device)
        obs=TensorDict({'actor':x,'critic':x.clone()},batch_size=[1]);groups={'actor':['actor'],'critic':['critic']}
        keys=('hidden_dims','activation','obs_normalization')
        ac=deepcopy({k:runner.cfg['actor'][k] for k in keys})
        ac['distribution_cfg']=deepcopy(runner.cfg['actor']['distribution_cfg'])
        expected='src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian'
        if not isinstance(ac['distribution_cfg'],dict) or ac['distribution_cfg'].get('class_name',expected)!=expected:
            raise ValueError('canonical pilot reference requires its exact Gaussian')
        ac['distribution_cfg']['class_name']=expected
        cc=deepcopy({k:runner.cfg['critic'][k] for k in keys})
        cc['distribution_cfg']=deepcopy(runner.cfg['critic'].get('distribution_cfg'))
        reference_actor=PilotBaselineActor(obs,groups,'actor',13,**ac).to(device)
        reference_critic=ImpactSpeedAgeModel(obs,groups,'critic',1,**cc).to(device)
    a,c=runner.alg.actor,runner.alg.critic;event=cell['critic_mode']=='event'
    # Reuse only the strict deterministic-critic validation/preparation primitives;
    # the legacy helper's actor path intentionally does not admit pilot models.
    _check_model(c,(CreditEventCritic,) if event else (ImpactSpeedAgeModel,),'critic',47 if event else 43)
    _check_model(reference_critic,(ImpactSpeedAgeModel,),'critic',43)
    prepared_critic=_prepared_state(c,reference_critic,extend_critic=event)
    _reject_storage_aliases((a,c,reference_actor,reference_critic))
    # The complete critic is validated before the actor helper's first write.
    initialize_matched_pilot_actor(a,reference_actor)
    c.load_state_dict(prepared_critic,strict=True)
    initial_std=a.distribution.std_param.detach()
    expected_std=initial_std.new_tensor(([.1]*6+[.3]*6+[1.]) if cell['setting']=='vic_position_low' else [.3]*12+[1.])
    if not torch.equal(initial_std,expected_std):raise ValueError('actual initial pilot std differs')
    def without_std(model):return {k:v for k,v in model.state_dict().items() if k!='distribution.std_param'}
    actor_shared=tensor_state_digest(without_std(a))
    state=c.state_dict();name=next(n for n,l in c.mlp.named_modules() if isinstance(l,torch.nn.Linear));key=f'mlp.{name}.weight'
    if event and torch.count_nonzero(state[key][:,43:]):raise ValueError('initial event critic columns must be zero')
    shared=dict(state);shared[key]=shared[key][:,:43];critic_shared=tensor_state_digest(shared)
    if actor_shared!=tensor_state_digest(without_std(reference_actor)) or critic_shared!=tensor_state_digest(reference_critic.state_dict()):
        raise ValueError('actual initial non-treatment model state differs from canonical reference')
    random.seed(cell['seed']);np.random.seed(cell['seed']);torch.manual_seed(cell['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cell['seed'])
    ns=np.random.get_state();serial=[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])]
    cuda=torch.cuda.get_rng_state(device).cpu().numpy().tobytes() if device.type=='cuda' else None
    return dict(actor=tensor_state_digest(a.state_dict()),actor_shared_non_std=actor_shared,
        shared_critic43=critic_shared,critic=tensor_state_digest(c.state_dict()),
        actor_normalizer=_normalizer_hash(a),critic_normalizer=_normalizer_hash(c),
        collection_torch_rng=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        collection_python_rng=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        collection_numpy_rng=hashlib.sha256(json.dumps(serial,allow_nan=False).encode()).hexdigest(),
        collection_cuda_rng=hashlib.sha256(cuda).hexdigest() if cuda is not None else None),expected_std.tolist()


def _mismatch(output,row,reference,reason):
    if output is not None:
        # Store bounded actual/expected fingerprints, not arbitrary caller payloads.
        def fields(value,keys):
            value=value if isinstance(value,dict) else {}
            return {k:v if v is None or (type(v) is str and len(v)<=128) else {'invalid_reference_type':type(v).__name__}
                    for k in keys for v in [value.get(k)]}
        ref=reference if isinstance(reference,dict) else {}
        _write_once(output/'initial-state-mismatch.json',dict(status='INITIAL_STATE_REFERENCE_MISMATCH_NO_TRAINING',
            reason=reason,actual_initial_state=row,expected_hashes=fields(ref.get('hashes'),COMMON_INITIAL_HASHES),
            expected_initial_observations=fields(ref.get('initial_observations'),('actor_sha256','critic_shared43_sha256'))))
    raise ValueError(reason)


@contextmanager
def matched_pilot_initial_state_gate(index,output_dir=None,*,credit_deadline_ms=200,critic_mode='current',
                                     enable_trace=True,reference_initial_state=None):
    cell=pilot_cell_for_index(index,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode);output=_directory(output_dir)
    if type(enable_trace) is not bool or (enable_trace and output is None):raise ValueError('both traces require explicit boolean and output directory')
    if output is not None and any((output/n).exists() or (output/n).is_symlink() for n in ('initial-state.json','initial-state-mismatch.json')):
        raise FileExistsError('initial-state evidence already consumed')
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'learn' in _PATCHED:raise ValueError('pilot initial gate already active')
    original=HammerOnPolicyRunner.learn;evidence=[]
    def learn(runner,num_learning_iterations,init_at_random_ep_len=False):
        if (evidence or type(num_learning_iterations) is not int or num_learning_iterations!=UPDATES
                or type(runner.current_learning_iteration) is not int or runner.current_learning_iteration!=0
                or type(runner.env.unwrapped.common_step_counter) is not int or runner.env.unwrapped.common_step_counter!=0
                or runner.alg.optimizer.state):raise ValueError('fresh once-only finite learning and empty optimizer required')
        metadata=_live_recipe(runner,cell);marker=metadata['spline_exploration_pilot']
        observed=_initial_observation_hashes(runner.env.unwrapped,marker);hashes,std=_initialize(runner,cell)
        row=dict(cell=cell,phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT',actual_counter=0,critic_width=marker['critic_width'],
            contract=metadata,hashes=hashes,initial_std=std,initial_observations=observed,
            model_initialization_seed=cell['seed'],collection_seed=cell['seed'],
            live_cat=dict(velocity_pmax=.5,impulse_pmax=.2,scheduler=False))
        if reference_initial_state is not None:
            ref=reference_initial_state;expected=pilot_cell_for_index(index//4*4,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode)
            if (not isinstance(ref,dict) or ref.get('cell')!=expected or ref.get('phase')!=row['phase']
                    or type(ref.get('actual_counter')) is not int or ref['actual_counter']!=0):
                _mismatch(output,row,ref,'common initial baseline reference identity differs')
            ref_hashes=ref.get('hashes')
            if not isinstance(ref_hashes,dict) or any(hashes[k]!=ref_hashes.get(k) for k in COMMON_INITIAL_HASHES):
                _mismatch(output,row,ref,'common initial non-treatment model/RNG fingerprints differ')
            ref_observed=ref.get('initial_observations')
            if not isinstance(ref_observed,dict) or any(observed[k]!=ref_observed.get(k) for k in ('actor_sha256','critic_shared43_sha256')):
                _mismatch(output,row,ref,'common initial observed state differs')
        if output is not None:_write_once(output/'initial-state.json',row)
        evidence.append(row)
        with ExitStack() as stack:
            if enable_trace:
                from src.tasks.hammer.rl.credit_training_trace import attach_credit_training_trace
                from src.tasks.hammer.rl.velocity_population_trace import attach_velocity_population_trace
                stack.enter_context(attach_credit_training_trace(runner,output/'training-trace.jsonl',max_updates=UPDATES))
                stack.enter_context(attach_velocity_population_trace(runner,output/'velocity-population.jsonl',max_updates=UPDATES))
            return original(runner,num_learning_iterations,init_at_random_ep_len)
    _PATCHED.add('learn');HammerOnPolicyRunner.learn=learn
    try:
        yield evidence
        if len(evidence)!=1:raise ValueError('training must pass pilot gate exactly once')
    finally:HammerOnPolicyRunner.learn=original;_PATCHED.remove('learn')


@contextmanager
def selected_pilot_checkpoint_saves(index,output_dir=None,*,credit_deadline_ms=200,critic_mode='current'):
    cell=pilot_cell_for_index(index,credit_deadline_ms=credit_deadline_ms,critic_mode=critic_mode);output=_directory(output_dir)
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    if 'save' in _PATCHED:raise ValueError('pilot checkpoint filter already active')
    original=HammerOnPolicyRunner.save;evidence=[];seen={};owner=[]
    def save(runner,path,infos=None):
        path=Path(path);match=re.fullmatch(r'model_(\d+)\.pt',path.name)
        if match is None:raise ValueError('unexpected checkpoint filename')
        label=int(match.group(1))
        if label not in range(UPDATES):raise ValueError('checkpoint outside finite allocation')
        if label not in SAVE_INDICES:return None
        if output is not None and path!=output/'train'/path.name:raise ValueError('checkpoint outside exact pilot train directory')
        counter=runner.env.unwrapped.common_step_counter
        if (type(counter) is not int or counter!=(label+1)*STEPS or type(runner.current_learning_iteration) is not int
                or runner.current_learning_iteration!=label):raise ValueError('checkpoint transition clock differs')
        if owner and owner[0] is not runner:raise ValueError('checkpoint context belongs to another runner')
        metadata=_live_recipe(runner,cell);saved_infos=dict(infos or {})
        if 'impact_speed_contract' in saved_infos and saved_infos['impact_speed_contract']!=metadata:
            raise ValueError('checkpoint full pilot metadata differs')
        saved_infos['impact_speed_contract']=metadata
        hashes={k:tensor_state_digest(getattr(runner.alg,k).state_dict()) for k in ('actor','critic')}
        if label in seen:
            prior=seen[label]
            if (label!=UPDATES-1 or prior.get('deduplicated') or prior['hashes']!=hashes or prior['checkpoint_sha256']!=_file_sha(path)):
                raise ValueError('unexpected repeated checkpoint save')
            prior['deduplicated']=True
            row=dict(iteration=label,common_step_counter=counter,hashes=hashes,checkpoint_sha256=prior['checkpoint_sha256'],deduplicated_final_save=True)
            if output is not None:_write_once(output/f'checkpoint-{label}-final-save-deduplicated.json',row)
            evidence.append(row);return None
        if path.exists() or path.is_symlink():raise FileExistsError('checkpoint output is consumed')
        result=original(runner,str(path),saved_infos)
        row=dict(iteration=label,common_step_counter=counter,hashes=hashes,checkpoint_sha256=_file_sha(path),deduplicated_final_save=False)
        if output is not None:_write_once(output/f'checkpoint-{label}-live.json',row)
        seen[label]=dict(row);owner[:]=[runner];evidence.append(row)
        return result
    _PATCHED.add('save');HammerOnPolicyRunner.save=save
    try:yield evidence
    finally:HammerOnPolicyRunner.save=original;_PATCHED.remove('save')

"""Closed B500 recovery study, independent of frozen factorial identities."""
from copy import copy,deepcopy
import json
from mjlab.managers.manager_base import ManagerTermBase
from src.tasks.hammer.mdp.impact_speed import ImpactSpeedCost
from . import safety_factorial_campaign as base

MARKER='recovery_budget'
CELLS=('N0','Nplus','Eplus','C01','C10')
CONTACT_CELLS={'C01':.1,'C10':1.}
SEEDS=base.SEEDS
CAPS=base.CAPS


def identity(cell):
    if type(cell) is not str or cell not in CELLS:raise ValueError('unknown recovery cell')
    result=base.identity('B')
    result.update(schema='z1_recovery_budget_v1',cell_id=cell,scope=MARKER,
        contact_horizon_s=.5,critic_contact_age_cap=2.5,
        recovery_budget=0. if cell=='N0' else .1,recovery_installments=15,
        recovery_rate=0. if cell=='N0' else 1/3,
        recovery_payment='D200<ending_tick<=D200+1200; native_started; terminal_before_reset',
        recovery_gate='native_first_strike_started_only_no_productivity_or_geometry_gate',
        recovery_transform='positive_producer_once_preserve_negative',
        entropy_coefficient=.01 if cell=='Eplus' else .02,
        reset_distribution='original_exact_U22',historical_parent='B200')
    if cell in CONTACT_CELLS:
        result.update(recovery_budget=0.,recovery_rate=0.,contact_bonus=CONTACT_CELLS[cell],
            contact_bonus_radius_m=.010,contact_bonus_distance='Euclidean_3D_hammer_head_site_at_first_physical_contact',
            contact_bonus_endpoint='reference._nail_frozen_after_reset_before_first_action_not_followthrough_target',
            contact_bonus_gate='first_physical_hammer_nail_sensor_contact_only_no_arming_order_velocity_or_productivity_gate',
            contact_bonus_consumption='first_contact_consumes_including_offtarget_no_recontact_payment',
            contact_bonus_payment='bonus_divided_by_policy_dt_weight1_rewardmanager_integrates_once_before_CaT',
            contact_bonus_transform='positive_producer_once_preserve_negative')
    return result


def validate_identity(value):
    if type(value) is not dict or json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(identity(value.get('cell_id')),sort_keys=True,allow_nan=False):
        raise ValueError('recovery identity differs from closed recipe')
    return identity(value['cell_id'])


class RecoverySpeedCost(ImpactSpeedCost):
    def __init__(self,cfg,env):
        if configured(env.cfg) is None or cfg.weight!=0. or cfg.params!={}:
            raise ValueError('recovery requires its scoped zero speed reader')
        ManagerTermBase.__init__(self,env)


def weights(marker):
    result=dict(base.weights(base.identity('B')),recovery_budget=1.)
    if marker['cell_id'] in CONTACT_CELLS:result['contact_endpoint_bonus']=1.
    return result


def configure(cfg,cell):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from src.tasks.hammer.mdp import recovery_budget as mdp
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    marker=identity(cell)
    if getattr(cfg,MARKER,None) is not None:raise ValueError('duplicate recovery profile')
    base.configure(cfg,'B');delattr(cfg,base.MARKER)
    cfg.rewards['speed_goal'].func=RecoverySpeedCost
    cfg.rewards['recovery_budget']=RewardTermCfg(func=mdp.RecoveryBudgetReward,weight=1.,params={'budget':marker['recovery_budget']})
    cfg.terminations['nail_driven'].func=mdp.recovery_window_complete
    cfg.terminations['nail_driven'].params={}
    cfg.metrics['native_depth_owner031'].func=mdp.RecoveryDepthObserver
    cfg.metrics['recovery_budget_accounting']=MetricsTermCfg(func=mdp.accounting_metric,reduce='last')
    for field in CREDIT_EVENT_FIELDS:cfg.observations['critic'].terms[field].func=mdp.event_observation
    if cell in CONTACT_CELLS:
        from src.tasks.hammer.mdp import contact_endpoint_bonus as contact
        cfg.rewards['contact_endpoint_bonus']=RewardTermCfg(func=contact.ContactEndpointBonus,weight=1.,params={'bonus':marker['contact_bonus']})
        robot=deepcopy(cfg.rewards['accepted_spline_strike'].params['robot_cfg'])
        cfg.metrics['contact_endpoint_onset']=MetricsTermCfg(func=contact.ContactEndpointOnset,per_substep=True,reduce='last',
            params={'robot_cfg':robot,'contact_sensor_name':'hammer_nail_contact','width_m':.010})
        cfg.metrics['contact_endpoint_bonus_accounting']=MetricsTermCfg(func=contact.accounting_metric,reduce='last')
    setattr(cfg,MARKER,marker);configured(cfg)
    return cfg


def configured(cfg):
    from src.tasks.hammer.mdp import recovery_budget as mdp
    from src.tasks.hammer.mdp.spline_credit_deadline import CREDIT_EVENT_FIELDS
    from src.tasks.hammer.mdp.split_impulse import event_observation
    from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
    from src.tasks.hammer.config.z1.spline_exploration038_cfg import ExplorationDepthObserver038
    from .exploration038_contract import _raw_term
    value=getattr(cfg,MARKER,None)
    if value is None:return None
    marker=validate_identity(value)
    if any(getattr(cfg,key,None) is not None for key in ('safety_factorial','split_impulse_campaign','reward_grid_campaign','demo_progress_campaign')):
        raise ValueError('recovery cannot mix historical campaign identities')
    reward=cfg.rewards.get('recovery_budget');terminal=cfg.terminations['nail_driven'];account=cfg.metrics.get('recovery_budget_accounting')
    if (set(cfg.rewards)!=set(weights(marker)) or reward.func is not mdp.RecoveryBudgetReward
            or reward.weight!=1. or reward.params!={'budget':marker['recovery_budget']}
            or terminal.func is not mdp.recovery_window_complete or terminal.params!={} or terminal.time_out is not False
            or cfg.metrics['native_depth_owner031'].func is not mdp.RecoveryDepthObserver
            or cfg.rewards['speed_goal'].func is not RecoverySpeedCost
            or account is None or account.func is not mdp.accounting_metric or account.params!={}
            or account.per_substep is not False or account.reduce!='last'
            or list(cfg.metrics).index('recovery_budget_accounting')<=list(cfg.metrics).index('cat_soft')
            or any(not _raw_term(cfg.observations['critic'].terms[f],mdp.event_observation,{'field':f}) for f in CREDIT_EVENT_FIELDS)):
        raise ValueError('recovery reward/horizon/observer/critic registration differs')
    if marker['cell_id'] in CONTACT_CELLS:_validate_contact_config(cfg,marker)
    elif any(name in cfg.metrics for name in ('contact_endpoint_onset','contact_endpoint_bonus_accounting')):
        raise ValueError('historical recovery cannot register contact bonus')
    view=copy(cfg)
    for key in ('rewards','terminations','metrics','observations'):setattr(view,key,deepcopy(getattr(cfg,key)))
    delattr(view,MARKER);setattr(view,base.MARKER,base.identity('B'))
    del view.rewards['recovery_budget'];del view.metrics['recovery_budget_accounting']
    if marker['cell_id'] in CONTACT_CELLS:
        del view.rewards['contact_endpoint_bonus'];del view.metrics['contact_endpoint_onset'];del view.metrics['contact_endpoint_bonus_accounting']
    view.rewards['speed_goal'].func=base.FactorialSpeedCost
    view.terminations['nail_driven'].func=horizon_window_complete;view.terminations['nail_driven'].params={'window_s':.2}
    view.metrics['native_depth_owner031'].func=ExplorationDepthObserver038
    for field in CREDIT_EVENT_FIELDS:view.observations['critic'].terms[field].func=event_observation
    base.configured(view)
    return marker


def live(env):
    marker=configured(env.cfg)
    if marker is not None and (env._sampled_pd500.gain_mode!='variable' or env._sampled_pd500.startup_ramp_s!=0.
            or getattr(env,'_u22_reference_clock',None) is not None or getattr(env,'u22_lookahead_ablation',None) is not None):
        raise ValueError('recovery live controller/reference differs')
    return marker


def exploration_identity(cell):
    result=base.exploration_identity('B');result.pop('safety_factorial_cell')
    result.update(schema_version='spline_exploration038_recovery_budget_v1',ablation=MARKER,recovery_cell=cell)
    result['depth']['owner']='src.tasks.hammer.mdp.recovery_budget:RecoveryDepthObserver'
    result['depth']['completion']['boundary']='first_genuine_2000_elapsed_contact_ticks_native_policy_boundary'
    result['critic_event_semantics']['credit_contact_age']='elapsed_integer_physics_ticks_divided_by200ms_clipped_0_2.5'
    result['recovery_budget']=identity(cell)['recovery_budget']
    if cell in CONTACT_CELLS:result['contact_endpoint_bonus']={key:value for key,value in identity(cell).items() if key.startswith('contact_bonus')}
    return result


def validate_event_layout(env,marker):
    from src.tasks.hammer.mdp import recovery_budget as mdp
    from .split_impulse_campaign import _validate_event_layout
    _validate_event_layout(env,base._base(base.identity('B')),mdp.recovery_window_complete,{},mdp.event_observation)
    reward=env.reward_manager.get_term_cfg('recovery_budget')
    if (type(reward.func) is not mdp.RecoveryBudgetReward or reward.func is not env._recovery_budget
            or reward.func.budget!=marker['recovery_budget'] or reward.weight!=1.
            or reward.params!={'budget':marker['recovery_budget']}):
        raise ValueError('recovery live reward owner differs')
    if marker['cell_id'] in CONTACT_CELLS:_validate_contact_live(env,marker)


def validate_metadata(metadata):
    value=metadata.get(MARKER)
    if value is None:
        if metadata.get('spline_exploration038',{}).get('schema_version')=='spline_exploration038_recovery_budget_v1':
            raise ValueError('recovery marker missing')
        return None
    marker=validate_identity(value)
    terms=metadata.get('reward_profile',{}).get('terms',{})
    if (metadata.get('spline_exploration038')!=exploration_identity(marker['cell_id'])
            or set(terms)!=set(weights(marker))
            or terms['recovery_budget']!={'impl':'src.tasks.hammer.mdp.recovery_budget.RecoveryBudgetReward','weight':1.,'params':{'budget':marker['recovery_budget']}}
            or metadata['reward_profile']['termination'].get('impl')!='src.tasks.hammer.mdp.recovery_budget.recovery_window_complete'
            or metadata['reward_profile']['termination'].get('params')!={}
            or metadata['reward_profile']['termination'].get('success_semantics')!='native_first_contact_500ms_window_complete_not_nail_depth'
            or terms['speed_goal']['impl']!=__name__+'.RecoverySpeedCost'
            or any(metadata.get(key) is not None for key in ('safety_factorial','split_impulse_campaign','reward_grid_campaign','demo_progress_campaign'))):
        raise ValueError('recovery checkpoint recipe differs')
    if marker['cell_id'] in CONTACT_CELLS:
        if terms['contact_endpoint_bonus']!={'impl':'src.tasks.hammer.mdp.contact_endpoint_bonus.ContactEndpointBonus','weight':1.,'params':{'bonus':marker['contact_bonus']}}:
            raise ValueError('contact bonus checkpoint recipe differs')
    canonical=canonical_contract(metadata)
    base.validate_metadata(canonical)
    return marker


def canonical_contract(metadata):
    """Read-only comparison with B; only declared differences are normalized."""
    result=deepcopy(metadata);result.pop(MARKER)
    result[base.MARKER]=base.identity('B');result['spline_exploration038']=base.exploration_identity('B')
    profile=result['reward_profile'];profile['terms'].pop('recovery_budget')
    profile['terms'].pop('contact_endpoint_bonus',None)
    profile['terms']['speed_goal']['impl']=base.__name__+'.FactorialSpeedCost'
    profile['termination']=dict(impl='src.tasks.hammer.mdp.spline_horizon.horizon_window_complete',params={'window_s':.2},success_semantics='accepted_strike_200ms_window_complete_not_nail_depth')
    return result


def install_model_mask(actor,critic,metadata):
    if validate_metadata(metadata) is not None:
        actor.obs_normalizer.mask_lookahead=True;critic.obs_normalizer.mask_lookahead=False


def cell_for_index(index):
    if type(index) is not int or not 0<=index<3*len(CELLS):raise ValueError('invalid recovery index')
    cell=CELLS[index//3]
    return dict(index=index,cell_id=cell,seed_index=index%3,seed=SEEDS[index%3],worlds=1024,
        rollout_steps=24,updates=200,scope=MARKER,**{MARKER:identity(cell)})


def normalized_initial(value):
    result=deepcopy(value);marker=validate_metadata(result['contract'])
    if marker is None or result['cell'].get(MARKER)!=marker:raise ValueError('recovery initialization marker differs')
    result['contract']=canonical_contract(result['contract'])
    result['cell'].pop(MARKER);result['cell'][base.MARKER]=base.identity('B')
    result['cell']['cell_id']='B';result['cell']['index']=3+result['cell']['seed_index']
    if result['cell']['scope']!= 'smoke':result['cell']['scope']='safety_factorial'
    return base.normalized_initial(result)


def verify_initialization(actual_path,reference_path):
    from pathlib import Path
    actual=json.loads(Path(actual_path).read_text());expected=json.loads(Path(reference_path).read_text())
    if actual['cell']['worlds']!=1024 or expected['cell']['worlds']!=1024 or expected['cell'].get(base.MARKER)!=base.identity('B'):
        raise ValueError('recovery requires qualified same-seed1024-world original B initialization')
    a=normalized_initial(actual);e=base.normalized_initial(expected)
    a['cell']['updates'],a['cell']['scope']=e['cell']['updates'],e['cell']['scope']
    if a!=e:raise ValueError('recovery initial model/normalizer/physical/RNG state differs from original B')
    return True


from .demo_progress_campaign import validate_runner_budget
from .split_impulse_campaign import verify_zero_ledger


def _validate_contact_config(cfg,marker):
    from src.tasks.hammer.mdp import contact_endpoint_bonus as contact
    reward=cfg.rewards.get('contact_endpoint_bonus');onset=cfg.metrics.get('contact_endpoint_onset')
    account=cfg.metrics.get('contact_endpoint_bonus_accounting')
    if (reward is None or reward.func is not contact.ContactEndpointBonus or reward.weight!=1.
            or reward.params!={'bonus':marker['contact_bonus']} or onset is None
            or onset.func is not contact.ContactEndpointOnset or onset.per_substep is not True or onset.reduce!='last'
            or set(onset.params)!={'robot_cfg','contact_sensor_name','width_m'}
            or onset.params['contact_sensor_name']!='hammer_nail_contact' or onset.params['width_m']!=.010
            or onset.params['robot_cfg'].name!='robot'
            or tuple(onset.params['robot_cfg'].site_names)!=('hammer_head_site',)
            or account is None or account.func is not contact.accounting_metric or account.params!={}
            or account.per_substep is not False or account.reduce!='last'
            or list(cfg.metrics).index('contact_endpoint_bonus_accounting')<=list(cfg.metrics).index('cat_soft')):
        raise ValueError('contact endpoint bonus registration differs')


def _validate_contact_live(env,marker):
    from src.tasks.hammer.mdp import contact_endpoint_bonus as contact
    reward=env.reward_manager.get_term_cfg('contact_endpoint_bonus')
    onset=env.metrics_manager.cfg['contact_endpoint_onset']
    snapshot=getattr(env,'_contact_endpoint_onset',None)
    account=env.metrics_manager.cfg['contact_endpoint_bonus_accounting']
    if (type(reward.func) is not contact.ContactEndpointBonus or reward.func is not env._contact_endpoint_bonus
            or reward.func._env is not env or reward.func.bonus!=marker['contact_bonus']
            or reward.params!={'bonus':marker['contact_bonus']} or reward.weight!=1.
            or type(snapshot) is not contact.ContactEndpointOnset or onset.func is not snapshot
            or not onset.per_substep or onset.reduce!='last' or snapshot.width_m!=.010
            or snapshot._env is not env or snapshot._sensor is not env.scene['hammer_nail_contact']
            or snapshot._robot is not env.scene['robot'] or snapshot._site_ids!=env._hammer_first_strike._site_ids
            or onset.params['width_m']!=.010 or onset.params['contact_sensor_name']!='hammer_nail_contact'
            or account.func is not contact.accounting_metric or account.params!={} or account.per_substep
            or list(env.metrics_manager.cfg).index('contact_endpoint_bonus_accounting')<=list(env.metrics_manager.cfg).index('cat_soft')):
        raise ValueError('contact endpoint live owner differs')

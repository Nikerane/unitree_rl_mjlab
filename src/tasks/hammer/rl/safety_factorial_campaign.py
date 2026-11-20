"""Closed six-cell safety study; historical profiles retain their exact guards."""
from copy import copy, deepcopy
import json
from src.tasks.hammer.mdp.impact_speed import ImpactSpeedCost
from mjlab.managers.manager_base import ManagerTermBase

MARKER='safety_factorial'
from src.tasks.hammer.mdp.human_reference import FIVE_REFERENCE, FIVE_DATA_SHA256
CELLS=tuple('ABCDEF')+('B5',)
SEEDS=(2026110201,2026110202,2026110203)
CAPS=[.369,.246,.738,.369,.246,.0164]
_VALUES={'A':(.5,.2,1.,'variable'),'B':(.5,.5,1.,'variable'),
         'C':(.25,.2,1.,'variable'),'D':(.25,.5,1.,'variable'),
         'E':(.5,.5,1.,'fixed'),'F':(.5,.5,0.,'variable')}


class FactorialSpeedCost(ImpactSpeedCost):
    """Admit this study's zero coefficient without changing bank-hashed readers.

    The historical constructor only validates its coefficient and initializes
    ManagerTermBase. Inherit its event/state/evidence call path byte-for-byte.
    """
    def __init__(self,cfg,env):
        if configured(env.cfg) is None or type(cfg.weight) not in (float,int) or cfg.weight!=0. or cfg.params!={}:
            raise ValueError('factorial speed reader requires exact scoped zero-weight recipe')
        ManagerTermBase.__init__(self,env)


def identity(cell):
    if type(cell) is not str or cell not in CELLS:
        raise ValueError('safety factorial requires cell A through F')
    if cell == 'B5':
        return dict(identity('B'), schema='z1_safety_factorial_five_reference_v1', cell_id='B5',
            reference_id=FIVE_REFERENCE, training_demo_ids=[4, 13, 18, 20, 22],
            training_world_counts=[205, 205, 205, 205, 204], training_data_sha256=FIVE_DATA_SHA256)
    early,pressure,progress,gains=_VALUES[cell]
    return dict(schema='z1_safety_factorial_v1',cell_id=cell,reference_id='U22',
        updates=200,worlds=1024,rollout_steps=24,early_fraction=early,
        load_max_p=pressure,velocity_max_p=.5,progress_weight=progress,gain_mode=gains,
        load_caps_nms=list(CAPS),velocity_limit_rad_s=3.1415,tau=.95,min_p=0.,impulse_seed=.001,
        impulse_event_weight=1.,early_deadline_ms=50,reserve_deadline_ms=200,
        episode_length_s=5.,contact_horizon_s=.2,impulse_window_s=.05,
        finalize_on_depth=False,productive_gate_m=.0005,startup_ramp_s=0.,
        physics_dt_s=.00025,policy_dt_s=.02,pd_dt_s=.002,
        actor_mask='normalized_strike_ref_error',actor_columns=[31,32,33],
        actor_width=43,actor_outputs=13,critic_width=47,
        gain_action_indices=list(range(6,12)),gain_probability_mask=0 if gains=='fixed' else 1,
        gain_execution='existing_fixed_nominal' if gains=='fixed' else 'existing_sampled_variable',
        gain_draws='preserve_all13_common_random_draws',
        action_rate_scope='joint_position_only',last_action_scope='joint_position_only',
        reward_transform='preserve_negative',pressure_schedule=None,
        normalizer_checkpoint='ordinary_caT_raw_EMA_and_seed_flags',
        default_evaluation_normalizers='restored',scope='safety_factorial')


def validate_identity(value):
    if type(value) is not dict or json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(identity(value.get('cell_id')),sort_keys=True,allow_nan=False):
        raise ValueError('safety factorial identity differs from the exact six-cell recipe')
    return identity(value['cell_id'])


def cell_for_index(index):
    if type(index) is not int or not (0<=index<18 or index==19):
        raise ValueError('safety factorial requires historical index 0 through17 or five-demo index19')
    cell=CELLS[index//3]
    return dict(index=index,cell_id=cell,seed_index=index%3,seed=SEEDS[index%3],
        worlds=1024,rollout_steps=24,updates=200,scope='safety_factorial',
        **{MARKER:identity(cell)})


def _base(marker):
    from .split_impulse_campaign import identity as split_identity
    return split_identity('S05' if marker['early_fraction']==.5 else 'S06')


def weights(marker):
    from .split_impulse_campaign import weights as split_weights
    result=split_weights(_base(marker));result['spline_progress']=marker['progress_weight']
    return result


def configure(cfg,cell):
    from . import split_impulse_campaign as split
    if any(getattr(cfg,k,None) is not None for k in (MARKER,split.MARKER,'demo_progress_campaign','reward_grid_campaign')):
        raise ValueError('safety factorial requires a pristine independent U22 recipe')
    marker=identity(cell)
    split.configure(cfg,_base(marker)['setting_id'],reference_id=marker['reference_id'])
    delattr(cfg,split.MARKER)
    cfg.rewards['spline_progress'].weight=marker['progress_weight']
    cfg.rewards['speed_goal'].func=FactorialSpeedCost
    cfg.metrics['cat_soft'].params['imp_max_p']=marker['load_max_p']
    if marker['gain_mode']=='fixed':
        cfg.metrics['native_depth_owner031'].params['exploration_setting']='fixed_nominal'
        for group in ('actor','critic'):
            cfg.observations[group].terms['startup_age'].params={'setting':'fixed_nominal'}
    setattr(cfg,MARKER,marker)
    configured(cfg)
    return cfg


def configured(cfg):
    value=getattr(cfg,MARKER,None)
    if value is None:return None
    marker=validate_identity(value)
    from . import split_impulse_campaign as split
    if any(getattr(cfg,k,None) is not None for k in (split.MARKER,'demo_progress_campaign','reward_grid_campaign')):
        raise ValueError('safety factorial cannot claim another campaign')
    expected=dict(use_vel=True,use_impulse=True,imp_limit=CAPS,imp_max_p=marker['load_max_p'],
        imp_seed=.001,limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')
    p=cfg.metrics['cat_soft'].params
    if {k:v for k,v in p.items() if k!='robot_cfg'}!=expected:
        raise ValueError('safety factorial constant pressure or physical caps differ')
    if any(cfg.rewards[name].weight!=weight for name,weight in weights(marker).items()):
        raise ValueError('safety factorial reward weights differ')
    setting='fixed_nominal' if marker['gain_mode']=='fixed' else 'independent'
    if cfg.metrics['native_depth_owner031'].params.get('exploration_setting')!=setting:
        raise ValueError('safety factorial controller/observer setting differs')
    for group in ('actor','critic'):
        terms=cfg.observations[group].terms
        if terms['startup_age'].params!={'setting':'fixed_nominal' if setting=='fixed_nominal' else 'vic_baseline'}:
            raise ValueError('safety factorial startup observation differs')
        if terms['actions'].params!={'action_name':'joint_position'}:
            raise ValueError('ignored gains leaked into last-action observation')
    if cfg.rewards['speed_goal'].func is not FactorialSpeedCost or cfg.rewards['speed_goal'].params!={}:
        raise ValueError('factorial requires its exact zero-weight inherited speed reader')
    if cfg.rewards['action_rate'].params!={'action_name':'joint_position'}:
        raise ValueError('ignored gains leaked into action-rate reward')
    # Check every existing split invariant on an isolated view after normalizing
    # only the explicitly admitted treatment differences above.
    view=copy(cfg);view.rewards=deepcopy(cfg.rewards);view.metrics=deepcopy(cfg.metrics)
    delattr(view,MARKER);setattr(view,split.MARKER,_base(marker))
    view.rewards['spline_progress'].weight=1.
    view.metrics['native_depth_owner031'].params['exploration_setting']='independent'
    split.configured(view,reference_id=marker['reference_id'])
    return marker


def live(env):
    marker=configured(env.cfg)
    if marker is None:return None
    if (env._sampled_pd500.gain_mode!=marker['gain_mode'] or env._sampled_pd500.startup_ramp_s!=0.
            or getattr(env,'_u22_reference_clock',None) is not None
            or getattr(env,'u22_lookahead_ablation',None) is not None):
        raise ValueError('safety factorial live controller/reference treatment differs')
    return marker


def exploration_identity(cell):
    from . import split_impulse_campaign as split
    from .exploration038_contract import identity as historical
    marker=identity(cell);result=split.exploration_identity(_base(marker)['setting_id'])
    result.pop('split_impulse_setting')
    result.update(schema_version='spline_exploration038_safety_factorial_v1',
        ablation=MARKER,safety_factorial_cell=cell,load_max_p=marker['load_max_p'])
    if marker['gain_mode']=='fixed':
        fixed=historical('fixed_nominal')
        for key in ('setting','gain_mode','effective_gain_entropy_weight','gain_entropy_mode',
                    'likelihood_gain_mask','ignored_action_indices','exploration'):
            result[key]=deepcopy(fixed[key])
    return result


def validate_event_layout(env,marker):
    from .split_impulse_campaign import validate_event_layout as validate
    validate(env,_base(marker))


def validate_metadata(metadata):
    value=metadata.get(MARKER)
    if value is None:
        if metadata.get('spline_exploration038',{}).get('schema_version')=='spline_exploration038_safety_factorial_v1':
            raise ValueError('safety factorial metadata marker is missing')
        return None
    marker=validate_identity(value)
    if (any(metadata.get(k) is not None for k in ('split_impulse_campaign','reward_grid_campaign','demo_progress_campaign','u22_lookahead_ablation'))
            or metadata.get('spline_exploration038')!=exploration_identity(marker['cell_id'])):
        raise ValueError('safety factorial checkpoint mixes treatment identities')
    if marker['cell_id'] == 'B5':
        from src.tasks.hammer.mdp.human_reference import reference_identity
        if metadata.get('human_reference') != reference_identity(FIVE_REFERENCE):
            raise ValueError('five-demo checkpoint reference geometry or membership differs')
    terms=metadata.get('reward_profile',{}).get('terms',{})
    expected=weights(marker)
    if set(terms)!=set(expected) or any(terms[n].get('weight')!=w for n,w in expected.items()):
        raise ValueError('safety factorial checkpoint reward profile differs')
    if terms['speed_goal'].get('impl')!=__name__+'.FactorialSpeedCost' or terms['speed_goal'].get('params')!={}:
        raise ValueError('factorial checkpoint speed reader differs')
    from .split_impulse_campaign import _validate_impulse_params
    _validate_impulse_params(terms['delivered_impulse']['params'],_base(marker))
    cat=metadata.get('safety',{}).get('cat_soft',{})
    expected_cat=dict(use_vel=True,use_impulse=True,imp_limit=CAPS,imp_max_p=marker['load_max_p'],
        imp_seed=.001,limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')
    if cat!=expected_cat:raise ValueError('safety factorial checkpoint pressure/caps differ')
    if metadata['reward_profile'].get('credit_timing')!=exploration_identity(marker['cell_id'])['credit_timing']:
        raise ValueError('safety factorial checkpoint payout timing differs')
    return marker


def install_model_mask(actor,critic,metadata):
    if metadata.get(MARKER) is None:return
    validate_metadata(metadata)
    actor.obs_normalizer.mask_lookahead=True
    critic.obs_normalizer.mask_lookahead=False


def normalized_initial(value):
    """Canonicalize only declared treatments; retain every model/state/RNG hash."""
    result=deepcopy(value);contract=result['contract'];marker=validate_metadata(contract)
    if marker is None or result['cell'].get(MARKER)!=marker:
        raise ValueError('factorial initialization marker differs')
    canonical=identity('B')
    result['cell'][MARKER]=canonical
    result['cell']['cell_id']='B'
    result['cell']['index']=3+result['cell']['seed_index']
    contract[MARKER]=canonical
    contract['spline_exploration038']=exploration_identity('B')
    for name,weight in weights(canonical).items():contract['reward_profile']['terms'][name]['weight']=weight
    contract['reward_profile']['terms']['delivered_impulse']['params']['early_fraction']=.5
    contract['reward_profile']['credit_timing']=contract['spline_exploration038']['credit_timing']
    contract['safety']['cat_soft']['imp_max_p']=.5
    diagnostics=result.get('exploration_diagnostics')
    if diagnostics is not None:
        if diagnostics.get('mode')!='independent' or diagnostics.get('fixed_nominal')!=(marker['gain_mode']=='fixed'):
            raise ValueError('factorial initialization exploration differs')
        diagnostics['fixed_nominal']=False
    if marker['gain_mode']=='fixed' and 'sampled_pd' in contract:
        controller=contract['sampled_pd']
        if (controller.get('version')!='fixed_pd500_v1' or controller.pop('ignored_action_indices',None)!=list(range(6,12))
                or contract.get('action_type')!='joint_position_fixed_sampled_pd'
                or contract.get('controller_identity')!='sampled_pd500_v1'
                or 'sampled_vic_treatment' in contract):
            raise ValueError('fixed nominal initialization controller differs')
        controller.update(version='variable_impedance_pd500_v1',gain_mode='variable',
                          mapping_family='author_v1_exponential',C=1.25,p_bounds=[-1.,1.])
        contract['action_type']='joint_position_variable_impedance_sampled_pd'
        contract['controller_identity']='sampled_vic500_v1'
        contract['sampled_vic_treatment']=deepcopy(contract['split_physical_treatment'])
    return result


def verify_initialization(actual_path,reference_path):
    from pathlib import Path
    actual=json.loads(Path(actual_path).read_text());expected=json.loads(Path(reference_path).read_text())
    if actual['cell'][MARKER]['cell_id'] == 'B5':
        if (actual['cell'] != cell_for_index(19) or expected['cell'] != dict(cell_for_index(19), updates=1, scope='smoke')
                or validate_metadata(actual['contract']) != identity('B5')
                or validate_metadata(expected['contract']) != identity('B5')):
            raise ValueError('five-demo initialization requires its qualified same-seed 1024-world smoke')
        comparable = dict(actual, cell=dict(actual['cell'], updates=1, scope='smoke'))
        if comparable != expected:
            raise ValueError('five-demo production initial state differs from qualification')
        return True
    if (actual['cell']['updates']!=200 or actual['cell']['scope']!='safety_factorial'
            or expected['cell']['updates']!=1 or expected['cell']['scope']!='smoke'
            or actual['cell']['worlds']!=1024 or expected['cell']['worlds']!=1024
            or expected['cell'][MARKER]['cell_id']!='B'):
        raise ValueError('factorial requires same-seed qualified1024-world B smoke initialization')
    a,e=normalized_initial(actual),normalized_initial(expected)
    a['cell']['updates'],a['cell']['scope']=1,'smoke'
    if a!=e:raise ValueError('factorial initialization differs from qualification')
    return True


from .demo_progress_campaign import validate_runner_budget
from .split_impulse_campaign import verify_zero_ledger

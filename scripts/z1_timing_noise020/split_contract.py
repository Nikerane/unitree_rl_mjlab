"""Versioned split identities and file-only validators (stdlib imports only)."""
from copy import deepcopy
import json
import math
import re


def need(ok,message):
    if not ok:raise ValueError(message)


def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def credit_timing(alpha):
    need(type(alpha) is float and alpha in (.5,1.),'split alpha must be .5 or1.')
    return dict(schema_version='spline_split_credit_v1',early_fraction=alpha,reserve_fraction=1.-alpha,
        early_deadline_ms=50,reserve_deadline_ms=200,raw_accepted_event_units=1.,
        eligibility='original_first_onset_consumed_including_rejected_contact',
        boundary='first_native_policy_boundary_at_or_after_each_integer_tick_deadline',
        eligible_failure_and_timeout='pay_all_remaining_buckets_once',external_reset='clear_without_payment',
        continuing_contact_required=False,late_safety_reversal=False,timeout_bootstrap='unchanged',
        observability='unchanged_current43_speed_pending_index42')


def split_identity(setting,early_fraction=.5):
    need(type(setting) is str and setting in ('vic_baseline','vic_position_low'),'split setting must be baseline or position_low')
    timing=credit_timing(early_fraction)
    return dict(schema_version='spline_split_credit_v1',family='split_credit',
        owner='src.tasks.hammer.mdp.spline_split_credit:SplitSplineStrikeReward',setting=setting,
        early_fraction=early_fraction,early_deadline_ms=50,reserve_deadline_ms=200,credit_timing=timing,
        episode_contact_window_ms=200,critic_mode='current',actor_width=43,critic_width=43,critic_event_fields=[],
        gain_mode='variable',actor_class='src.tasks.hammer.rl.spline_exploration_pilot:'+('PilotBaselineActor' if setting=='vic_baseline' else 'PilotPositionLowActor'),
        critic_class='src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedAgeModel',
        distribution_class='src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian',
        initial_exploration_std=[.1]*6+[.3]*6+[1.] if setting=='vic_position_low' else [.3]*12+[1.],
        gain_entropy_weight=1.,effective_gain_entropy_weight=1.,gain_entropy_mode='sum',likelihood_gain_mask=1,
        selector_mask='existing_raw_pending',speed_pending_index=42,action_draws=13,
        fixed_nominal_kp=[256.,384.,384.,256.,192.,128.],fixed_nominal_kd=[6.4]*6,ignored_action_indices=[],
        startup_age=dict(source='sampled_pd500_physics_substeps_since_own_reset',normalization_s=.1,clip=[0.,1.],observation_index=40),
        physics_dt_s=.00025,pd_dt_s=.002,policy_dt_s=.02)


def validate_identity(value):
    need(type(value) is dict,'split marker missing')
    expected=split_identity(value.get('setting'),value.get('early_fraction'))
    need(same(value,expected),'split marker differs from closed schema')
    return expected


def _vector(value,*,boolean=False,integer=False,low=0,high=None):
    need(type(value) is list and len(value)==2,'split vector must contain two worlds')
    for x in value:
        if boolean:need(type(x) is bool,'split boolean vector required')
        else:
            need(type(x) in ((int,) if integer else (int,float)) and math.isfinite(x),'finite split numeric required')
            need((low is None or x>=low) and (high is None or x<=high),'split value outside domain')


def _close(a,b,message):
    need(math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-5),message)


def validate_trace_row(row,update,*,previous=None):
    """Join the validated preceding terminal state; return a detached next state."""
    need(type(row) is dict and row.get('schema')=='z1_split_credit_training_trace_v1','exact new split trace schema required')
    need(type(update) is int and 0<=update<500 and type(row.get('update')) is int and row['update']==update
         and type(row.get('max_updates')) is int and row['max_updates']==500,'split500 update scope differs')
    need(same(row.get('split_credit'),credit_timing(.5)),'split trace marker differs')
    need(row.get('physics_dt_s')==.00025 and row.get('policy_dt_s')==.02,'split trace clock differs')
    envelope={'schema','update','max_updates','physics_dt_s','policy_dt_s','split_credit','world_ids','scope','gamma','lam','steps','final_values',
        'executed_action_semantics','storage_reward_semantics','raw_advantage_semantics'}
    need(set(row)==envelope,'split trace envelope fields differ')
    # Intentional transient projection for unchanged common schema/shape checks.
    # The persisted record and all owner/ledger validation stay new-schema only.
    from scripts.z1_exploration_pilot_campaign import validate_trace
    view=dict(row,schema='z1_credit_training_trace_v1');validate_trace(view,update)
    json.dumps(row,allow_nan=False)
    need(previous is None if update == 0 else type(previous) is dict,
         "split update0 must start fresh; later updates require previous terminal state")
    bools={'consumed_before','consumed_after','started','native_reset'}
    integers={'elapsed_ticks','generation','resets_since_call','call_index'}
    units={'early_before','reserve_before','new_accepted','early_paid','reserve_paid','early_after','reserve_after',
           'externally_discarded_early','externally_discarded_reserve'}
    prekeys={'raw_reward','reward_components','accepted_credit_paid','r_neg_raw_minus_positive','r_neg_semantics','r_pos','delta',
        'native_reset','timeout','started','contact_age_s','unpaid_credit','split_credit','native_causes','terminated',
        'value_at_action','latent_actions','executed_actions','distribution_mean','distribution_std','constraints','constraint_status'}
    storedkeys={'values','rewards_after_cat_and_bootstrap','soft_dones','hard_dones','returns','next_values','raw_advantage','td_residual'}
    if previous is not None:
        need(set(previous)==prekeys and type(previous.get('split_credit')) is dict
             and set(previous['split_credit'])==bools|integers|units, 'previous split terminal schema differs')
        prior=previous['split_credit']
        _vector(previous['native_reset'],boolean=True)
        for key in bools:_vector(prior[key],boolean=True)
        for key in integers:_vector(prior[key],integer=True,high=1000000000)
        for key in units:_vector(prior[key],high=1)
        need(prior['call_index']==[update*24]*2 and prior['native_reset']==previous['native_reset'],
             'previous split terminal clock/native flags differ')
    for step,record in enumerate(row['steps']):
        pre,st=record['pre_reset'],record['stored'];ledger=pre.get('split_credit')
        need(set(record)=={'step','pre_reset','stored'} and set(pre)==prekeys and set(st)==storedkeys,'split row fields differ')
        need(type(ledger) is dict and set(ledger)==bools|integers|units,'closed split ledger required')
        for key in bools:_vector(ledger[key],boolean=True)
        for key in integers:_vector(ledger[key],integer=True,high=1000000000)
        for key in units:_vector(ledger[key],high=1)
        _vector(pre['terminated'],boolean=True)
        causes=pre['native_causes']
        need(type(causes) is dict and 0<len(causes)<=64 and all(type(n) is str and n for n in causes),'native causes missing')
        for value in causes.values():
            need(type(value) is dict and set(value)=={'time_out','flags'} and type(value['time_out']) is bool,'native cause schema differs')
            _vector(value['flags'],boolean=True)
        for w in range(2):
            at=lambda key:ledger[key][w]
            native=pre['native_reset'][w];timeout=pre['timeout'][w];terminated=pre['terminated'][w]
            need(native==(timeout or terminated) and native==at('native_reset') and pre['started'][w]==at('started'),'native/ledger union differs')
            need(timeout==any(v['flags'][w] for v in causes.values() if v['time_out'])
                 and terminated==any(v['flags'][w] for v in causes.values() if not v['time_out']),'native cause union differs')
            need(at('call_index') == update*24+step+1, 'split reward call chronology differs')
            if previous is None:
                need(at('early_before')==0. and at('reserve_before')==0.
                     and not at('consumed_before') and at('externally_discarded_early')==0.
                     and at('externally_discarded_reserve')==0., 'fresh split owner cannot carry prior credit/eligibility/discards')
                need(at('generation')==at('resets_since_call'), 'fresh split setup reset provenance differs')
            event=at('started') and not at('consumed_before')
            need(at('new_accepted') in (0,1) and (not at('new_accepted') or event)
                 and at('consumed_after')==(at('consumed_before') or at('started')),'once-only event consumption differs')
            for bucket,deadline in (('early',200),('reserve',800)):
                before,paid,after=(at(bucket+'_'+suffix) for suffix in ('before','paid','after'))
                need(all(x in (0.,.5) for x in (before,paid,after)),'fractional bucket must be zero or half unit')
                available=before+.5*at('new_accepted')
                _close(available,paid+after,'split bucket conservation differs')
                due=native or (at('started') and at('elapsed_ticks')>=deadline)
                _close(paid,available if due else 0.,'split payment timing differs')
                if previous is not None:
                    prev=previous['split_credit'];resets=at('resets_since_call')
                    need(at('generation')==prev['generation'][w]+resets,'split reset generation differs')
                    need(not previous['native_reset'][w] or resets>=1,'native reset must advance reward generation')
                    discarded=at('externally_discarded_'+bucket)
                    _close(prev[bucket+'_after'][w],before+discarded,'split inter-row clear conservation differs')
                    need(not discarded or resets>0,'unattributed external discard')
                    if resets:need(before==0. and not at('consumed_before'),'reset must clear balance and eligibility')
                    else:need(at('consumed_before')==prev['consumed_after'][w],'eligibility changed without reset')
            _close(at('early_paid')+at('reserve_paid'),pre['accepted_credit_paid'][w],'split raw units/component join differs')
            _close(at('early_after')+at('reserve_after'),pre['unpaid_credit'][w],'split unpaid total differs')
            _close(sum(v[w] for v in pre['reward_components'].values()),pre['raw_reward'][w],'reward component sum differs')
            _close(pre['raw_reward'][w]-pre['r_pos'][w],pre['r_neg_raw_minus_positive'][w],'negative reward join differs')
            _close(pre['constraints']['delta'][w],pre['delta'][w],'constraint/CaT delta differs')
            expected_reward=pre['raw_reward'][w]-pre['delta'][w]*pre['r_pos'][w]+(1-pre['delta'][w])*row['gamma']*st['values'][w]*timeout
            _close(expected_reward,st['rewards_after_cat_and_bootstrap'][w],'CaT/timeout stored reward differs')
            nxt=row['final_values'][w] if step==23 else row['steps'][step+1]['stored']['values'][w]
            _close(nxt,st['next_values'][w],'next value chronology differs')
            continuation=(1-pre['delta'][w])*(1-native)
            td=st['rewards_after_cat_and_bootstrap'][w]+row['gamma']*continuation*nxt-st['values'][w]
            _close(td,st['td_residual'][w],'TD join differs')
            _close(st['returns'][w]-st['values'][w],st['raw_advantage'][w],'advantage/return join differs')
            next_adv=0. if step==23 else row['steps'][step+1]['stored']['raw_advantage'][w]
            _close(td+row['gamma']*row['lam']*continuation*next_adv,st['raw_advantage'][w],'GAE chronology differs')
        previous=pre
    return deepcopy(previous)


def validate_initial(initial,training_cell):
    """Require fresh B6 identity and the actual new full saved contract."""
    from scripts.z1_exploration_pilot_campaign import COMMON_INITIAL_HASHES
    need(type(initial) is dict and type(training_cell) is dict,'split initial/cell dictionaries required')
    index=training_cell.get('index')
    need(type(index) is int and 0<=index<6,'split local index differs')
    expected=dict(index=index,setting=('vic_baseline','vic_position_low')[index%2],seed=2026103101+index//2,
        critic_mode='current',updates=500,worlds=1024,rollout_steps=24,early_fraction=.5,early_deadline_ms=50,reserve_deadline_ms=200)
    need(same(training_cell,expected) and same(initial.get('cell'),expected),'split cell differs')
    need(initial.get('phase')=='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT'
         and type(initial.get('actual_counter')) is int and initial['actual_counter']==0
         and type(initial.get('critic_width')) is int and initial['critic_width']==43,'split fresh phase/counter/width differs')
    for key in ('model_initialization_seed','collection_seed'):
        need(type(initial.get(key)) is int and initial[key]==expected['seed'],'split initialization seed differs')
    need(same(initial.get('live_cat'),dict(velocity_pmax=.5,impulse_pmax=.2,scheduler=False)),'split live CaT differs')
    hashes=initial.get('hashes')
    need(type(hashes) is dict and set(hashes)==set(COMMON_INITIAL_HASHES)|{'actor','critic'},'split initial hash fields differ')
    need(all(type(x) is str and re.fullmatch('[0-9a-f]{64}',x) for x in hashes.values()),'split hash digest missing')
    obs=initial.get('initial_observations')
    need(type(obs) is dict and set(obs)=={'source','actor_shape','critic_shape','actor_sha256','critic_shared43_sha256','scope'}
         and obs['source']=='existing_env_obs_buf_no_recompute'
         and obs['scope']=='saved observations only; not full physical-state identity'
         and same(obs['actor_shape'],[1024,43]) and same(obs['critic_shape'],[1024,43]),'split existing observation identity differs')
    for key in ('actor_sha256','critic_shared43_sha256'):
        need(type(obs[key]) is str and re.fullmatch('[0-9a-f]{64}',obs[key]),'split observation hash missing')
    contract=initial.get('contract');need(type(contract) is dict,'split full contract missing')
    marker=validate_identity(contract.get('spline_split_credit'))
    need(same(marker,split_identity(expected['setting'])),'split marker differs from selected cell')
    need(not any(k in contract for k in ('spline_exploration_pilot','credit_critic_experiment','spline_learning_variant','pilot_physical_treatment')),'split mixed legacy markers forbidden')
    std=initial.get('initial_std');need(type(std) is list and len(std)==13,'split initial std missing')
    for value,want in zip(std,marker['initial_exploration_std']):
        need(type(value) in (int,float) and math.isfinite(value) and math.isclose(value,want,rel_tol=1e-6,abs_tol=1e-8),'split actual std differs')
    fields=dict(schema_version='impact_speed_v1',action_type='joint_position_variable_impedance_sampled_pd',controller_identity='sampled_vic500_v1',
        action_dim=13,action_terms=['joint_position','joint_stiffness','strike_speed'],action_term_dims=[6,6,1],raw_policy_clip=1.,
        physics_dt_s=.00025,control_decimation=80,speed_bounds_mps=[1.,1.5],speed_mode='policy',speed_axis_world=[0.,0.,-1.],
        observation_widths={'actor':43,'critic':43},observation_names=['joint_pos','joint_vel','ee_pos','ee_vel','head_pos','head_vel','nail_top_pos',
            'nail_depth','strike_phase','strike_ref_error','actions','startup_age','impact_speed_goal','impact_speed_pending'],
        observation_term_widths=[7,7,3,3,3,3,3,1,1,3,6,1,1,1],
        observation_normalization={'type':'physical40_empirical_fixed_tail_v1','empirical_width':40,'fixed_tail':['startup_age','impact_speed_goal','impact_speed_pending'],'empirical_eps':.01})
    for key,want in fields.items():need(same(contract.get(key),want),'split execution contract differs: '+key)
    need(not any(k in contract for k in ('critic_observation_names','critic_observation_term_widths')),'split extra critic fields forbidden')
    speed=contract.get('speed_cost')
    need(same(speed,dict(coefficient=.8493676,units='dimensionless_per_event',reward_term_units='cost_per_second_before_manager_dt',timeout_semantics='reset_boundary_shaping')),'split speed cost differs')
    controller=contract.get('sampled_pd');need(type(controller) is dict,'split sampled controller missing')
    for key,want in dict(version='variable_impedance_pd500_v1',gain_mode='variable',mapping_family='author_v1_exponential',C=1.25,p_bounds=[-1.,1.],
                         physics_dt_s=.00025,pd_dt_s=.002,policy_dt_s=.02,kp=marker['fixed_nominal_kp'],kd=marker['fixed_nominal_kd']).items():
        need(same(controller.get(key),want),'split sampled controller differs: '+key)
    need('ignored_action_indices' not in controller and 'startup_target_ramp' not in controller,'split fixed gains/startup ramp forbidden')
    physical=contract.get('split_physical_treatment')
    need(type(physical) is dict and same(physical,contract.get('sampled_vic_treatment'))
         and all(type(physical.get(k)) is dict and physical[k] for k in ('reference','guidance','load')),'split physical treatment missing/different')
    safety=contract.get('safety',{}).get('cat_soft')
    wanted=dict(use_vel=True,use_impulse=True,imp_limit=[.369,.246,.738,.369,.246,.0164],imp_max_p=.2,imp_seed=.001,
                limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')
    need(same(safety,wanted),'split safety recipe differs')
    profile=contract.get('reward_profile');need(type(profile) is dict,'split reward profile missing')
    need(same(profile.get('credit_timing'),credit_timing(.5)),'split reward timing differs')
    term=profile.get('terms',{}).get('accepted_spline_strike')
    need(type(term) is dict and set(term)=={'impl','weight','params'}
         and term['impl']=='src.tasks.hammer.mdp.spline_split_credit.SplitSplineStrikeReward' and term['weight']==1.,'split actual reward owner/unit differs')
    params=term['params'];need(type(params) is dict and set(params)=={'robot_cfg','width_m','early_fraction','early_deadline_ms','reserve_deadline_ms'},'split reward parameters differ')
    for key,want in dict(width_m=.01,early_fraction=.5,early_deadline_ms=50,reserve_deadline_ms=200).items():
        need(same(params[key],want),'split reward parameter differs: '+key)
    need(same(profile.get('termination'),dict(impl='src.tasks.hammer.mdp.spline_horizon.horizon_window_complete',params={'window_s':.2},
        success_semantics='accepted_strike_200ms_window_complete_not_nail_depth')),'split200ms horizon differs')
    json.dumps(initial,allow_nan=False)
    return marker

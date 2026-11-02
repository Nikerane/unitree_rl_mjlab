"""Strict opt-in live/controller and model identity for exploration pilots."""
import json
import math
import torch
from torch import nn

from src.tasks.hammer.mdp.spline_exploration_pilot import pilot_setting,pilot_startup_age,_require_setting
from src.tasks.hammer.mdp.sampled_pd500 import ARM_KP,ARM_KD
from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_ACTOR_CLASSES,PilotGaussian
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel,ImpactSpeedNormalization
from src.tasks.hammer.rl.spline_credit_critic import CreditEventCritic
from src.tasks.hammer.rl.spline_credit_critic_contract import credit_critic_metadata,validate_experiment_identity


def _need(ok,message):
    if not ok:raise ValueError(message)


def _same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def _identity(setting,experiment):
    _require_setting(setting);experiment=validate_experiment_identity(experiment)
    fixed=setting=='fixed_nominal';weight=1/6 if setting=='vic_gain_mean' else 1.
    return dict(schema_version='spline_exploration_pilot_v1',setting=setting,
        gain_mode='fixed' if fixed else 'variable',actor_width=43,critic_width=experiment['critic_width'],
        actor_class='src.tasks.hammer.rl.spline_exploration_pilot:'+PILOT_ACTOR_CLASSES[setting].__name__,
        critic_class=('src.tasks.hammer.rl.spline_credit_critic:CreditEventCritic' if experiment['critic_mode']=='event' else 'src.tasks.hammer.rl.impact_speed_policy:ImpactSpeedAgeModel'),
        distribution_class='src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian',
        initial_exploration_std=([.1]*6+[.3]*6+[1.] if setting=='vic_position_low' else [.3]*12+[1.]),
        gain_entropy_weight=weight,effective_gain_entropy_weight=0. if fixed else weight,
        gain_entropy_mode='masked_out' if fixed else ('mean' if setting=='vic_gain_mean' else 'sum'),
        likelihood_gain_mask=0 if fixed else 1,selector_mask='existing_raw_pending',action_draws=13,
        fixed_nominal_kp=list(ARM_KP),fixed_nominal_kd=list(ARM_KD),
        ignored_action_indices=list(range(6,12)) if fixed else [],
        startup_age=dict(source='sampled_pd500_physics_substeps_since_own_reset',normalization_s=.1,clip=[0.,1.],observation_index=40),
        credit_critic_experiment=experiment)


def validate_pilot_identity(value):
    """Require an exact declared pilot marker before it can relax any legacy rule."""
    _need(type(value) is dict,'pilot metadata marker missing')
    expected=_identity(value.get('setting'),value.get('credit_critic_experiment'))
    _need(_same(value,expected),'pilot metadata differs from exact declared schema')
    return expected


def pilot_metadata(env):
    setting=pilot_setting(env.cfg)
    if setting is None:return None
    controller=getattr(env,'_sampled_pd500',None);fixed=setting=='fixed_nominal'
    _need(controller is not None and controller.gain_mode==('fixed' if fixed else 'variable'), 'pilot live controller gain mode differs')
    _need(controller.startup_ramp_s==0.,'pilot requires unchanged zero startup ramp')
    age=controller.startup_age
    _need(isinstance(age,torch.Tensor) and torch.is_floating_point(age) and age.shape==(env.num_envs,)
          and bool(torch.isfinite(age).all()) and bool(((age>=0)&(age<=1)).all()),'pilot requires finite per-world normalized startup age')
    for group in ('actor','critic'):
        term=env.observation_manager.get_term_cfg(group,'startup_age')
        _need(term.func is pilot_startup_age and term.params=={'setting':setting},'pilot live startup marker differs')
        _need(term.noise is None and term.clip is None and term.scale is None
              and term.history_length==0 and term.delay_min_lag==0 and term.delay_max_lag==0
              and term.delay_hold_prob==0 and term.delay_update_period==0,'pilot startup age must preserve raw clock semantics')
    experiment=credit_critic_metadata(env)
    _need(experiment is not None,'pilot requires the declared deadline/critic anchor')
    metadata=controller.metadata()  # Existing controller's own comprehensive execution checks.
    for name,expected in (('kp',list(ARM_KP)),('kd',list(ARM_KD)),('pd_dt_s',.002),('physics_dt_s',.00025),('policy_dt_s',.02)):
        _need(_same(metadata.get(name),expected),'pilot nominal controller/timing differs: '+name)
    for name,expected in (('arm_kp',ARM_KP),('arm_kd',ARM_KD)):
        value=getattr(controller,name,None)
        _need(isinstance(value,torch.Tensor) and value.shape==(6,) and torch.equal(value,value.new_tensor(expected)),'pilot live nominal gains differ')
    if fixed:
        _need(metadata.get('version')=='fixed_pd500_v1' and _same(metadata.get('ignored_action_indices'),list(range(6,12))),'pilot fixed controller must ignore precisely six gain actions')
        for name,expected in (('_kp',ARM_KP),('_kd',ARM_KD)):
            value=getattr(controller.gains,name,None)
            _need(isinstance(value,torch.Tensor) and value.shape==(env.num_envs,6)
                  and torch.equal(value,value.new_tensor(expected).expand_as(value)),'pilot fixed applied gain buffers differ from nominal')
    else:
        _need(metadata.get('version')=='variable_impedance_pd500_v1' and metadata.get('gain_mode')=='variable'
              and metadata.get('mapping_family')=='author_v1_exponential' and metadata.get('C')==1.25
              and _same(metadata.get('p_bounds'),[-1.,1.]) and 'ignored_action_indices' not in metadata,'pilot variable gain mapping differs')
    return _identity(setting,experiment)


def validate_pilot_models(actor,critic,expected_metadata):
    _need(type(expected_metadata) is dict,'pilot requires explicit full metadata')
    identity=expected_metadata.get('spline_exploration_pilot')
    expected=validate_pilot_identity(identity)
    sibling=expected_metadata.get('credit_critic_experiment')
    if sibling is not None:_need(_same(validate_experiment_identity(sibling),expected['credit_critic_experiment']),'pilot anchor differs from full metadata')
    _need(expected_metadata.get('spline_learning_variant') is None,'pilot cannot claim a legacy N identity')
    setting=expected['setting'];event=expected['credit_critic_experiment']['critic_mode']=='event'
    for group,model,cls,width,out in (('actor',actor,PILOT_ACTOR_CLASSES[setting],43,13),('critic',critic,CreditEventCritic if event else ImpactSpeedAgeModel,47 if event else 43,1)):
        _need(type(model) is cls and model.obs_dim==width and tuple(model.obs_groups)==(group,),'pilot model class/observation signature differs')
        layers=[layer for layer in model.mlp if isinstance(layer,nn.Linear)]
        _need(bool(layers) and layers[0].in_features==width and layers[-1].out_features==out,'pilot model layer signature differs')
        normalizer=model.obs_normalizer
        _need(type(normalizer) is ImpactSpeedNormalization and normalizer.physical.mean.shape==(40,)
              and normalizer.physical.eps==.01 and normalizer.physical.until is None,'pilot normalizer differs')
    d=actor.distribution
    _need(actor.pilot_setting==setting and type(d) is PilotGaussian and d.output_dim==13
          and d.std_type=='scalar' and isinstance(d.std_param,nn.Parameter) and d.std_param.requires_grad
          and d.std_param.shape==(13,) and bool(torch.isfinite(d.std_param).all())
          and tuple(d.std_range)==(1e-6,1e6) and tuple(d.log_std_range)==(math.log(1e-6),math.log(1e6)), 'pilot Gaussian semantics differ')
    _need(d.gain_entropy_weight==(1/6 if setting=='vic_gain_mean' else 1.),'pilot entropy treatment differs')
    _need(critic.distribution is None,'pilot critic must be deterministic')
    mask=getattr(d,'activity_mask',None)
    if mask is not None:
        _need(isinstance(mask,torch.Tensor) and mask.ndim==2 and mask.shape[-1]==13
              and bool((mask[:,:6]==1).all()) and bool((mask[:,6:12]==expected['likelihood_gain_mask']).all())
              and bool(((mask[:,12]==0)|(mask[:,12]==1)).all()),'pilot live activity mask differs')

"""Opt-in identity for combined rewards and position-only exploration."""
import json
import torch
from torch import nn
from src.tasks.hammer.rl.reward_exploration032 import MODES, SDE_INTERVAL, RewardExplorationActor, RewardExplorationGaussian
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel, ImpactSpeedNormalization
from src.tasks.hammer.rl.spline_split_credit_contract import split_metadata
from src.tasks.hammer.mdp.depth_reward031 import depth_progress_reward031, depth_completion_reward031, depth_collect_metric031
from src.tasks.hammer.mdp.rewards import FirstStrikeDeliveredRewardTerm
from src.tasks.hammer.config.z1 import _FIC_CONTROLLED_DROP_I_REF_N_S as I_REF_DELIVERED032


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(mode):
    need(mode in MODES, 'unknown 032 exploration mode')
    from src.tasks.hammer.rl.depth_reward031_contract import depth_identity031
    reused = depth_identity031('A3')
    return dict(schema='reward_exploration032_v1', mode=mode, actor_width=43, critic_width=43,
        impulse=dict(weight=4., i_ref=I_REF_DELIVERED032, saturate=False,
                     reader='FirstStrikeDeliveredRewardTerm', units='legacy_normalized_before_manager_dt',
                     window_ms=50, payment='productive_first_window_once'),
        depth={key:reused[key] for key in ('measurement','progress','completion','collection','reset','observability')},
        accepted_credit=dict(early_fraction=.5, early_ms=50, reserve_ms=200),
        initial_std=[.3]*12+[1.], position_indices=list(range(6)),
        gain_speed_law='existing_scalar_Gaussian', selector_mask='raw_pending',
        sde=dict(interval_steps=SDE_INTERVAL, features='unit_L2_actor_normalized43_plus_constant_detached',
                 per_world_matrix=True, resample_on_reset=True) if mode == 'gsde' else None)


def validate_identity(value):
    need(type(value) is dict, '032 identity missing')
    expected = identity(value.get('mode'))
    need(json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
         '032 reward/exploration identity differs')
    return expected


def metadata032(env):
    from src.tasks.hammer.config.z1.spline_reward_exploration032_cfg import ExplorationDepthObserver032
    owner = getattr(env, '_z1_depth_credit031', None)
    if type(owner) is not ExplorationDepthObserver032:
        return None
    from src.tasks.hammer.cat.hook import CatSoftHook
    hook = env.metrics_manager.cfg['cat_soft'].func
    need(type(hook) is CatSoftHook and hook._max_p == .5 and hook._imp_max_p == .2
         and hook._velocity_pressure is None and hook._use_vel and hook._use_impulse,
         '032 requires unchanged live velocity/impulse CaT treatment')
    marker = identity(owner.exploration_mode)
    from src.tasks.hammer.mdp.depth_credit031 import DepthCreditLedger031
    need(type(owner.ledger) is DepthCreditLedger031 and owner.ledger.num_envs == env.num_envs, '032 depth ledger differs')
    need(owner._env is env and owner.arm == 'A3' and owner.ledger.progress_enabled
         and owner.ledger.completion_enabled, '032 requires both native depth rewards')
    need(env.physics_dt == .00025 and env.step_dt == .02, '032 clocks differ')
    split = split_metadata(env)
    need(split is not None and split['setting'] == 'vic_baseline' and split['early_fraction'] == .5,
         '032 requires unchanged split/current43/variable controller')
    for name, func in [('native_depth_progress031', depth_progress_reward031),
                       ('native_depth_completion031', depth_completion_reward031)]:
        term = env.reward_manager.get_term_cfg(name)
        need(term.func is func and term.weight == 1. and term.params == {}, '032 depth reward differs')
    term = env.reward_manager.get_term_cfg('delivered_impulse')
    need(type(term.func) is FirstStrikeDeliveredRewardTerm and term.weight == 4.
         and term.params['i_ref'] == I_REF_DELIVERED032 and term.params['saturate'] is False,
         '032 must restore the historical normalized linear impulse reward')
    need(env.cfg.rewards['completion'].weight == 0., 'legacy repeating completion must stay off')
    registered = env.metrics_manager.cfg['native_depth_owner031']
    collector = env.metrics_manager.cfg['native_depth_collect031']
    need(registered.func is owner and registered.params['arm'] == 'A3'
         and registered.params['exploration_mode'] == owner.exploration_mode
         and registered.per_substep is True and registered.reduce == 'last', '032 depth observer differs')
    need(collector.func is depth_collect_metric031 and collector.params == {}
         and collector.per_substep is False and collector.reduce == 'last', '032 collector differs')
    return marker


def validate_models(actor, critic, metadata):
    marker = validate_identity(metadata.get('spline_reward_exploration032'))
    need(type(actor) is RewardExplorationActor and type(actor.distribution) is RewardExplorationGaussian
         and actor.distribution.mode == marker['mode'], '032 actor/exploration differs')
    need(type(critic) is ImpactSpeedAgeModel and critic.distribution is None, '032 current critic differs')
    for group, model, out in [('actor', actor, 13), ('critic', critic, 1)]:
        need(model.obs_dim == 43 and tuple(model.obs_groups) == (group,), '032 observation signature differs')
        normalizer = model.obs_normalizer
        need(type(normalizer) is ImpactSpeedNormalization and normalizer.physical.mean.shape == (40,)
             and normalizer.physical.eps == .01 and normalizer.physical.until is None, '032 normalizer differs')
        layers = [x for x in model.mlp if isinstance(x, nn.Linear)]
        need(layers[0].in_features == 43 and layers[-1].out_features == out, '032 model shape differs')
        need(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()), '032 nonfinite model state')
    d = actor.distribution
    need(d.std_type == 'scalar' and tuple(d.std_range) == (1e-6,1e6)
         and d.std_param.shape == (13,) and d.std_param.requires_grad, '032 ordinary gain/speed Gaussian differs')
    need((actor.std_head is not None) == (marker['mode'] == 'dependent'), '032 std head differs')
    if actor.std_head is not None:
        need(type(actor.std_head) is nn.Linear and actor.std_head.weight.shape == (6,43), '032 std head shape differs')
    need(hasattr(d,'sde_log_std') == (marker['mode'] == 'gsde'), '032 extraneous gSDE parameters')
    if marker['mode'] == 'gsde':
        need(d.sde_log_std.shape == (44,6) and d.sde_log_std.requires_grad, '032 gSDE shape differs')

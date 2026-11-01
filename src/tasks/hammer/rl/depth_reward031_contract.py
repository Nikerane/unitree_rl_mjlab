"""Truthful031 identity layered on the unchanged current43 split contract."""
from scripts.z1_timing_noise020.split_contract import split_identity, same, need
from src.tasks.hammer.mdp.depth_reward031 import (
    arm_flags, NativeDepthCreditObserver031, depth_collect_metric031,
    depth_progress_reward031, depth_completion_reward031,
)
from src.tasks.hammer.rl.spline_split_credit_contract import split_metadata, validate_split_models


def depth_identity031(arm):
    progress, completion = arm_flags(arm)
    return dict(schema_version='spline_depth_reward031_v1', arm=arm,
        owner='src.tasks.hammer.mdp.depth_reward031:NativeDepthCreditObserver031',
        baseline_split=split_identity('vic_baseline'), actor_width=43, critic_width=43,
        initial_exploration_std=[.3]*12+[1.],
        measurement=dict(source='integrated_nail_slide_qpos', cadence='every_physics_substep',
            physics_dt_s=.00025, policy_dt_s=.02, clamp_m=[0.,.032],
            window='reset_through_actual_native_policy_boundary_including_overshoot',
            independent_of_first_strike_peak=True, rng_draws=0, additional_model_calls=0),
        progress=dict(enabled=progress, reward_key='native_depth_progress031',
            weight=float(progress), episode_budget_units=1., dead_zone_m=.004, threshold_m=.030,
            potential='clip((native_peak_m-.004)/(.030-.004),0,1)',
            payment='positive_new_potential_divided_by_actual_step_dt', rebound='no_repayment'),
        completion=dict(enabled=completion, reward_key='native_depth_completion031',
            weight=float(completion), episode_budget_units=1., threshold_m=.030,
            boundary='first_genuine_800_elapsed_contact_ticks_native_policy_boundary',
            simultaneous_timeout='horizon_wins', earlier_timeout='forfeit_without_payment',
            external_reset='forfeit_without_payment', training_stop='no_payment',
            eligibility='geometric_attainment_only_no_acceptance_productivity_safety_gate'),
        collection='one_cached_receipt_per_common_step_counter_after_final_physics_before_reset',
        reset=dict(owner_only=True, sample='new_scene_depth_after_scene_events_reset',
            initial_progress='already_credited_potential_no_payment',
            initial_attainment='eligible_if_initial_depth_at_least_30mm',
            selected_worlds_only=True, tracing='preserve_last_pre_reset_receipt'),
        observability='unchanged_current43_hidden_peak_and_earned_credit_after_rebound')


def validate_depth_identity031(value):
    need(type(value) is dict, 'depth031 marker missing')
    expected = depth_identity031(value.get('arm'))
    need(same(value, expected), 'depth031 marker differs from closed semantics')
    return expected


def depth_metadata031(env):
    owner = getattr(env, '_z1_depth_credit031', None)
    cfg = getattr(getattr(env, 'metrics_manager', None), 'cfg', None)
    configured = isinstance(cfg, dict) and any(k.startswith('native_depth_') for k in cfg)
    configured = configured or any(k.startswith('native_depth_') for k in env.reward_manager.active_terms)
    if owner is None and not configured:
        return None
    need(type(owner) is NativeDepthCreditObserver031 and owner._env is env,
         'depth031 requires exact live observer owner')
    identity = depth_identity031(owner.arm)
    need(same(split_metadata(env), identity['baseline_split']), 'depth031 baseline split differs')
    need(env.physics_dt == .00025 and env.step_dt == .02, 'depth031 timing differs')
    observation = cfg.get('native_depth_owner031')
    collector = cfg.get('native_depth_collect031')
    need(observation is not None and observation.func is owner and observation.per_substep is True
         and observation.reduce == 'last' and set(observation.params) == {'arm','nail_cfg'}
         and observation.params['arm'] == owner.arm, 'depth031 observer registration differs')
    nail = observation.params['nail_cfg']
    need(nail is owner.nail_cfg and nail.name == 'nail_block'
         and tuple(nail.joint_names) == ('nail_slide',), 'depth031 nail coordinate differs')
    need(collector is not None and collector.func is depth_collect_metric031
         and collector.per_substep is False and collector.reduce == 'last'
         and collector.params == {}, 'depth031 policy collector differs')
    progress, completion = arm_flags(owner.arm)
    need(owner.ledger.progress_enabled is progress and owner.ledger.completion_enabled is completion
         and owner.ledger.num_envs == env.num_envs, 'depth031 ledger flags differ')
    for name, func, enabled in (('native_depth_progress031', depth_progress_reward031, progress),
                                 ('native_depth_completion031', depth_completion_reward031, completion)):
        live = env.reward_manager.get_term_cfg(name)
        declared = env.cfg.rewards[name]
        need(live.func is func and declared.func is func and live.params == declared.params == {}
             and type(live.weight) is float and live.weight == declared.weight == float(enabled),
             'depth031 reward function, fixed weight or parameters differ')
    return identity


def validate_depth_models031(actor, critic, expected_metadata):
    need(type(expected_metadata) is dict, 'depth031 requires full metadata')
    marker = validate_depth_identity031(expected_metadata.get('spline_depth_reward031'))
    need(same(expected_metadata.get('spline_split_credit'), marker['baseline_split']),
         'depth031 model metadata baseline differs')
    validate_split_models(actor, critic, expected_metadata)

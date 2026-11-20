"""Explicit opt-in demonstration progress campaign; historical038 stays exact."""
import json

MARKER = 'demo_progress_campaign'
REFERENCES = ('U22', 'U20', 'U22+U20')
PROGRESS_WEIGHTS = (1., 2., 4.)
ZERO_REWARDS = ('native_depth_progress031', 'native_depth_completion031', 'speed_goal')


def identity(reference_id, progress_weight):
    if (reference_id not in REFERENCES or type(progress_weight) not in (int, float)
            or progress_weight not in PROGRESS_WEIGHTS):
        raise ValueError('demo campaign requires U22/U20/U22+U20 and progress weight1/2/4')
    return dict(schema='z1_demo_progress_campaign_v2', reference_id=reference_id,
        episode_length_s=5., impulse_window_s=.05, finalize_on_depth=False,
        progress_weight=float(progress_weight), delivered_impulse_weight=8.,
        native_depth_progress_weight=0., native_depth_completion_weight=0., speed_goal_weight=0.,
        r_tt_weight=-1., guidance='ordered_geometric_path_no_clock', startup_ramp_s=0.,
        actor_mask='normalized_strike_ref_error', actor_columns=[31,32,33],
        depth_metrics='unchanged_enabled_ledger',
        scope='exploratory_mixed' if '+' in reference_id else 'main_single_reference')


def validate_identity(value):
    if type(value) is not dict:
        raise ValueError('demo campaign identity missing')
    expected = identity(value.get('reference_id'), value.get('progress_weight'))
    # Completed v1 checkpoints retain their original event/timeout semantics.
    if value.get('schema') == 'z1_demo_progress_campaign_v1':
        expected['schema'] = value['schema']
        for key in ('episode_length_s', 'impulse_window_s', 'finalize_on_depth'):
            expected.pop(key)
    if json.dumps(value,sort_keys=True,allow_nan=False) != json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('demo campaign identity differs from exact recipe')
    return expected


def configured(cfg):
    value = getattr(cfg, MARKER, None)
    if value is None:
        return None
    marker = validate_identity(value)
    if marker['schema'] == 'z1_demo_progress_campaign_v2':
        params = cfg.metrics['first_strike'].params
        if (cfg.episode_length_s != 5. or params.get('finalize_on_depth') is not False
                or abs(params['window_substeps'] * cfg.sim.mujoco.timestep - .05) > 1e-9):
            raise ValueError('demo campaign requires a full50ms impulse window and5s timeout')
    event = cfg.events.get('sample_strike_route_signs')
    from src.tasks.hammer.mdp.human_reference import reset_human_reference
    if (event is None or event.func is not reset_human_reference
            or event.params != {'reference_id':marker['reference_id']}):
        raise ValueError('demo campaign requires the selected native human reference')
    for name, weight in weights(marker).items():
        if name not in cfg.rewards or cfg.rewards[name].weight != weight:
            raise ValueError('demo campaign configured reward differs: '+name)
    owner = cfg.metrics['native_depth_owner031']
    if (owner.params.get('exploration_setting') != 'independent'
            or owner.params.get('impulse_reward_weight') != 8.):
        raise ValueError('demo campaign requires independent038 weight8 observer')
    return marker


def weights(marker):
    from src.tasks.hammer.rl.spline_split_event036_contract import EVENT_REWARD_WEIGHTS
    return dict(EVENT_REWARD_WEIGHTS,delivered_impulse=8.,native_depth_progress031=0.,
        native_depth_completion031=0.,speed_goal=0.,spline_progress=marker['progress_weight'])


def configure(cfg, reference_id, progress_weight):
    from src.tasks.hammer.config.z1.spline_exploration038_cfg import set_impulse_reward_weight
    marker = identity(reference_id, progress_weight)
    if getattr(cfg, MARKER, None) is not None:
        raise ValueError('demo campaign already configured')
    set_impulse_reward_weight(cfg, 8.)
    for name in ZERO_REWARDS:
        cfg.rewards[name].weight = 0.
    cfg.rewards['spline_progress'].weight = marker['progress_weight']
    cfg.episode_length_s = marker['episode_length_s']
    cfg.metrics['first_strike'].params['finalize_on_depth'] = False
    setattr(cfg, MARKER, marker)
    configured(cfg)
    return cfg


def live(env):
    marker = configured(env.cfg)
    if marker is None:
        return None
    if (getattr(env,'_u22_reference_clock',None) is not None
            or getattr(env,'u22_lookahead_ablation',None) is not None
            or env._sampled_pd500.startup_ramp_s != 0.
            or env._sampled_pd500.gain_mode != 'variable'):
        raise ValueError('demo campaign requires native geometric guidance and zero-ramp VIC')
    from src.tasks.hammer.mdp.human_reference import human_metadata
    if human_metadata(env)['reference_id'] != marker['reference_id']:
        raise ValueError('demo campaign live human reference differs')
    return marker


def validate_metadata(metadata):
    marker = metadata.get(MARKER)
    if marker is None:
        return None
    marker = validate_identity(marker)
    from src.tasks.hammer.rl.exploration038_contract import identity as exploration_identity
    from src.tasks.hammer.mdp.human_reference import reference_identity
    if (metadata.get('u22_lookahead_ablation') is not None
            or metadata.get('spline_exploration038') != exploration_identity('independent',8.)
            or metadata.get('human_reference') != reference_identity(marker['reference_id'])):
        raise ValueError('demo campaign checkpoint mixes incompatible treatment identities')
    terms = metadata.get('reward_profile',{}).get('terms',{})
    if set(terms) != set(weights(marker)) or any(terms[n].get('weight') != w for n,w in weights(marker).items()):
        raise ValueError('demo campaign checkpoint reward profile differs')
    return marker


def install_model_mask(actor, critic, metadata):
    if metadata.get(MARKER) is None:
        return
    validate_metadata(metadata)
    if critic.obs_normalizer.mask_lookahead:
        raise ValueError('demo campaign critic must retain lookahead')
    actor.obs_normalizer.mask_lookahead = True


def verify_zero_ledger(path):
    """Reject missing updates or any payment from disabled reward terms."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or [r['update'] for r in rows] != list(range(len(rows))):
        raise ValueError('demo campaign reward ledger update sequence differs')
    for row in rows:
        for name in ZERO_REWARDS:
            if row['reward_terms'].get(name) != dict(mean=0.,sum=0.,min=0.,max=0.):
                raise ValueError('disabled demo reward paid credit: '+name)
    return len(rows)


def validate_runner_budget(runner, cell, *, smoke):
    env = runner.env.unwrapped
    expected_worlds = cell['worlds'] if smoke else 1024
    if (type(env.num_envs) is not int or env.num_envs != expected_worlds
            or cell['worlds'] != expected_worlds or cell['rollout_steps'] != 24
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env'] != 24
            or type(runner.cfg['seed']) is not int or runner.cfg['seed'] != cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed != cell['seed']):
        raise ValueError('demo campaign live worlds, rollout or seed differs')

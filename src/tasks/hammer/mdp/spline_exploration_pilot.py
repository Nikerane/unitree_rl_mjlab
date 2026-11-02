"""Opt-in pilot identity on the unchanged sampled-controller startup clock."""
import torch

from src.tasks.hammer.rl.spline_exploration_pilot import PILOT_SETTINGS


def _require_setting(setting):
    if type(setting) is not str or setting not in PILOT_SETTINGS:
        raise ValueError('pilot setting must be one of the four declared treatments')


def pilot_startup_age(env, *, setting):
    """Read physical reset age; zero-fill only the observation construction probe."""
    _require_setting(setting)
    controller=getattr(env,'_sampled_pd500',None)
    if controller is None:
        if hasattr(env,'observation_manager'):
            raise RuntimeError('pilot startup age requires its sampled controller after manager construction')
        return torch.zeros((env.num_envs,1),device=env.device)
    return controller.startup_age.unsqueeze(-1)


def pilot_setting(cfg):
    """Resolve exact opt-in markers; leave all other identities to legacy checks."""
    terms=[cfg.observations[group].terms.get('startup_age') for group in ('actor','critic')]
    if not any(term is not None and term.func is pilot_startup_age for term in terms):
        return None
    if any(term is None or term.func is not pilot_startup_age
           or type(term.params) is not dict or set(term.params)!={'setting'} for term in terms):
        raise ValueError('pilot requires matching exact actor/critic startup-age markers')
    setting=terms[0].params['setting'];_require_setting(setting)
    if terms[1].params['setting']!=setting:
        raise ValueError('pilot actor/critic settings differ')
    return setting

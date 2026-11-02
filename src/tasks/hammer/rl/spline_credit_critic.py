"""Critic-only event context and explicit common initialization for the credit screen."""
from __future__ import annotations

import torch
from torch import nn

from src.tasks.hammer.rl.impact_speed_policy import (
    ImpactSpeedAgeModel, ImpactSpeedLowNoiseModel, ImpactSpeedModel,
    ImpactSpeedNormalization, ImpactSpeedGaussian,
)


class CreditEventCritic(ImpactSpeedModel):
    """Physical40 + unchanged age/goal/pending + four raw credit-event scalars."""

    observation_width = 47

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256),
                 activation='elu', obs_normalization=True, distribution_cfg=None):
        if (obs_set != 'critic' or tuple(obs_groups.get(obs_set, ())) != ('critic',)
                or type(output_dim) is not int or output_dim != 1
                or distribution_cfg is not None):
            raise ValueError('CreditEventCritic requires deterministic critic observations and one output')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, distribution_cfg)
        # Initial context cannot change the common reference function. The paired
        # initializer below also supplies the shared physical/legacy parameters.
        first = next(layer for layer in self.mlp if isinstance(layer, nn.Linear))
        with torch.no_grad():
            first.weight[:, 42:].zero_()


def _first_linear(model):
    return next((layer for layer in model.mlp if isinstance(layer, nn.Linear)), None)


def _check_model(model, allowed_classes, group, width):
    if (type(model) not in allowed_classes or model.obs_dim != width
            or tuple(model.obs_groups) != (group,)):
        raise ValueError('paired credit initialization requires exact model class, width and observation group')
    first = _first_linear(model)
    if first is None or first.in_features != width:
        raise ValueError('paired credit model has an inconsistent first layer')
    last = next(layer for layer in reversed(model.mlp) if isinstance(layer, nn.Linear))
    if group == 'actor':
        d = model.distribution
        if (type(d) is not ImpactSpeedGaussian or d.std_type != 'scalar'
                or d.std_param.shape != (13,) or not d.std_param.requires_grad
                or last.out_features != 13):
            raise ValueError('paired actor requires unchanged learnable13D Gaussian semantics')
    elif model.distribution is not None or last.out_features != 1:
        raise ValueError('paired critic must remain deterministic with one output')
    normalizer = model.obs_normalizer
    if (type(normalizer) is not ImpactSpeedNormalization
            or normalizer.physical.mean.shape != (40,)
            or normalizer.physical.eps != .01 or normalizer.physical.until is not None):
        raise ValueError('paired credit model requires the original physical40 normalizer')


def _architecture(model):
    """Compare non-state semantics too, including activation type and parameters."""
    rows = []
    for layer in model.mlp:
        if isinstance(layer, nn.Linear):
            rows.append((type(layer), 43 if not rows else layer.in_features,
                         layer.out_features, layer.bias is not None))
        else:
            rows.append((type(layer), repr(layer)))
    return tuple(rows)


def _prepared_state(destination, reference, *, extend_critic=False):
    if _architecture(destination) != _architecture(reference):
        raise ValueError('paired credit model architectures differ')
    target, source = destination.state_dict(), reference.state_dict()
    if set(target) != set(source):
        raise ValueError('paired credit models have different state keys')
    first_name = next(name for name, layer in destination.mlp.named_modules()
                      if isinstance(layer, nn.Linear))
    first_key = f'mlp.{first_name}.weight'
    result = {}
    for key, value in source.items():
        expected = target[key]
        if (not isinstance(value, torch.Tensor) or not isinstance(expected, torch.Tensor)
                or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise ValueError('paired credit state must have finite tensors with matching dtypes')
        if extend_critic and key == first_key:
            if value.ndim != 2 or expected.shape != (value.shape[0], 47) or value.shape[1] != 43:
                raise ValueError('paired credit first-layer embedding requires 43 to 47 inputs')
            result[key] = torch.cat((value.clone(), value.new_zeros((value.shape[0], 4))), dim=1)
        else:
            if value.shape != expected.shape:
                raise ValueError('paired credit state shapes differ')
            result[key] = value.clone()
    return result


def initialize_matched_credit_models(actor, critic, reference_actor, reference_critic):
    """Copy a common 43-input reference without RNG draws or reference mutation.

    Both complete states are validated and prepared before either destination is
    changed. This helper is initialization, not a checkpoint/optimizer transfer;
    the caller must supply fresh optimizers and separately fix rollout RNG state.
    """
    if len({id(x) for x in (actor, critic, reference_actor, reference_critic)}) != 4:
        raise ValueError('paired credit models and references must be distinct objects')
    _check_model(actor, (ImpactSpeedLowNoiseModel,), 'actor', 43)
    _check_model(reference_actor, (ImpactSpeedLowNoiseModel,), 'actor', 43)
    _check_model(reference_critic, (ImpactSpeedAgeModel,), 'critic', 43)
    event = type(critic) is CreditEventCritic
    _check_model(critic, (ImpactSpeedAgeModel, CreditEventCritic), 'critic', 47 if event else 43)
    for model in (reference_actor, reference_critic):
        if torch.count_nonzero(_first_linear(model).weight[:, 42]).item() != 0:
            raise ValueError('paired credit initialization requires the fresh zero pending-input column')
    if (tuple(actor.distribution.std_range) != tuple(reference_actor.distribution.std_range)
            or tuple(actor.distribution.log_std_range) != tuple(reference_actor.distribution.log_std_range)):
        raise ValueError('paired actor exploration bounds differ')
    actor_state = _prepared_state(actor, reference_actor)
    critic_state = _prepared_state(critic, reference_critic, extend_critic=event)
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)

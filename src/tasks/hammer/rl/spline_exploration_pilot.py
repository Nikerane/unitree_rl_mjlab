"""Opt-in exploration factors for the prospective twelve-run velocity pilot.

No registration or legacy model behavior changes on import. Fixed nominal keeps
all thirteen Gaussian draws; only scoring ignores the six unused gain requests.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import torch
from torch import nn

from src.tasks.hammer.rl.impact_speed_policy import (
    ImpactSpeedGaussian, ImpactSpeedLowNoiseModel, ImpactSpeedNormalization,
)

PILOT_SETTINGS = ('vic_baseline', 'vic_position_low', 'vic_gain_mean', 'fixed_nominal')


class PilotGaussian(ImpactSpeedGaussian):
    """Original likelihood/KL, with a separately declared gain-entropy weight."""

    gain_entropy_weight = 1.

    @property
    def entropy(self):
        elementwise = self._distribution.entropy() * self.activity_mask
        return (elementwise[..., :6].sum(-1)
                + self.gain_entropy_weight * elementwise[..., 6:12].sum(-1)
                + elementwise[..., 12])


class _PilotActor(ImpactSpeedLowNoiseModel):
    pilot_setting = None

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256,256),
                 activation='elu', obs_normalization=True, distribution_cfg=None):
        if (type(self) not in PILOT_ACTOR_CLASSES.values()
                or obs_set != 'actor' or tuple(obs_groups.get(obs_set, ())) != ('actor',)
                or type(output_dim) is not int or output_dim != 13
                or distribution_cfg is None):
            raise ValueError('pilot requires an explicit approved 43-input/13-output actor')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        d = self.distribution
        if (type(d) is not PilotGaussian or tuple(d.std_range) != (1e-6,1e6)
                or d.gain_entropy_weight != 1.):
            raise ValueError('pilot requires its Gaussian and unchanged exploration bounds')
        d.gain_entropy_weight = 1/6 if self.pilot_setting == 'vic_gain_mean' else 1.
        if self.pilot_setting == 'vic_position_low':
            with torch.no_grad():
                d.std_param[:6].fill_(.1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        result = super().get_latent(obs, masks, hidden_state)
        if self.pilot_setting == 'fixed_nominal':
            self.distribution.activity_mask[..., 6:12] = 0
        return result


class PilotBaselineActor(_PilotActor):
    pilot_setting = 'vic_baseline'


class PilotPositionLowActor(_PilotActor):
    pilot_setting = 'vic_position_low'


class PilotGainMeanActor(_PilotActor):
    pilot_setting = 'vic_gain_mean'


class PilotFixedNominalActor(_PilotActor):
    pilot_setting = 'fixed_nominal'


PILOT_ACTOR_CLASSES = MappingProxyType(dict(zip(PILOT_SETTINGS, (
    PilotBaselineActor, PilotPositionLowActor, PilotGainMeanActor, PilotFixedNominalActor,
))))


def _validate(model, classes):
    if (type(model) not in classes or model.obs_dim != 43
            or tuple(model.obs_groups) != ('actor',)):
        raise ValueError('pilot initialization requires exact supported actor classes and signature')
    d = model.distribution
    if (type(d) not in (PilotGaussian, ImpactSpeedGaussian) or d.std_type != 'scalar'
            or not isinstance(d.std_param, nn.Parameter) or not d.std_param.requires_grad
            or d.output_dim != 13 or d.std_param.shape != (13,)
            or tuple(d.std_range) != (1e-6,1e6)
            or tuple(d.log_std_range) != (math.log(1e-6), math.log(1e6))):
        raise ValueError('pilot initialization requires unchanged learnable Gaussian semantics')
    if type(model) in PILOT_ACTOR_CLASSES.values():
        setting = next(key for key, cls in PILOT_ACTOR_CLASSES.items() if type(model) is cls)
        if (model.pilot_setting != setting or type(d) is not PilotGaussian
                or d.gain_entropy_weight != (1/6 if setting == 'vic_gain_mean' else 1.)):
            raise ValueError('pilot class, setting and entropy semantics disagree')
    elif type(d) is not ImpactSpeedGaussian:
        raise ValueError('legacy reference requires its original Gaussian')
    normalizer = model.obs_normalizer
    if (type(normalizer) is not ImpactSpeedNormalization
            or normalizer.physical.mean.shape != (40,)
            or normalizer.physical.eps != .01 or normalizer.physical.until is not None):
        raise ValueError('pilot initialization requires the physical40 normalizer')
    layers = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
    if (not layers or layers[0].in_features != 43 or layers[-1].out_features != 13
            or layers[0].weight.shape[1] != 43
            or torch.count_nonzero(layers[0].weight[:,42])):
        raise ValueError('pilot initialization requires a fresh zero pending-input column')


def initialize_matched_pilot_actor(actor, reference_actor):
    """Copy all reference state, then apply only the declared initial-std factor.

    Validate and clone the entire state before mutation; do not reset or consume
    RNG. This is fresh initialization, not optimizer/checkpoint transfer.
    """
    if actor is reference_actor:
        raise ValueError('pilot destination and reference must be distinct')
    _validate(actor, tuple(PILOT_ACTOR_CLASSES.values()))
    _validate(reference_actor, (PilotBaselineActor, ImpactSpeedLowNoiseModel))
    reference_std = reference_actor.distribution.std_param.detach()
    expected_std = reference_std.new_tensor([.3]*12+[1.])
    if not torch.equal(reference_std, expected_std):
        raise ValueError('pilot reference must have the declared baseline initial stds')
    if repr(actor.mlp) != repr(reference_actor.mlp):
        raise ValueError('pilot model architectures differ')
    destination, source = actor.state_dict(), reference_actor.state_dict()
    if destination.keys() != source.keys():
        raise ValueError('pilot model state keys differ')
    prepared = {}
    for key, value in source.items():
        target = destination[key]
        if (not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor)
                or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all()):
            raise ValueError('pilot state requires matching finite tensors')
        prepared[key] = value.clone()
    # load_state_dict writes into destination storage. Reject shared storage
    # across any keys/models, including views and internal ties, before that
    # first write. Whole-storage ranges deliberately reject disjoint views too.
    intervals = []
    for state in (destination, source):
        for value in state.values():
            if value.layout != torch.strided or value.device.type == 'meta':
                raise ValueError('pilot initialization requires inspectable tensor storage')
            storage = value.untyped_storage()
            if storage.nbytes():
                intervals.append((str(value.device), storage.data_ptr(),
                                  storage.data_ptr() + storage.nbytes()))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[0] == current[0] and current[1] < previous[2]:
            raise ValueError('pilot initialization forbids aliased tensor storage')
    if actor.pilot_setting == 'vic_position_low':
        prepared['distribution.std_param'][:6].fill_(.1)
    actor.load_state_dict(prepared, strict=True)

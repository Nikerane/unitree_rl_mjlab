"""Opt-in event-critic exploration through the existing PPO/CaT pipeline.

The feature variance and matrix-noise operations below adapt SB3 v2.7.1
StateDependentNoiseDistribution, commit 8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458.
Project additions: position-only noise, finite parameter bounds, strict world
batching and reset-row sampling. Probability scoring uses SB3's additive epsilon.

MIT License — Copyright (c) 2019 Antonin Raffin
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.utils import unpad_trajectories

from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedModel
from src.tasks.hammer.rl.reward_exploration032 import (
    RewardExplorationGaussian, RewardExplorationPPO, SDE_INTERVAL,
)
from src.tasks.hammer.rl.runner import HammerOnPolicyRunner

SETTINGS = ('independent', 'dependent', 'gsde', 'fixed_nominal')


class ExplorationGaussian038(RewardExplorationGaussian):
    """SB3 feature-noise kernel with the existing masked thirteen-action law."""

    variance_epsilon = 1e-6
    feature_map = 'actor_hidden_detached'

    def __init__(self, output_dim, *, mode, feature_dim=256, fixed_nominal=False, **kwargs):
        if type(feature_dim) is not int or feature_dim <= 0 or type(fixed_nominal) is not bool:
            raise ValueError('explicit feature dimension and impedance flag required')
        if fixed_nominal and mode != 'independent':
            raise ValueError('fixed nominal control uses independent position exploration')
        super().__init__(output_dim, mode=mode, **kwargs)
        self.feature_dim = feature_dim
        self.fixed_nominal = fixed_nominal
        if mode == 'gsde':
            self.sde_log_std = nn.Parameter(torch.full((feature_dim, 6), math.log(.3)))

    @property
    def setting(self):
        return 'fixed_nominal' if self.fixed_nominal else self.mode

    def update(self, means):
        if self.mode != 'gsde':
            return super().update(means)
        weights = self.sde_log_std.clamp(*self.log_std_range).exp()
        variance = self.features.square() @ weights.square()
        position_std = (variance + self.variance_epsilon).sqrt()
        other_std = self.std_param[6:].clamp(*self.std_range).expand(*means.shape[:-1], 7)
        self._distribution = Normal(means, torch.cat((position_std, other_std), -1))

    def _noise(self, count):
        scale = self.sde_log_std.clamp(*self.log_std_range).exp()
        return torch.randn(count, self.feature_dim, 6, device=scale.device, dtype=scale.dtype) * scale

    def begin_collection(self, step, num_envs):
        self.collecting = True
        if self.mode == 'gsde':
            if self.noise_matrices is not None and self.noise_matrices.shape[0] != num_envs:
                raise ValueError('gSDE world count changed within collection')
            if self.noise_matrices is None or step % SDE_INTERVAL == 0:
                with torch.no_grad():
                    self.noise_matrices = self._noise(num_envs)

    def reset_noise(self, dones):
        if self.mode != 'gsde' or self.noise_matrices is None or dones is None:
            return
        if dones.numel() != self.noise_matrices.shape[0]:
            raise ValueError('reset flags must identify all collection worlds')
        ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
        if ids.numel():
            with torch.no_grad():
                self.noise_matrices[ids] = self._noise(ids.numel())


class ExplorationActor038(ImpactSpeedModel):
    """Same mean MLP, optional existing RSL std head, detached hidden gSDE features."""

    observation_width = 43

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.distribution
        if type(d) is not ExplorationGaussian038 or list(self.mlp)[-1].in_features != d.feature_dim:
            raise ValueError('038 requires its distribution and matching hidden feature width')
        self.std_head = None
        if d.mode == 'dependent':
            with torch.random.fork_rng():
                self.std_head = nn.Linear(43, 6)
            nn.init.zeros_(self.std_head.weight)
            nn.init.constant_(self.std_head.bias, math.log(.3))

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)
        d = self.distribution
        if getattr(self, 'use_selector_as_clock', False):
            d.activity_mask[..., 12] = 1
        if d.mode == 'dependent':
            d.position_log_std = self.std_head(latent)
        if d.fixed_nominal:
            d.activity_mask[..., 6:12] = 0
        return latent

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if self.distribution.mode != 'gsde':
            return super().forward(obs, masks, hidden_state, stochastic_output)
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        hidden = self.get_latent(obs, masks, hidden_state)
        # Sequential iteration preserves reused activation modules; children()
        # deduplicates them and would silently change the original mean network.
        layers = list(self.mlp)
        for layer in layers[:-1]:
            hidden = layer(hidden)
        means = layers[-1](hidden)
        self.distribution.features = hidden.detach()
        if stochastic_output:
            self.distribution.update(means)
            return self.distribution.sample()
        return means

    @torch.no_grad()
    def calibrate_exploration(self, obs):
        d = self.distribution
        self(obs)  # no sample/RNG draw and no extra feature network
        if d.mode == 'gsde':
            energy = d.features.square().sum(-1).mean()
            if not bool(torch.isfinite(energy)) or float(energy) <= 0:
                raise ValueError('finite nonzero hidden feature energy required for gSDE calibration')
            scale = .3 / energy.sqrt()
            if not bool((scale >= d.std_range[0]) & (scale <= d.std_range[1])):
                raise ValueError('initial gSDE calibration exceeds existing parameter bounds')
            d.sde_log_std.fill_(scale.log())
            d.noise_matrices = None
        return self.exploration_diagnostics(obs)

    @torch.no_grad()
    def exploration_diagnostics(self, obs):
        d = self.distribution
        d.update(self(obs))  # probability evaluation only; does not sample or refresh matrices
        std = d.std[..., :6]
        sampled_var = (std.square() - d.variance_epsilon).clamp_min(0.) if d.mode == 'gsde' else std.square()
        return dict(mode=d.mode, fixed_nominal=d.fixed_nominal,
            position_std_mean=std.mean(0).cpu().tolist(), position_std_min=std.amin(0).cpu().tolist(),
            position_std_max=std.amax(0).cpu().tolist(),
            sampled_position_noise_rms=sampled_var.mean(0).sqrt().cpu().tolist(),
            gain_speed_std=d.std[..., 6:].mean(0).cpu().tolist())

    def reset(self, dones=None, hidden_state=None):
        super().reset(dones, hidden_state)
        self.distribution.reset_noise(dones)


class ExplorationPPO038(RewardExplorationPPO):
    """Freeze only normalization, preserving inherited CaT/GAE/reset bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalization_primed = False
        self.pending_normalization = []
        self.initial_exploration = None
        self.on_prime = None

    def prime(self, obs):
        if self.normalization_primed:
            return
        # Collection normally runs inside inference_mode; buffers/features that
        # survive into a gradient update must be ordinary tensors instead.
        with torch.inference_mode(False), torch.no_grad():
            sample = obs.detach().clone()
            for model in (self.actor, self.critic):
                model.update_normalization(sample)
            self.initial_exploration = self.actor.calibrate_exploration(sample)
        self.normalization_primed = True
        if self.on_prime is not None:
            self.on_prime(obs)

    def act(self, obs):
        self.prime(obs)
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        with torch.inference_mode(False), torch.no_grad():
            sample = obs.detach().clone()
        modules = [module for model in (self.actor, self.critic)
                   for module in model.obs_normalizer.modules()]
        flags = [module.training for module in modules]
        try:
            for module in modules:
                module.training = False
            super().process_env_step(obs, rewards, dones, extras)
        finally:
            for module, flag in zip(modules, flags, strict=True):
                module.training = flag
        self.pending_normalization.append(sample)

    def update(self):
        # Publish only after a successful update. Final bootstrap and every PPO
        # epoch saw exactly the statistics that generated the stored actions.
        result = super().update()
        with torch.inference_mode(False), torch.no_grad():
            for obs in self.pending_normalization:
                self.actor.update_normalization(obs)
                self.critic.update_normalization(obs)
        self.pending_normalization.clear()
        self.collection_step = 0
        self.actor.distribution.noise_matrices = None
        return result


def _model_hashes(runner):
    from scripts.z1_exploration_pilot_training import tensor_state_digest, _normalizer_hash
    actor, critic = runner.alg.actor, runner.alg.critic
    shared_actor = {key: value for key, value in actor.state_dict().items()
                    if not key.startswith(('distribution.', 'std_head.'))}
    shared_critic = dict(critic.state_dict())
    key = next(name for name, layer in critic.mlp.named_modules() if isinstance(layer, nn.Linear))
    key = f'mlp.{key}.weight'
    shared_critic[key] = shared_critic[key][:, :43]
    ns = np.random.get_state()
    numpy_state = [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])]
    device = torch.device(runner.device)
    cuda = torch.cuda.get_rng_state(device).cpu().numpy().tobytes() if device.type == 'cuda' else None
    return dict(actor=tensor_state_digest(actor.state_dict()), critic=tensor_state_digest(critic.state_dict()),
        actor_shared_non_std=tensor_state_digest(shared_actor), shared_critic43=tensor_state_digest(shared_critic),
        actor_normalizer=_normalizer_hash(actor), critic_normalizer=_normalizer_hash(critic),
        collection_torch_rng=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        collection_python_rng=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        collection_numpy_rng=hashlib.sha256(json.dumps(numpy_state, separators=(',', ':')).encode()).hexdigest(),
        collection_cuda_rng=hashlib.sha256(cuda).hexdigest() if cuda is not None else None)


def initialize_shared_models(runner):
    """Reuse the canonical 037 mean/critic initialization, excluding noise extras."""
    from tensordict import TensorDict
    from src.tasks.hammer.rl.spline_exploration_pilot import PilotBaselineActor
    from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
    from src.tasks.hammer.rl.spline_credit_critic import _prepared_state
    seed = runner.cfg['seed']
    device = torch.device(runner.device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        x = torch.zeros(1, 43, device=device)
        obs = TensorDict({'actor': x, 'critic': x.clone()}, [1])
        groups = {'actor': ['actor'], 'critic': ['critic']}
        keys = ('hidden_dims', 'activation', 'obs_normalization')
        ac = deepcopy({key: runner.cfg['actor'][key] for key in keys})
        ac['distribution_cfg'] = {'class_name': 'src.tasks.hammer.rl.spline_exploration_pilot:PilotGaussian'}
        cc = deepcopy({key: runner.cfg['critic'][key] for key in keys})
        reference_actor = PilotBaselineActor(obs, groups, 'actor', 13, **ac).to(device)
        reference_critic = ImpactSpeedAgeModel(obs, groups, 'critic', 1, distribution_cfg=None, **cc).to(device)
    actor, critic = runner.alg.actor, runner.alg.critic
    prepared_critic = _prepared_state(critic, reference_critic, extend_critic=True)
    actor.mlp.load_state_dict(reference_actor.mlp.state_dict(), strict=True)
    actor.obs_normalizer.load_state_dict(reference_actor.obs_normalizer.state_dict(), strict=True)
    critic.load_state_dict(prepared_critic, strict=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return _model_hashes(runner)


class ExplorationRunner038(HammerOnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if type(self.alg) is not ExplorationPPO038:
            raise ValueError('038 requires its shared normalization and exploration algorithm')
        if self.cfg.get('resume') or self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError('038 supports fresh training only')
        initialize_shared_models(self)
        from scripts.z1_exploration038 import cell_for_index, SEEDS
        from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
        from scripts.z1_exploration_pilot_training import tensor_state_digest
        setting = self.alg.actor.distribution.setting
        selected = cell_for_index(SETTINGS.index(setting) * 3 + SEEDS.index(self.cfg['seed']))
        cell = getattr(self, 'study_cell', selected)
        output = Path(getattr(self, 'study_output_dir', Path(self.logger.log_dir).parent))
        path = output / 'initial-state.json'
        if path.exists() or path.is_symlink():
            raise FileExistsError('038 initial-state evidence already exists')
        self.initial_state_record = None

        def record(obs):
            actor_obs, critic_obs = obs['actor'], obs['critic']
            initial_observations = dict(actor_shape=list(actor_obs.shape), critic_shape=list(critic_obs.shape),
                actor_sha256=tensor_state_digest({'obs': actor_obs}),
                critic_sha256=tensor_state_digest({'obs': critic_obs}),
                critic_shared43_sha256=tensor_state_digest({'obs': critic_obs[..., :43]}))
            self.initial_state_record = dict(schema='z1_exploration038_initial_v1', cell=cell,
                phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT', actual_counter=0, critic_width=47,
                contract=build_impact_speed_metadata(self.env.unwrapped, self.env.clip_actions),
                hashes=_model_hashes(self), initial_observations=initial_observations,
                initial_trainable_parameter_hashes={name: tensor_state_digest({
                    key: parameter for key, parameter in getattr(self.alg, name).named_parameters()
                    if parameter.requires_grad}) for name in ('actor', 'critic')},
                initial_std=[.3] * 12 + [1.], exploration_diagnostics=self.alg.initial_exploration,
                model_initialization_seed=cell['seed'], collection_seed=cell['seed'])
            if 'human_reference' in self.initial_state_record['contract']:
                from src.tasks.hammer.mdp.human_reference import human_initial_state
                self.initial_state_record['human_reference_initialization'] = human_initial_state(self.env.unwrapped)
            with path.open('x') as stream:
                json.dump(self.initial_state_record, stream, sort_keys=True, indent=2, allow_nan=False)

        self.alg.on_prime = record
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.on_prime = None

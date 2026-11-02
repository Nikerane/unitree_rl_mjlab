"""Position-only exploration adapters for the existing masked PPO/CaT policy.

gSDE uses the marginal-Gaussian variance and persistent feature-noise construction
in SB3 StateDependentNoiseDistribution. Feature choice here is explicit: unit-L2
normalized actor inputs plus a constant, detached from the actor. At initialization
this gives exactly 0.3 position std without simulator-based calibration.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedModel, ImpactSpeedGaussian
from src.tasks.hammer.rl.cat_ppo import CatPPO

MODES = ('independent', 'dependent', 'gsde')
SDE_INTERVAL = 4


class RewardExplorationGaussian(ImpactSpeedGaussian):
    def __init__(self, output_dim, *, mode, **kwargs):
        if output_dim != 13 or mode not in MODES:
            raise ValueError('reward exploration requires a declared mode and 13 actions')
        super().__init__(output_dim, **kwargs)
        if self.std_type != 'scalar' or tuple(self.std_range) != (1e-6, 1e6):
            raise ValueError('retain existing scalar Gaussian bounds for gains/speed')
        with torch.no_grad():
            self.std_param[:12].fill_(.3)
            self.std_param[12] = 1.
        self.mode = mode
        self.position_distribution = (HeteroscedasticGaussianDistribution(6, init_std=.3, std_type='log')
                                      if mode == 'dependent' else None)
        if mode == 'gsde':
            self.sde_log_std = nn.Parameter(torch.full((44, 6), math.log(.3)))
        self.noise_matrices = None
        self.collecting = False
        self.features = None
        self.position_log_std = None
        self.last_position_noise = None

    def update(self, means):
        if self.mode == 'independent':
            return super().update(means)
        if self.mode == 'dependent':
            self.position_distribution.update(torch.stack((means[..., :6], self.position_log_std), dim=-2))
            position_std = self.position_distribution.std
        else:
            # Same feature-noise marginal as SB3; no clipping or tanh likelihood substitution.
            weights = self.sde_log_std.clamp(*self.log_std_range).exp()
            position_std = (self.features.square() @ weights.square()).clamp_min(1e-12).sqrt()
        other_std = self.std_param[6:].clamp(*self.std_range).expand(*means.shape[:-1], 7)
        self._distribution = Normal(means, torch.cat((position_std, other_std), dim=-1))

    def begin_collection(self, step, num_envs):
        self.collecting = True
        if self.mode == 'gsde' and (self.noise_matrices is None or step % SDE_INTERVAL == 0):
            with torch.no_grad():
                scale = self.sde_log_std.clamp(*self.log_std_range).exp()
                self.noise_matrices = torch.randn(num_envs, 44, 6, device=scale.device) * scale

    def end_collection(self):
        self.collecting = False

    def reset_noise(self, dones):
        if self.mode != 'gsde' or self.noise_matrices is None or dones is None:
            return
        ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
        if ids.numel():
            with torch.no_grad():
                scale = self.sde_log_std.clamp(*self.log_std_range).exp()
                self.noise_matrices[ids] = torch.randn(ids.numel(), 44, 6, device=scale.device) * scale

    def sample(self):
        if self.mode != 'gsde' or not self.collecting:
            return super().sample()
        if self.noise_matrices.shape[0] != self.features.shape[0]:
            raise ValueError('gSDE requires one exploration matrix per rollout world')
        noise = torch.einsum('bf,bfa->ba', self.features, self.noise_matrices)
        self.last_position_noise = noise.detach().clone()
        other = super().sample()[..., 6:]
        return torch.cat((self.mean[..., :6] + noise, other), dim=-1)


class RewardExplorationActor(ImpactSpeedModel):
    observation_width = 43

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self.distribution) is not RewardExplorationGaussian:
            raise ValueError('032 actor requires its position-only distribution')
        self.std_head = None
        if self.distribution.mode == 'dependent':
            # Do not advance collection/critic initialization RNG for the extra head.
            with torch.random.fork_rng():
                self.std_head = nn.Linear(43, 6)
            nn.init.zeros_(self.std_head.weight)
            nn.init.constant_(self.std_head.bias, math.log(.3))

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)
        d = self.distribution
        if d.mode == 'dependent':
            d.position_log_std = self.std_head(latent)
        elif d.mode == 'gsde':
            features = torch.cat((latent.detach(), torch.ones_like(latent[..., :1])), dim=-1)
            d.features = features / features.norm(dim=-1, keepdim=True)
        return latent

    def reset(self, dones=None, hidden_state=None):
        super().reset(dones, hidden_state)
        self.distribution.reset_noise(dones)


class RewardExplorationPPO(CatPPO):
    """Only add collection-time gSDE sampling; retain all CaT reward/GAE code."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_step = 0

    def act(self, obs):
        d = self.actor.distribution
        d.begin_collection(self.collection_step, obs.batch_size[0])
        try:
            return super().act(obs)
        finally:
            d.end_collection()
            self.collection_step += 1

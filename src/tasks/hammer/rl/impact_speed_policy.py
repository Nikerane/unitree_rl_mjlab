"""Fixed-scale observations and masked likelihoods for the experimental speed actor."""

from __future__ import annotations

import math

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution


class ImpactSpeedNormalization(nn.Module):
    """Normalize legacy physical features; pass bounded goal and pending through."""

    def __init__(self) -> None:
        super().__init__()
        self.physical = EmpiricalNormalization(40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.physical(x[..., :40]), x[..., 40:]), dim=-1)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        self.physical.update(x[..., :40])


class ImpactSpeedGaussian(GaussianDistribution):
    """Diagonal Gaussian with per-transition selector participation."""

    activity_mask: torch.Tensor

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self.mean, self.std, self.activity_mask.clone()

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return (self._distribution.log_prob(outputs) * self.activity_mask).sum(dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        return (self._distribution.entropy() * self.activity_mask).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std, old_mask = old_params
        new_mean, new_std, new_mask = new_params
        if not torch.equal(old_mask, new_mask):
            raise ValueError("recorded action masks disagree")
        if not torch.all((old_mask == 0) | (old_mask == 1)):
            raise ValueError("recorded action masks must be binary")
        elementwise = (
            torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square())
            - 0.5
        )
        return (elementwise * old_mask).sum(dim=-1)


class ImpactSpeedModel(MLPModel):
    """One model type for 13-output actor or 1-output critic."""

    observation_width = 42

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )
        if self.obs_dim != self.observation_width:
            raise ValueError(f"{type(self).__name__} requires {self.observation_width} observations, got {self.obs_dim}")
        if not obs_normalization:
            raise ValueError("ImpactSpeedModel requires observation normalization")
        if self.distribution is not None and (output_dim != 13 or not isinstance(self.distribution, ImpactSpeedGaussian)):
            raise ValueError("stochastic impact-speed actor requires 13 outputs and ImpactSpeedGaussian")
        if self.distribution is None and output_dim != 1:
            raise ValueError("deterministic impact-speed critic requires one output")
        self.obs_normalizer = ImpactSpeedNormalization()
        first_linear = next(layer for layer in self.mlp if isinstance(layer, nn.Linear))
        with torch.no_grad():
            first_linear.weight[:, -1] = 0.0

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        if self.distribution is not None:
            raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            pending = raw[..., -1:]
            if not torch.all((pending == 0) | (pending == 1)):
                raise ValueError("raw pending flag must be binary")
            self.distribution.activity_mask = torch.cat((torch.ones_like(raw[..., :12]), pending), dim=-1)
        return super().get_latent(obs, masks, hidden_state)


class ImpactSpeedAgeModel(ImpactSpeedModel):
    """Physical40 + raw startup age + goal + pending; preserve the last-bit mask."""

    observation_width = 43

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        distribution = self.distribution
        if distribution is not None and (
                distribution.std_type != "scalar"
                or distribution.std_param.shape != (13,)
                or not distribution.std_param.requires_grad
                or not torch.equal(distribution.std_param.detach(), torch.ones_like(distribution.std_param))):
            raise ValueError("age treatment requires thirteen learnable unit-initialized scalar stds")


class ImpactSpeedLowNoiseModel(ImpactSpeedAgeModel):
    """Initialize only motion/gain exploration differently from the age control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        distribution = self.distribution
        if (distribution is None
                or not distribution.std_range[0] <= .3 <= distribution.std_range[1]):
            raise ValueError("low-noise treatment requires an actor supporting standard deviation 0.3")
        with torch.no_grad():
            distribution.std_param[:12].fill_(.3)


def _require_zero_pending_input(model: ImpactSpeedModel) -> None:
    """Qualification must not have learned a response to the unseen pending=1 flag."""
    if not isinstance(model, ImpactSpeedModel):
        raise ValueError("impact speed transfer requires an ImpactSpeedModel")
    first = next((layer for layer in model.mlp if isinstance(layer, nn.Linear)), None)
    if first is None or first.in_features != 42 or first.weight.shape[1] != 42:
        raise ValueError("impact speed transfer requires a 42-input first linear layer")
    if bool(torch.count_nonzero(first.weight[:, 41])):
        raise ValueError("qualification pending input column must be zero before transfer")


def initialize_selector_for_autonomy(actor: ImpactSpeedModel, *, initial_std: float = 1.0) -> None:
    """Start the autonomous selector at midpoint mean, without changing learned control."""
    _require_zero_pending_input(actor)
    final = next((layer for layer in reversed(actor.mlp) if isinstance(layer, nn.Linear)), None)
    if final is None or final.out_features != 13 or final.bias is None:
        raise ValueError("impact speed transfer requires a 13-output final linear layer")
    distribution = actor.distribution
    if (
        not isinstance(distribution, ImpactSpeedGaussian)
        or distribution.std_type != "scalar"
        or not isinstance(getattr(distribution, "std_param", None), nn.Parameter)
        or distribution.std_param.shape != (13,)
        or not distribution.std_param.requires_grad
    ):
        raise ValueError("selector transfer requires a learnable scalar std_param distribution")
    if (
        not math.isfinite(initial_std)
        or initial_std <= 0
        or not distribution.std_range[0] <= initial_std <= distribution.std_range[1]
    ):
        raise ValueError("initial_std must be finite, positive, and inside the distribution range")
    with torch.no_grad():
        final.weight[12].zero_()
        final.bias[12].zero_()
        distribution.std_param[12].fill_(initial_std)

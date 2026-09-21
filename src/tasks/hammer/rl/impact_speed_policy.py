"""Fixed-scale observations and masked likelihoods for the experimental speed actor."""

from __future__ import annotations

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
        if self.obs_dim != 42:
            raise ValueError(f"ImpactSpeedModel requires 42 observations, got {self.obs_dim}")
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

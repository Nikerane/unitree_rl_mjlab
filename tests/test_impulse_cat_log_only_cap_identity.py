"""Real-path identity regression for the log-only impulse-CaT control.

This deliberately uses the registered VIC-TT environment, wrapper, CatPPO, rollout storage,
dual-mask GAE, and optimizer. The harness changes only ``imp_limit`` between two seeded runs and
observes the actual one-update pathway; it does not reproduce any CaT or PPO calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import random
from typing import Any

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks.hammer.config.z1  # noqa: F401  (registers tasks)
from src.tasks.hammer.cat.hook import CatSoftHook
from src.tasks.hammer.rl.cat_ppo import CatPPO
from src.tasks.hammer.rl.cat_storage import CatRolloutStorage


pytestmark = pytest.mark.integration

VIC_TT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-VariableImpedance-TT"
)
HISTORICAL_DIAGNOSTIC_CAPS = (0.738, 1.476, 0.738, 0.738, 0.738, 0.738)
PROVISIONAL_CAPS = (0.82, 1.64, 0.82, 0.82, 0.82, 0.82)
SEED = 2
NUM_ENVS = 2


def _snapshot(value: Any) -> Any:
    """Detach mutable training state while preserving its exact nested representation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def _rng_state() -> dict[str, Any]:
    """Capture every host RNG used by mjlab's seeded CPU path."""
    return {
        "python": _snapshot(random.getstate()),
        "numpy": _snapshot(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def _assert_exact(left: Any, right: Any, *, path: str = "root") -> None:
    """Recursive bitwise equality with a useful path on the first divergence."""
    assert type(left) is type(right), f"{path}: {type(left)} != {type(right)}"
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype, f"{path}: dtype {left.dtype} != {right.dtype}"
        assert tuple(left.shape) == tuple(right.shape), (
            f"{path}: shape {tuple(left.shape)} != {tuple(right.shape)}"
        )
        assert torch.equal(left, right), f"{path}: tensor values differ"
        return
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype, f"{path}: dtype {left.dtype} != {right.dtype}"
        assert left.shape == right.shape, f"{path}: shape {left.shape} != {right.shape}"
        assert np.array_equal(left, right), f"{path}: array values differ"
        return
    if isinstance(left, Mapping):
        assert list(left) == list(right), f"{path}: keys differ"
        for key in left:
            _assert_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right), f"{path}: length differs"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    assert left == right, f"{path}: {left!r} != {right!r}"


def _hook_of(env: ManagerBasedRlEnv) -> CatSoftHook:
    manager = env.metrics_manager
    hook = manager._term_cfgs[manager._term_names.index("cat_soft")].func
    assert isinstance(hook, CatSoftHook)
    return hook


def _batch_snapshot(batch: Any) -> dict[str, Any]:
    return {
        "observations": _snapshot(batch.observations),
        "actions": _snapshot(batch.actions),
        "values": _snapshot(batch.values),
        "advantages": _snapshot(batch.advantages),
        "returns": _snapshot(batch.returns),
        "old_actions_log_prob": _snapshot(batch.old_actions_log_prob),
        "old_distribution_params": _snapshot(batch.old_distribution_params),
        "hidden_states": _snapshot(batch.hidden_states),
        "masks": _snapshot(batch.masks),
    }


def _run_one_update(tmp_path, caps: tuple[float, ...]) -> dict[str, Any]:
    """Run the real two-environment VIC-TT CatPPO path with one cap-only cfg mutation."""
    # ManagerBasedRlEnv.seed() resets Python, NumPy, Torch, and Warp. Seed here too so config
    # loading and construction before that call cannot depend on state left by the other arm.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_cfg = load_env_cfg(VIC_TT_TASK)
    env_cfg.seed = SEED
    env_cfg.scene.num_envs = NUM_ENVS
    cat_params = env_cfg.metrics["cat_soft"].params
    assert cat_params["imp_max_p"] == 0.0
    cat_params["imp_limit"] = list(caps)

    agent_cfg = load_rl_cfg(VIC_TT_TASK)
    runner_cfg = asdict(agent_cfg)
    runner_cfg["logger"] = "tensorboard"
    runner_cfg["upload_model"] = False
    assert runner_cfg["num_steps_per_env"] == 24

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(VIC_TT_TASK)
        runner = runner_cls(env, runner_cfg, str(tmp_path), "cpu")
        alg = runner.alg
        storage = alg.storage
        assert isinstance(alg, CatPPO)
        assert isinstance(storage, CatRolloutStorage)
        hook = _hook_of(raw_env)
        assert hook._imp_max_p == 0.0
        assert torch.equal(hook._imp_limit.cpu(), torch.tensor(caps, dtype=torch.float32))

        obs = env.get_observations().to("cpu")
        alg.train_mode()
        sampled_actions = []
        raw_rewards = []
        aggregate_delta = []
        hook_rng_states = []
        impulse_margins = []
        impulse_utilization = []
        impulse_delta = []
        velocity_delta = []
        lambda_per_joint = []

        initial_algorithm_state = _snapshot(alg.save())
        for _ in range(runner_cfg["num_steps_per_env"]):
            with torch.inference_mode():
                actions = alg.act(obs)
                sampled_actions.append(actions.detach().cpu().clone())
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                raw_rewards.append(rewards.detach().cpu().clone())
                aggregate_delta.append(extras[alg.DELTA_KEY].detach().cpu().clone())

                # Pull-only telemetry must itself leave RNG untouched.
                rng_before = _rng_state()
                telemetry = hook.constraint_telemetry()
                rng_after = _rng_state()
                _assert_exact(rng_before, rng_after, path="telemetry_rng_noop")
                hook_rng_states.append(rng_after)
                impulse_margins.append(telemetry["raw_margin_per_joint"].cpu())
                impulse_utilization.append(telemetry["cap_utilization_per_joint"].cpu())
                impulse_delta.append(telemetry["delta_impulse"].cpu())
                velocity_delta.append(telemetry["delta_velocity"].cpu())
                lambda_per_joint.append(telemetry["lambda_per_joint"].cpu())

                obs = obs.to("cpu")
                rewards = rewards.to("cpu")
                dones = dones.to("cpu")
                alg.process_env_step(obs, rewards, dones, extras)

        pre_return_storage = {
            "observations": _snapshot(storage.observations),
            "actions": _snapshot(storage.actions),
            "rewards": _snapshot(storage.rewards),
            "hard_dones": _snapshot(storage.dones),
            "soft_dones": _snapshot(storage.soft_dones),
            "values": _snapshot(storage.values),
            "actions_log_prob": _snapshot(storage.actions_log_prob),
            "distribution_params": _snapshot(storage.distribution_params),
        }
        alg.compute_returns(obs)
        returns = storage.returns.detach().cpu().clone()
        advantages = storage.advantages.detach().cpu().clone()

        # Observe the exact batches consumed by the real optimizer without changing their order.
        minibatches = []
        original_generator = storage.mini_batch_generator

        def recording_generator(num_mini_batches: int, num_epochs: int = 8):
            for batch in original_generator(num_mini_batches, num_epochs):
                minibatches.append(_batch_snapshot(batch))
                yield batch

        storage.mini_batch_generator = recording_generator
        rng_before_update = _rng_state()
        alg.update()

        return {
            "sampled_actions": torch.stack(sampled_actions),
            "raw_rewards": torch.stack(raw_rewards),
            "aggregate_delta": torch.stack(aggregate_delta),
            "scaled_rewards": pre_return_storage["rewards"],
            "soft_dones": pre_return_storage["soft_dones"],
            "pre_return_storage": pre_return_storage,
            "returns": returns,
            "advantages": advantages,
            "minibatches": minibatches,
            "initial_algorithm_state": initial_algorithm_state,
            "post_update_algorithm_state": _snapshot(alg.save()),
            "hook_rng_states": hook_rng_states,
            "rng_before_update": rng_before_update,
            "rng_after_update": _rng_state(),
            # These fields may reflect the cap-only mutation and are therefore not identity fields.
            "active_caps": hook._imp_limit.detach().cpu().clone(),
            "impulse_margins": torch.stack(impulse_margins),
            "impulse_utilization": torch.stack(impulse_utilization),
            "impulse_delta": torch.stack(impulse_delta),
            "velocity_delta": torch.stack(velocity_delta),
            "lambda_per_joint": torch.stack(lambda_per_joint),
        }
    finally:
        raw_env.close()


def test_log_only_impulse_cap_vector_is_exactly_training_invariant(tmp_path) -> None:
    """Changing only log-only caps cannot alter any action, PPO datum, RNG, or update."""
    historical = _run_one_update(tmp_path / "historical", HISTORICAL_DIAGNOSTIC_CAPS)
    provisional = _run_one_update(tmp_path / "provisional", PROVISIONAL_CAPS)

    cap_dependent = {"active_caps", "impulse_margins", "impulse_utilization"}
    assert set(historical) == set(provisional)
    for name in historical.keys() - cap_dependent:
        _assert_exact(historical[name], provisional[name], path=name)

    assert torch.equal(
        historical["active_caps"], torch.tensor(HISTORICAL_DIAGNOSTIC_CAPS)
    )
    assert torch.equal(provisional["active_caps"], torch.tensor(PROVISIONAL_CAPS))
    assert not torch.equal(historical["impulse_margins"], provisional["impulse_margins"])

    # The log-only arm is zero at every real control read in both runs, while the active velocity
    # arm and aggregate soft-OR remain covered by the exact equality assertions above.
    assert torch.count_nonzero(historical["impulse_delta"]) == 0
    assert torch.count_nonzero(provisional["impulse_delta"]) == 0
    for run in (historical, provisional):
        assert torch.equal(
            run["aggregate_delta"],
            torch.maximum(run["velocity_delta"], run["impulse_delta"]),
        )
        assert torch.isfinite(run["impulse_utilization"]).all()

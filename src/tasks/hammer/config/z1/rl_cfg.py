"""PPO configuration for the Z1 hammer-nail task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def z1_hammer_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Z1 hammer-nail task.

  Network sizes are smaller than locomotion tasks (manipulation has
  lower-dim obs/action), but architecture matches the repo pattern.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="z1_hammer",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=5000,
    # FUTURE_UPDATES #1 (applied 2026-06-10): bound Gaussian policy outputs.
    # Unclipped, iter-500 policies emitted actions up to ±7.8, which the IK
    # saturated into ~40 cm/step requests — hostile to sim-to-real and to the
    # action_rate penalty's assumed scale.
    clip_actions=1.0,
  )

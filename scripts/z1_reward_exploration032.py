"""One policy at a time through scripts.train; no scheduler or campaign logic."""
from dataclasses import replace
from pathlib import Path
from src.tasks.hammer.rl.reward_exploration032 import MODES

SEEDS = (2026110101, 2026110102, 2026110103)
_REGISTERED = False


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 9:
        raise ValueError('032 index must be integer 0..8')
    return dict(index=index, mode=MODES[index//3], seed=SEEDS[index%3], updates=500,
                worlds=1024, rollout_steps=24, checkpoints=(99,499))


def build_train_config(index):
    global _REGISTERED
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_reward_exploration032_cfg import register_tasks,task_id
    cell = cell_for_index(index)
    if not _REGISTERED:
        register_tasks();_REGISTERED = True
    task = task_id(cell['mode'])
    cfg = replace(TrainConfig.from_task(task), sampled_pd500=True, sampled_pd500_gain_mode='variable',
                  speed_bounds=(1.,1.5), speed_cost=.8493676, speed_mode='policy',
                  qualification_checkpoint=None, enable_nan_guard=True)
    cfg.env.scene.num_envs = 1024
    cfg.env.seed = cfg.agent.seed = cell['seed']
    cfg.agent.max_iterations = 500
    cfg.agent.resume = False
    cfg.agent.save_interval = 1
    cfg.agent.logger = 'tensorboard'
    return task,cfg


def run(index, output):
    from scripts.train import run_train
    output = Path(output)
    task,cfg = build_train_config(index)
    output.mkdir(parents=True,exist_ok=False)
    # Reuse the existing six-label filter, including model_99 and model_499.
    from scripts.z1_critic_noise_training021 import _checkpoint_saves
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.reward_exploration032_contract import validate_models,identity
    cell = cell_for_index(index)
    def recipe(runner, ignored):
        metadata = build_impact_speed_metadata(runner.env.unwrapped,runner.env.clip_actions)
        if metadata.get('spline_reward_exploration032') != identity(cell['mode']):
            raise ValueError('live 032 setting differs')
        validate_models(runner.alg.actor,runner.alg.critic,metadata)
        return metadata
    with _checkpoint_saves(cell,output,recipe=recipe):
        run_train(task,cfg,output/'train')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,required=True,choices=range(9))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.index,args.output)


if __name__ == '__main__':
    main()

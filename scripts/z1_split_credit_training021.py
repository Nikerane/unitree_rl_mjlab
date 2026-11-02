"""Fresh B6 split-credit500 adapter. No training or allocation authority."""
from dataclasses import replace
from scripts import z1_exploration_pilot_training as _pilot
from scripts import z1_timing_noise_training020 as _timing
from scripts.z1_critic_noise_training021 import _initial_gate,_checkpoint_saves

UPDATES,WORLDS,STEPS=500,1024,24
SAVE_INDICES=(0,24,49,99,249,499)
COMMON_INITIAL_HASHES=_pilot.COMMON_INITIAL_HASHES
_REGISTERED=False


def cell_for_index(index):
    if type(index) is not int or not 0<=index<6:raise ValueError('split index must be integer0..5')
    return dict(index=index,setting=('vic_baseline','vic_position_low')[index%2],seed=2026103101+index//2,
        critic_mode='current',updates=UPDATES,worlds=WORLDS,rollout_steps=STEPS,
        early_fraction=.5,early_deadline_ms=50,reserve_deadline_ms=200)


def build_train_config(index):
    global _REGISTERED
    cell=cell_for_index(index)
    from scripts.train import TrainConfig
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import register_split_tasks,split_task_id
    if not _REGISTERED:register_split_tasks();_REGISTERED=True
    task=split_task_id(setting=cell['setting'])
    cfg=replace(TrainConfig.from_task(task),sampled_pd500=True,sampled_pd500_gain_mode='variable',
        speed_bounds=(1.,1.5),speed_cost=.8493676,speed_mode='policy',qualification_checkpoint=None,enable_nan_guard=True)
    cfg.env.scene.num_envs=WORLDS;cfg.env.seed=cfg.agent.seed=cell['seed']
    cfg.agent.max_iterations=UPDATES;cfg.agent.resume=False;cfg.agent.save_interval=1;cfg.agent.logger='tensorboard'
    return task,cfg


def _live_recipe(runner,cell):
    from src.tasks.hammer.cat.hook import CatSoftHook
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.spline_split_credit_contract import validate_split_models,split_identity
    env=runner.env.unwrapped;hook=env.metrics_manager.cfg['cat_soft'].func
    if (type(hook) is not CatSoftHook or hook._max_p!=.5 or hook._imp_max_p!=.2
            or hook._velocity_pressure is not None or not hook._use_vel or not hook._use_impulse):
        raise ValueError('live constant velocity/impulse recipe differs')
    metadata=build_impact_speed_metadata(env,runner.env.clip_actions)
    if metadata.get('spline_split_credit')!=split_identity(cell['setting'],cell['early_fraction']):
        raise ValueError('live split treatment differs from selected cell')
    if (type(env.num_envs) is not int or env.num_envs!=WORLDS
            or type(runner.cfg['num_steps_per_env']) is not int or runner.cfg['num_steps_per_env']!=STEPS
            or type(runner.cfg['seed']) is not int or runner.cfg['seed']!=cell['seed']
            or type(env.cfg.seed) is not int or env.cfg.seed!=cell['seed']):
        raise ValueError('live worlds, rollout clock or matched seed differs')
    validate_split_models(runner.alg.actor,runner.alg.critic,metadata)
    return metadata


def matched_initial_state_gate(index,output_dir,*,enable_trace=True,reference_initial_state=None):
    return _initial_gate(cell_for_index(index),output_dir,enable_trace=enable_trace,
        reference_initial_state=reference_initial_state,
        reference_cell=_timing.cell_for_index(index//2*4+index%2),recipe=_live_recipe,
        marker_key='spline_split_credit',trace_module='src.tasks.hammer.rl.split_credit_training_trace',full_reference=True)


def selected_checkpoint_saves(index,output_dir):
    return _checkpoint_saves(cell_for_index(index),output_dir,recipe=_live_recipe)

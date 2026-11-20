"""Fresh six-cell Z1 safety factorial through the existing learner."""
from dataclasses import asdict,replace
from src.tasks.hammer.rl import safety_factorial_campaign as campaign

cell_for_index=campaign.cell_for_index


def build_config(cell,seed_index):
    if type(seed_index) is not int or seed_index not in (0,1,2):
        raise ValueError('safety factorial requires seed index0..2')
    marker=campaign.identity(cell)
    from scripts.z1_reward_transform import build_config as base
    if cell == 'B5' and seed_index != 1:
        raise ValueError('five-demo pilot requires seed index1')
    task,cfg=base(seed_index,'preserve_negative',reference_id=marker['reference_id'])
    campaign.configure(cfg.env,cell)
    cfg=replace(cfg,speed_cost=0.,sampled_pd500_gain_mode=marker['gain_mode'])
    cfg.agent.actor.distribution_cfg['fixed_nominal']=marker['gain_mode']=='fixed'
    cfg.agent.max_iterations=200
    return task,cfg


def main(argv=None):
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import torch
    import warp
    from scripts.train import run_train
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationRunner038
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.exploration038_contract import validate_models
    from src.tasks.hammer.rl.reward_transform_trace import attach_reward_transform_trace
    from src.tasks.hammer.rl.safety_factorial_state import checkpoint_state,SCHEMA as STATE_SCHEMA
    from src.tasks.hammer.rl.safety_factorial_trace import attach_tranche_trace,validate_trace
    from src.tasks.hammer.rl.u22_lookahead import verify_actor_mask
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',choices=campaign.CELLS,required=True)
    parser.add_argument('--seed-index',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-worlds',type=int,choices=(8,1024),default=1024)
    parser.add_argument('--initial-reference',type=Path)
    args=parser.parse_args(argv)
    if (args.smoke and args.initial_reference is not None) or (not args.smoke and args.initial_reference is None):
        parser.error('fresh production requires qualified same-seed B initialization; smoke does not consume a reference')
    if os.environ.get('Z1_PAIRED_CONTROL'):parser.error('factorial cannot mix a historical paired-control treatment')
    warp.config.use_precompiled_headers=False
    warp.config.kernel_cache_dir=os.environ['Z1_STUDY_CACHE']
    task,cfg=build_config(args.cell,args.seed_index)
    if args.smoke:cfg.env.scene.num_envs=args.smoke_worlds;cfg.agent.max_iterations=1
    updates=cfg.agent.max_iterations
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    cell=dict(campaign.cell_for_index(campaign.CELLS.index(args.cell)*3+args.seed_index),
        updates=updates,worlds=cfg.env.scene.num_envs,scope='smoke' if args.smoke else 'safety_factorial')
    (output/'recipe.json').write_text(json.dumps(dict(schema='z1_safety_factorial_recipe_v1',
        task=task,cell=cell,agent=asdict(cfg.agent),startup_ramp_s=0.,
        reward_weights={k:v.weight for k,v in cfg.env.rewards.items()},
        speed_bounds_mps=list(cfg.speed_bounds),ordinary_cat_state=STATE_SCHEMA),indent=2))
    original=ExplorationRunner038.learn
    def learn(runner,num_learning_iterations,init_at_random_ep_len=False):
        if num_learning_iterations!=updates or runner.current_learning_iteration!=0 or runner.alg.optimizer.state:
            raise ValueError('factorial requires one fresh fixed-budget learning call')
        contract=build_impact_speed_metadata(runner.env.unwrapped,runner.env.clip_actions)
        if campaign.validate_metadata(contract)!=cell[campaign.MARKER] or runner.alg.cat_reward_transform!='preserve_negative':
            raise ValueError('factorial live training identity differs')
        validate_models(runner.alg.actor,runner.alg.critic,contract)
        campaign.validate_runner_budget(runner,cell,smoke=args.smoke)
        runner.study_cell,runner.study_output_dir=cell,output
        with checkpoint_state(runner,args.cell):
            save,prime=runner.save,runner.alg.prime;written=set()
            names={'model_initial.pt','model_99.pt','model_199.pt',f'model_{updates-1}.pt'}
            def selected_save(path,infos=None):
                name=Path(path).name
                if name in names and name not in written:
                    save(path,infos);written.add(name)
            def prime_snapshot(obs):
                was_primed=runner.alg.normalization_primed;prime(obs)
                if not was_primed:
                    verify_actor_mask(runner.alg.actor,runner.alg.critic,obs)
                    if args.initial_reference:campaign.verify_initialization(output/'initial-state.json',args.initial_reference)
                    devices=[torch.device(runner.device).index or 0] if str(runner.device).startswith('cuda') else []
                    with torch.random.fork_rng(devices=devices):selected_save(str(output/'train/model_initial.pt'))
            runner.save,runner.alg.prime=selected_save,prime_snapshot
            try:
                with attach_reward_transform_trace(runner,output/'reward-transform.jsonl',max_updates=updates):
                    with attach_tranche_trace(runner,output/'tranche-pressure.jsonl',max_updates=updates):
                        return original(runner,num_learning_iterations,False)
            finally:runner.save,runner.alg.prime=save,prime
    ExplorationRunner038.learn=learn
    try:run_train(task,cfg,output/'train')
    finally:ExplorationRunner038.learn=original
    if campaign.verify_zero_ledger(output/'reward-transform.jsonl')!=updates:
        raise ValueError('factorial reward ledger is incomplete')
    validate_trace(output/'tranche-pressure.jsonl',updates)
    final=output/'train'/f'model_{updates-1}.pt';payload=torch.load(final,map_location='cpu',weights_only=False)
    infos=payload['infos'];clock=infos['env_state']['common_step_counter']
    if (clock!=24*updates or campaign.validate_metadata(infos['impact_speed_contract'])!=cell[campaign.MARKER]
            or infos.get(campaign.MARKER)!=cell[campaign.MARKER]
            or infos['ordinary_cat_state']['common_step_counter']!=clock
            or infos['cat_reward_transform']!='preserve_negative'):
        raise ValueError('factorial saved checkpoint budget or identity differs')
    def finite(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return True
    if not finite(payload):raise ValueError('nonfinite factorial checkpoint')
    (output/'training-complete.json').write_text(json.dumps(dict(schema='z1_safety_factorial_complete_v1',
        cell=cell,completed_updates=updates,common_step_counter=clock,transitions=clock*cfg.env.scene.num_envs,
        final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),finite_checkpoint=True,no_resume=True,
        ordinary_cat_state_saved=True,tranche_trace_updates=updates),indent=2))
    print('TRAINING_COMPLETE',args.cell,cell['seed'],updates,flush=True)


if __name__=='__main__':main()

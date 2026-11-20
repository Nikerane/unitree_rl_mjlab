"""Six paired reward-transform runs through the existing exploration038 trainer."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path

from mjlab.rl.config import RslRlPpoAlgorithmCfg
from scripts.z1_exploration038 import build_train_config, cell_for_index

TRANSFORMS = ('preserve_negative', 'reference_clipped')


@dataclass
class RewardTransformAlgorithmCfg(RslRlPpoAlgorithmCfg):
    cat_reward_transform: str = 'preserve_negative'


def build_config(index, transform, *, reference_id='D22', demo_progress_weight=None, reward_grid_setting=None, split_impulse_setting=None):
    if transform not in TRANSFORMS or type(index) is not int or index not in (0, 1, 2):
        raise ValueError('requires one of two transforms and paired seed index0..2')
    task, cfg = build_train_config(index, reference_id=reference_id)
    cfg.agent.algorithm = RewardTransformAlgorithmCfg(
        **asdict(cfg.agent.algorithm), cat_reward_transform=transform)
    if demo_progress_weight is not None:
        if transform != 'preserve_negative':
            raise ValueError('demo campaign requires preserve_negative')
        from src.tasks.hammer.rl.demo_progress_campaign import configure
        configure(cfg.env, reference_id, demo_progress_weight)
        cfg = replace(cfg, speed_cost=0.)
    if reward_grid_setting is not None:
        if split_impulse_setting is not None:
            raise ValueError("split impulse cannot mix reward grid")
        if reference_id != 'U22' or transform != 'preserve_negative' or demo_progress_weight is not None:
            raise ValueError('reward grid requires U22 preserve_negative without demo campaign')
        from src.tasks.hammer.rl.reward_grid_campaign import configure
        configure(cfg.env, reward_grid_setting)
        cfg = replace(cfg, speed_cost=0.)
        cfg.agent.max_iterations = 200
    if split_impulse_setting is not None:
        if reference_id != 'U22' or transform != 'preserve_negative' or demo_progress_weight is not None:
            raise ValueError('split impulse requires U22 preserve_negative without demo campaign')
        from src.tasks.hammer.rl.split_impulse_campaign import configure
        configure(cfg.env,split_impulse_setting)
        cfg = replace(cfg,speed_cost=0.)
        cfg.agent.max_iterations = 200
    return task, cfg


def main(argv=None):
    import torch
    import warp
    from scripts.train import run_train
    from src.tasks.hammer.rl.reward_exploration038 import ExplorationRunner038
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.rl.reward_transform_trace import attach_reward_transform_trace
    from scripts.z1_exploration038 import _live_recipe
    from src.tasks.hammer.mdp.u22_timing import MODES, install_u22_timing, timing_identity
    from src.tasks.hammer.rl import u22_lookahead, demo_progress_campaign as demo, reward_grid_campaign as grid, split_impulse_campaign as split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--transform', choices=TRANSFORMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='8 worlds, one update; not study evidence')
    parser.add_argument('--smoke-worlds', type=int, choices=(8, 1024), default=8)
    parser.add_argument('--peak-j3-budget-nm', type=float)
    parser.add_argument('--peak-budgets-nm', nargs=6, type=float)
    parser.add_argument('--precontact-motor-demand', action='store_true')
    parser.add_argument('--motor-max-p', type=float, choices=(.2, .5))
    parser.add_argument('--u22-timing', choices=MODES)
    parser.add_argument('--u22-lookahead', choices=u22_lookahead.MODES)
    parser.add_argument('--u22-path-controls', type=Path)
    parser.add_argument('--demo-reference', choices=demo.REFERENCES)
    parser.add_argument('--demo-progress-weight', type=float, choices=demo.PROGRESS_WEIGHTS)
    parser.add_argument('--reward-grid-setting', choices=grid.SETTINGS)
    parser.add_argument('--reward-grid-initial', type=Path)
    parser.add_argument('--split-impulse-setting', choices=split.SETTINGS)
    parser.add_argument('--split-impulse-initial', type=Path)
    args = parser.parse_args(argv)
    if args.split_impulse_initial and (not args.split_impulse_setting or args.smoke):
        parser.error('--split-impulse-initial requires a production split impulse run')
    if args.split_impulse_setting and not args.smoke and args.split_impulse_initial is None:
        parser.error('production split impulse requires --split-impulse-initial from qualification')
    if args.split_impulse_setting and (args.reward_grid_setting or args.reward_grid_initial
            or args.demo_reference or args.demo_progress_weight is not None or args.u22_timing
            or args.u22_lookahead or args.u22_path_controls or args.transform != 'preserve_negative'
            or args.precontact_motor_demand or args.motor_max_p is not None
            or args.peak_budgets_nm is not None or args.peak_j3_budget_nm is not None
            or os.environ.get('Z1_PAIRED_CONTROL')):
        parser.error('split impulse requires its fresh native recipe without other treatments')
    if args.reward_grid_initial and (not args.reward_grid_setting or args.smoke):
        parser.error('--reward-grid-initial requires a production reward grid run')
    if args.reward_grid_setting and not args.smoke and args.reward_grid_initial is None:
        parser.error('production reward grid requires --reward-grid-initial from qualification')
    if args.reward_grid_setting and (args.demo_reference or args.demo_progress_weight is not None
            or args.u22_timing or args.u22_lookahead or args.u22_path_controls
            or args.transform != 'preserve_negative' or args.precontact_motor_demand
            or args.motor_max_p is not None or args.peak_budgets_nm is not None
            or args.peak_j3_budget_nm is not None or os.environ.get('Z1_PAIRED_CONTROL')):
        parser.error('reward grid requires its fresh native recipe without other treatments')
    if (args.demo_reference is None) != (args.demo_progress_weight is None):
        parser.error('demo campaign requires both --demo-reference and --demo-progress-weight')
    if args.demo_reference and (args.u22_timing or args.u22_lookahead or args.u22_path_controls
            or args.transform != 'preserve_negative' or args.precontact_motor_demand
            or args.motor_max_p is not None or args.peak_budgets_nm is not None
            or args.peak_j3_budget_nm is not None or os.environ.get('Z1_PAIRED_CONTROL')):
        parser.error('demo campaign requires its fresh native constraint recipe without other treatments')
    if args.u22_lookahead and (args.u22_timing != 'path' or args.transform != 'preserve_negative'
            or args.u22_path_controls is None or (args.smoke and args.smoke_worlds != 1024)):
        parser.error('lookahead ablation requires the matched U22 path recipe and baseline runs')
    if args.motor_max_p is not None and not args.precontact_motor_demand:
        parser.error('--motor-max-p requires --precontact-motor-demand')
    motor_max_p = .2 if args.motor_max_p is None else args.motor_max_p
    paired_control = os.environ.get('Z1_PAIRED_CONTROL')
    if args.u22_timing and (paired_control or args.precontact_motor_demand or
            args.peak_budgets_nm is not None or args.peak_j3_budget_nm is not None):
        parser.error('U22 timing uses fresh controls and the native CaT channels')
    if motor_max_p == .5 and not args.smoke and not paired_control:
        parser.error('the 0.50 study requires a paired 0.20 control')
    if args.precontact_motor_demand and (args.peak_budgets_nm is not None or
            args.peak_j3_budget_nm is not None or args.transform != 'preserve_negative'):
        parser.error('motor-demand treatment requires the original preserve_negative recipe')
    if args.peak_budgets_nm is not None and args.peak_j3_budget_nm is not None:
        parser.error('choose one peak treatment')
    if (args.peak_j3_budget_nm is not None or args.peak_budgets_nm is not None) and args.transform != 'preserve_negative':
        parser.error('the paired peak study uses preserve_negative controls')
    warp.config.use_precompiled_headers = False
    warp.config.kernel_cache_dir = os.environ['Z1_STUDY_CACHE']
    reference_id = args.demo_reference or ('U22' if args.u22_timing or args.reward_grid_setting or args.split_impulse_setting else 'D22')
    task, cfg = build_config(args.index, args.transform, reference_id=reference_id,
                             demo_progress_weight=args.demo_progress_weight, reward_grid_setting=args.reward_grid_setting, split_impulse_setting=args.split_impulse_setting)
    if args.smoke:
        cfg.env.scene.num_envs = args.smoke_worlds
        cfg.agent.max_iterations = 1
    updates = cfg.agent.max_iterations
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cell = dict(cell_for_index(args.index), reference_id=reference_id,
                reward_transform=args.transform, updates=updates,
                worlds=cfg.env.scene.num_envs, scope='smoke' if args.smoke else 'paired_transform_ablation')
    if args.demo_reference:
        cell[demo.MARKER] = demo.identity(reference_id, args.demo_progress_weight)
        cell['scope'] = 'smoke' if args.smoke else cell[demo.MARKER]['scope']
    if args.reward_grid_setting:
        for name in ('early_fraction','early_deadline_ms','reserve_deadline_ms'):
            cell.pop(name)
        cell.update(payout='reward_grid_50ms',event_payment_ms=50)
        cell[grid.MARKER] = grid.identity(args.reward_grid_setting)
        cell['scope'] = 'smoke' if args.smoke else 'reward_grid'
    if args.split_impulse_setting:
        for name in ('early_fraction','early_deadline_ms','reserve_deadline_ms'):
            cell.pop(name)
        cell.update(payout='split_impulse_50_200ms',event_payment_ms=[50,200])
        cell[split.MARKER] = split.identity(args.split_impulse_setting)
        cell['scope'] = 'smoke' if args.smoke else 'split_impulse'
    if args.u22_timing:
        cell['timing'] = timing_identity(args.u22_timing)
    reward_weights = {k:v.weight for k,v in cfg.env.rewards.items()}
    if args.u22_lookahead:
        cell[u22_lookahead.MARKER] = u22_lookahead.identity(args.u22_lookahead)
        reward_weights['spline_progress'] = cell[u22_lookahead.MARKER]['progress_weight']
    if args.peak_j3_budget_nm is not None:
        cell['peak_contact_torque'] = dict(joint='joint3', budget_nm=args.peak_j3_budget_nm,
                                          budget_kind='experimental')
    if args.peak_budgets_nm is not None:
        cell['peak_contact_torque'] = dict(budgets_nm=args.peak_budgets_nm, budget_kind='experimental')
    if args.precontact_motor_demand:
        cell['precontact_motor_demand'] = dict(max_p=motor_max_p, normalization_floor=.1,
                                               joints=[1,2,3,4,5,6])
    final_name = f'model_{updates-1}.pt'
    saved_names = {'model_initial.pt', 'model_99.pt', 'model_199.pt', 'model_499.pt', final_name}
    (output/'recipe.json').write_text(json.dumps(dict(
        schema='z1_reward_transform_recipe_v1', task=task, cell=cell,
        agent=asdict(cfg.agent), startup_ramp_s=0.,
        reward_weights=reward_weights,
        speed_bounds_mps=cfg.speed_bounds,
        changed=('split the same J50 budget across50/200ms and vary progress weight' if args.split_impulse_setting
                 else 'reward grid: vary only impulse/progress weights and delayed binary amount; all event payments50ms' if args.reward_grid_setting
                 else 'demonstration-only physical recipe and normalized actor lookahead mask; vary only progress weight' if args.demo_reference
                 else 'U22 reference guidance timing' if args.u22_timing
                 else 'raise precontact motor-demand CaT ceiling from 0.20 to 0.50' if motor_max_p == .5
                 else 'add precontact motor-demand CaT channel' if args.precontact_motor_demand
                 else 'add peak contact torque CaT channel' if 'peak_contact_torque' in cell
                 else 'only PPO immediate reward transform; identical soft-CaT GAE and timeout bootstrap')), indent=2))
    original = ExplorationRunner038.learn

    def paired_learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations != updates or runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
            raise ValueError('one fresh fixed-budget learning call required')
        if args.u22_timing:
            install_u22_timing(runner.env.unwrapped, args.u22_timing)
            runner.alg.actor.use_selector_as_clock = args.u22_timing == 'adaptive'
            init_at_random_ep_len = False
        if args.demo_reference or args.reward_grid_setting or args.split_impulse_setting:
            init_at_random_ep_len = False
        if args.u22_lookahead:
            u22_lookahead.install(runner, args.u22_lookahead)
        contract = (build_impact_speed_metadata(runner.env.unwrapped, runner.env.clip_actions)
                    if args.smoke or args.demo_reference or args.reward_grid_setting or args.split_impulse_setting else _live_recipe(runner, cell))
        if (runner.env.unwrapped._sampled_pd500.startup_ramp_s != 0.
                or contract['spline_exploration038']['impulse']['weight'] != (1. if args.split_impulse_setting else grid.identity(args.reward_grid_setting)['impulse_event_weight'] if args.reward_grid_setting else 8. if args.demo_reference else 4.)
                or contract['speed_bounds_mps'] != [1., 1.5]
                or runner.alg.cat_reward_transform != args.transform):
            raise ValueError('live treatment or fixed recipe differs')
        if args.demo_reference:
            if contract.get(demo.MARKER) != cell[demo.MARKER]:
                raise ValueError('live demo campaign identity differs')
            from src.tasks.hammer.rl.exploration038_contract import validate_models
            validate_models(runner.alg.actor, runner.alg.critic, contract)
            demo.validate_runner_budget(runner,cell,smoke=args.smoke)
        if args.reward_grid_setting:
            if contract.get(grid.MARKER) != cell[grid.MARKER]:
                raise ValueError('live reward grid identity differs')
            from src.tasks.hammer.rl.exploration038_contract import validate_models
            validate_models(runner.alg.actor, runner.alg.critic, contract)
            grid.validate_runner_budget(runner,cell,smoke=args.smoke)
        if args.split_impulse_setting:
            if contract.get(split.MARKER) != cell[split.MARKER]:
                raise ValueError('live split impulse identity differs')
            from src.tasks.hammer.rl.exploration038_contract import validate_models
            validate_models(runner.alg.actor,runner.alg.critic,contract)
            split.validate_runner_budget(runner,cell,smoke=args.smoke)
        runner.study_cell, runner.study_output_dir = cell, output
        save, prime = runner.save, runner.alg.prime
        written = set()

        def selected_save(path, infos=None):
            name = Path(path).name
            if name in saved_names and name not in written:
                if 'peak_contact_torque' in cell:
                    infos = dict(infos or {}, peak_contact_torque=cell['peak_contact_torque'])
                if args.precontact_motor_demand:
                    infos = dict(infos or {}, precontact_motor_demand=cell['precontact_motor_demand'])
                save(path, infos)
                written.add(name)

        def prime_and_snapshot(obs):
            was_primed = runner.alg.normalization_primed
            prime(obs)
            if not was_primed:
                if args.split_impulse_setting:
                    u22_lookahead.verify_actor_mask(runner.alg.actor,runner.alg.critic,obs)
                    if args.split_impulse_initial:
                        split.verify_initialization(output/'initial-state.json',args.split_impulse_initial)
                if args.reward_grid_setting:
                    u22_lookahead.verify_actor_mask(runner.alg.actor, runner.alg.critic, obs)
                    if args.reward_grid_initial:
                        grid.verify_initialization(output/'initial-state.json', args.reward_grid_initial)
                if args.demo_reference:
                    u22_lookahead.verify_actor_mask(runner.alg.actor, runner.alg.critic, obs)
                if args.u22_lookahead:
                    control = args.u22_path_controls / f'path-s{args.index}'
                    u22_lookahead.verify_initialization(output, control, args.u22_lookahead)
                    if args.smoke:
                        u22_lookahead.verify_actor_mask(runner.alg.actor, runner.alg.critic, obs)
                if paired_control:
                    control = Path(paired_control)
                    expected = json.loads((control/'initial-state.json').read_text())
                    actual = json.loads((output/'initial-state.json').read_text())
                    if motor_max_p == .5:
                        key = 'precontact_motor_demand'
                        if actual['cell'][key]['max_p'] != .5 or expected['cell'][key]['max_p'] != .2:
                            raise ValueError('expected paired motor ceilings of 0.50 and 0.20')
                        actual['cell'][key]['max_p'] = .2
                        recipe = json.loads((output/'recipe.json').read_text())
                        reference_recipe = json.loads((control/'recipe.json').read_text())
                        recipe['cell'][key]['max_p'] = .2
                        recipe['changed'] = reference_recipe['changed']
                        if recipe != reference_recipe:
                            raise ValueError('training recipe differs from the paired 0.20 control')
                    else:
                        actual['cell'].pop('precontact_motor_demand' if args.precontact_motor_demand
                                           else 'peak_contact_torque')
                    if actual != expected:
                        raise ValueError('treatment initialization differs from its paired control')
                devices = [torch.device(runner.device).index or 0] if str(runner.device).startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    selected_save(str(output/'train/model_initial.pt'))

        if args.peak_j3_budget_nm is not None:
            from src.tasks.hammer.rl.peak_contact_torque import install_peak_contact_torque
            install_peak_contact_torque(runner, joint=2, budget_nm=args.peak_j3_budget_nm)
        if args.peak_budgets_nm is not None:
            from src.tasks.hammer.rl.peak_contact_torque import install_peak_contact_torque
            install_peak_contact_torque(runner, budget_nm=args.peak_budgets_nm)
        if args.precontact_motor_demand:
            from src.tasks.hammer.rl.motor_demand import install_motor_demand
            install_motor_demand(runner, max_p=motor_max_p)
        runner.save, runner.alg.prime = selected_save, prime_and_snapshot
        try:
            with attach_reward_transform_trace(runner, output/'reward-transform.jsonl', max_updates=updates):
                return original(runner, num_learning_iterations, init_at_random_ep_len)
        finally:
            runner.save, runner.alg.prime = save, prime

    ExplorationRunner038.learn = paired_learn
    try:
        run_train(task, cfg, output/'train')
    finally:
        ExplorationRunner038.learn = original
    if args.reward_grid_setting and grid.verify_zero_ledger(output/'reward-transform.jsonl') != updates:
        raise ValueError('reward grid ledger does not cover every update')
    if args.demo_reference and demo.verify_zero_ledger(output/'reward-transform.jsonl') != updates:
        raise ValueError('demo campaign reward ledger does not cover every update')
    if args.split_impulse_setting and split.verify_zero_ledger(output/'reward-transform.jsonl') != updates:
        raise ValueError('split impulse ledger does not cover every update')
    final_path = output/'train'/final_name
    payload = torch.load(final_path, map_location='cpu', weights_only=False)
    if args.demo_reference:
        demo.validate_metadata(payload['infos']['impact_speed_contract'])
        if payload['infos']['impact_speed_contract'][demo.MARKER] != cell[demo.MARKER]:
            raise ValueError('saved demo identity differs from selected treatment')
    if args.reward_grid_setting:
        grid.validate_metadata(payload['infos']['impact_speed_contract'])
        if payload['infos']['impact_speed_contract'][grid.MARKER] != cell[grid.MARKER]:
            raise ValueError('saved reward grid identity differs')
    if args.split_impulse_setting:
        split.validate_metadata(payload['infos']['impact_speed_contract'])
        if payload['infos']['impact_speed_contract'][split.MARKER] != cell[split.MARKER]:
            raise ValueError('saved split impulse identity differs')
    clock = payload['infos']['env_state']['common_step_counter']
    if clock != updates*24 or payload['infos']['cat_reward_transform'] != args.transform:
        raise ValueError('saved budget/transform identity differs')
    if args.u22_timing and payload['infos']['impact_speed_contract']['human_reference']['timing'] != cell['timing']:
        raise ValueError('saved timing contract differs from the run recipe')

    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True
    if not finite(payload):
        raise ValueError('nonfinite checkpoint')
    (output/'training-complete.json').write_text(json.dumps(dict(
        schema='z1_reward_transform_complete_v1', cell=cell, completed_updates=updates,
        common_step_counter=clock, transitions=clock*cfg.env.scene.num_envs,
        final_checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
        finite_checkpoint=True, no_resume=True), indent=2))
    print('TRAINING_COMPLETE', args.transform, cell['seed'], updates, flush=True)


if __name__ == '__main__':
    main()

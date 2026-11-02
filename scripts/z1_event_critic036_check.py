"""One technical PPO update/save/full reload; configure compiler in outer bootstrap."""
import argparse
from pathlib import Path


def height_reset_evidence(runner):
    """Read two native resets before learning; no extra environment or policy step."""
    import torch
    from scripts.z1_event_critic036_eval import _need
    from scripts.z1_credit_critic_training import tensor_state_digest
    env = runner.env.unwrapped
    asset = env.scene['nail_block']
    selection = env.event_manager.get_term_cfg('randomize_board_nail_height037').params['asset_cfg']
    global_ids = asset.indexing.body_ids[selection.body_ids]
    _need(global_ids.numel() == 2 and tuple(selection.body_names) == ('block','nail'),
          'technical height check requires board and nail body IDs')
    defaults = env.sim.get_default_field('body_pos')
    base = (defaults[:,global_ids] if 'body_pos' in env.sim.per_world_default_fields
            else defaults[global_ids].unsqueeze(0).expand(env.num_envs,-1,-1))
    resets = []
    previous = None
    for index in range(2):
        runner.env.reset()
        _need(type(env.common_step_counter) is int and env.common_step_counter == 0,
              'height qualification must precede all physics policy steps')
        actual = env.sim.model.body_pos[:,global_ids]
        shift = actual-base
        _need(actual.shape == (env.num_envs,2,3) and bool(torch.isfinite(actual).all()),
              'finite expanded height model body positions required')
        offset = shift[:,0,2]
        _need(bool((shift[:,:,:2].abs() <= 1e-7).all())
              and bool((shift[:,1,2]-offset).abs().le(1e-7).all())
              and bool(offset.abs().le(.005+1e-7).all())
              and bool(offset.abs().gt(1e-6).any()), 'shared board/nail height shift or range differs')
        if previous is not None:
            _need(bool((offset-previous).abs().gt(1e-6).any()), 'native height reset must redraw from defaults')
        previous = offset.detach().clone()
        _need(bool((asset.data.joint_pos == 0).all()), 'height reset must preserve zero nail slide depth')
        site = asset.data.site_pos_w[:,env._hammer_first_strike._nail_site_ids].squeeze(1)
        expected = torch.stack((offset.new_full(offset.shape,.5),torch.zeros_like(offset),.102+offset),-1)
        ref = env._strike_reference
        target = site.clone();target[:,2] -= ref.overshoot
        close = lambda left,right: left.shape == right.shape and bool(torch.isfinite(left).all()) and bool((left-right).abs().le(2e-6).all())
        _need(close(site,expected) and close(ref._nail,site) and close(ref._target,target)
              and bool(ref._anchored.all()), 'native nail site/reference must follow physical height shift')
        tensors = dict(height_offset_m=offset,body_positions_m=actual,nail_top_world_m=site,
                       reference_nail_world_m=ref._nail,reference_target_world_m=ref._target,
                       native_nail_slide_qpos_m=asset.data.joint_pos)
        sample_count = min(4,env.num_envs)
        resets.append(dict(index=index,
            all_world_summary=dict(world_count=env.num_envs,
                offset_min_m=float(offset.min()),offset_max_m=float(offset.max()),offset_mean_m=float(offset.mean()),
                max_xy_shift_m=float(shift[:,:,:2].abs().max()),
                max_shared_z_shift_error_m=float((shift[:,1,2]-offset).abs().max()),
                max_nail_site_error_m=float((site-expected).abs().max()),
                max_reference_nail_error_m=float((ref._nail-site).abs().max()),
                max_reference_target_error_m=float((ref._target-target).abs().max()),
                max_nail_slide_qpos_m=float(asset.data.joint_pos.abs().max())),
            all_world_tensor_sha256={name:tensor_state_digest({name:value}) for name,value in tensors.items()},
            all_world_tensor_shapes={name:list(value.shape) for name,value in tensors.items()},
            sample_worlds=dict(scope='EXPLICIT_SAMPLE_ONLY_NOT_ALL_WORLDS',world_ids=list(range(sample_count)),
                **{name:value[:sample_count].detach().cpu().tolist() for name,value in tensors.items()})))
    return dict(schema='z1_reward037_height_reset_technical_v1',status='PASS_TWO_NATIVE_HEIGHT_RESETS_ONLY',
                body_names=['block','nail'],global_body_ids=global_ids.detach().cpu().tolist(),
                offset_range_m=[-.005,.005],verified_worlds=env.num_envs,
                training_updates=0,physics_steps=0,resets=resets)


def run_check(index, output, *, study='event036'):
    import torch
    from scripts import train
    from scripts.z1_event_critic036_eval import _study_module, _schema
    selected_training = _study_module(study)
    from scripts.z1_credit_critic_training import _write_once, _file_sha, tensor_state_digest
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    task, cfg = selected_training.build_train_config(index)
    cfg.agent.max_iterations = 1
    cfg.agent.save_interval = 1
    cell = dict(selected_training.cell_for_index(index), updates=1)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = HammerOnPolicyRunner.learn
    results = []
    def learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        if (results or type(num_learning_iterations) is not int or num_learning_iterations != 1
                or runner.current_learning_iteration != 0 or runner.env.unwrapped.common_step_counter != 0
                or runner.cfg.get('resume') is not False or runner.alg.optimizer.state):
            raise ValueError('one fresh technical learning update required')
        metadata = selected_training._live_recipe(runner, cell)
        height_qualification = None
        if cell.get('height_dr') is True:
            height_row = height_reset_evidence(runner)
            height_path = output/'height-reset-technical.json'
            _write_once(height_path,height_row)
            height_qualification = dict(path=str(height_path),sha256=_file_sha(height_path),status=height_row['status'])
        key = 'spline_reward037' if study == 'reward037' else ('spline_split_event036' if cell['payout'] == 'split' else 'spline_exploration_pilot')
        observations = selected_training._pilot._initial_observation_hashes(runner.env.unwrapped, metadata[key])
        hashes, std = selected_training._pilot._initialize(runner, cell)
        _write_once(output/'initial-state.json', dict(schema=_schema(study, 'initial', technical=True),
            cell=cell, phase='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT', actual_counter=0, critic_width=47,
            contract=metadata, hashes=hashes, initial_std=std, initial_observations=observations,
            model_initialization_seed=cell['seed'], collection_seed=cell['seed']))
        result = original(runner, 1, init_at_random_ep_len)
        checkpoint = output/'train/model_0.pt'
        payload = torch.load(checkpoint, map_location=runner.device, weights_only=False)
        if (type(payload.get('iter')) is not int or payload['iter'] != 0
                or payload.get('infos', {}).get('env_state', {}).get('common_step_counter') != 24
                or runner.env.unwrapped.common_step_counter != 24
                or payload['infos'].get('impact_speed_contract') != metadata):
            raise ValueError('technical model0 saved clock/metadata differs')
        saved = {name:payload[name+'_state_dict'] for name in ('actor','critic')}
        with torch.inference_mode():
            for name in saved:
                for parameter in getattr(runner.alg, name).parameters():
                    parameter.zero_()
                for buffer in getattr(runner.alg, name).buffers():
                    buffer.fill_(17)
        runner.env.unwrapped.common_step_counter = 17
        zeroed = {name:sum(int(torch.count_nonzero(p)) for p in getattr(runner.alg,name).parameters()) for name in saved}
        if any(zeroed.values()):
            raise ValueError('technical parameter-zeroing failed')
        _write_once(output/'before-full-reload.json', dict(parameter_nonzero_counts=zeroed, checkpoint_label=0,
            buffer_sentinel=17, common_step_counter=runner.env.unwrapped.common_step_counter))
        with torch.inference_mode():
            runner.load(str(checkpoint), load_cfg={'actor':True,'critic':True}, strict=True)
        if type(runner.env.unwrapped.common_step_counter) is not int or runner.env.unwrapped.common_step_counter != 24:
            raise ValueError('technical full reload transition clock differs')
        reloaded = {}
        for name, expected in saved.items():
            actual = getattr(runner.alg, name).state_dict()
            if set(actual) != set(expected) or any(not torch.equal(actual[k],v) for k,v in expected.items()):
                raise ValueError('technical full '+name+' reload differs from saved tensors')
            reloaded[name] = tensor_state_digest(actual)
        row = dict(schema=_schema(study, 'result', technical=True), status='PASS_UPDATE_SAVE_FULL_RELOAD_ONLY',
            cell=cell, completed_updates=1, common_step_counter=24, checkpoint_label=0,
            checkpoint_sha256=_file_sha(checkpoint), initial_hashes=hashes, reloaded_hashes=reloaded,
            loaded_model_components=['actor','critic'], contract=metadata, study_trials=0,
            parameter_nonzero_before_reload=zeroed, rollout_transitions=1024*24)
        if height_qualification is not None:
            row['height_reset_qualification'] = height_qualification
        _write_once(output/'technical-result.json', row)
        results.append(row)
        return result
    HammerOnPolicyRunner.learn = learn
    try:
        train.run_train(task, cfg, output/'train')
        if len(results) != 1:
            raise ValueError('technical trainer must execute one learning call')
        return results[0]
    finally:
        HammerOnPolicyRunner.learn = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('event036','reward037'), default='event036')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return run_check(args.index, args.output, **({'study':args.study} if args.study != 'event036' else {}))


if __name__ == '__main__':
    main()

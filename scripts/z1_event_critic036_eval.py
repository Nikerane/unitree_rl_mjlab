"""One CPU32 matched-bank checkpoint replay using preserved measurement code."""
import argparse
import ast
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True  # The preserved vendor closure includes every file.

from scripts import z1_event_critic036 as training
from scripts.z1_thirty_evaluation023.records import finite_tree, validate_state_shapes

VENDOR = Path(__file__).with_name('z1_thirty_evaluation023') / 'vendor'
BANK = VENDOR / 'reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json'
BANK_SHA = 'b0baebc9dee774c89091a16ed7ba6ce5bfb3df4e4a639b6faabf56ec9da43f22'
BANK_ADAPTER_SHA = '63cb613f85f7e3121690fd15b56eb3c3816ad3ac03c154e0385cb9010ab44379'
EVALUATION_SEED = 2026103001
FIELDS = {'schema', 'policy_id', 'training_cell', 'checkpoint_label', 'completed_updates',
          'common_step_counter', 'task', 'checkpoint_path', 'checkpoint_sha256',
          'initial_state_sha256', 'impact_speed_contract'}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _study_module(study):
    _need(type(study) is str and study in ('event036','reward037'), 'selected study must be event036 or reward037')
    if study == 'reward037':
        from scripts import z1_reward_training037
        return z1_reward_training037
    return training


def _schema(study, kind, *, technical=False):
    _study_module(study)
    prefix = 'z1_reward037' if study == 'reward037' else 'z1_event_critic036'
    return prefix + ('_technical' if technical else '') + '_' + kind + '_v1'


def _task_id(cell, study):
    if study == 'reward037':
        from src.tasks.hammer.config.z1.spline_reward037_cfg import reward_task_id
        return reward_task_id(cell['setting'],height_dr=cell.get('height_dr',False))
    from src.tasks.hammer.config.z1.spline_event_study036_cfg import event_task_id
    return event_task_id(payout=cell['payout'], setting=cell['setting'])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(path):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), 'regular evidence file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record, *, technical=False, study='event036'):
    selected_training = _study_module(study)
    _need(type(technical) is bool, 'explicit technical boolean required')
    _need(type(record) is dict and set(record) == FIELDS, 'closed 036 policy record required')
    schema = _schema(study, 'policy', technical=technical)
    _need(record['schema'] == schema, '036 policy schema required')
    cell = record['training_cell']
    _need(type(cell) is dict, 'training cell required')
    selected = selected_training.cell_for_index(cell.get('index'))
    if technical:
        selected['updates'] = 1
    _need(_canonical(cell) == _canonical(selected), 'selected 036 cell differs')
    label = record['checkpoint_label']
    _need(type(label) is int and label in ((0,) if technical else (99, 499)), 'checkpoint label differs from selected scope')
    _need(type(record['completed_updates']) is int and record['completed_updates'] == label + 1
          and type(record['common_step_counter']) is int and record['common_step_counter'] == (label + 1) * 24,
          'checkpoint update/transition clock differs')
    policy_id = ('technical-' if technical else '') + f"cell-{cell['index']:02d}-model-{label}"
    _need(record['task'] == _task_id(cell, study)
          and record['policy_id'] == policy_id, '036 task/policy identity differs')
    path = Path(record['checkpoint_path'])
    _need(path.is_absolute() and path.name == f'model_{label}.pt', 'absolute selected checkpoint path required')
    for key in ('checkpoint_sha256', 'initial_state_sha256'):
        _need(type(record[key]) is str and re.fullmatch('[0-9a-f]{64}', record[key]), 'actual evidence SHA256 required')
    contract = record['impact_speed_contract']
    _need(type(contract) is dict, 'full initial checkpoint contract required')
    if study == 'reward037':
        from src.tasks.hammer.rl.spline_reward037_contract import reward_identity
        _need(_canonical(contract.get('spline_reward037')) == _canonical(reward_identity(
                  cell['setting'],height_dr=cell.get('height_dr',False))),
              'reward037 all-task-reward identity differs')
        forbidden = ('spline_split_event036', 'spline_exploration_pilot', 'credit_critic_experiment', 'spline_split_credit')
    elif cell['payout'] == 'split':
        from src.tasks.hammer.rl.spline_split_event036_contract import split_event_identity
        _need(_canonical(contract.get('spline_split_event036')) == _canonical(split_event_identity(cell['setting'])),
              'split/event identity differs')
        forbidden = ('spline_exploration_pilot', 'credit_critic_experiment', 'spline_split_credit', 'spline_reward037')
    else:
        from src.tasks.hammer.rl.spline_exploration_pilot_contract import validate_pilot_identity
        from src.tasks.hammer.rl.spline_credit_critic_contract import experiment_identity
        marker = validate_pilot_identity(contract.get('spline_exploration_pilot'))
        experiment = experiment_identity(cell['credit_deadline_ms'], 'event')
        _need(marker['setting'] == cell['setting'] and marker['credit_critic_experiment'] == experiment
              and contract.get('credit_critic_experiment') == experiment, 'single-deadline event identity differs')
        forbidden = ('spline_split_event036', 'spline_split_credit', 'spline_reward037')
    _need(not any(key in contract for key in (*forbidden, 'spline_learning_variant', 'spline_depth_reward031',
                                             'spline_reward_exploration032')), 'mixed study treatment forbidden')
    required = dict(schema_version='impact_speed_v1', observation_widths={'actor':43, 'critic':47},
                    physics_dt_s=.00025, control_decimation=80, speed_bounds_mps=[1.,1.5],
                    speed_mode='policy', action_dim=13,
                    action_type='joint_position_variable_impedance_sampled_pd', controller_identity='sampled_vic500_v1')
    _need(all(_canonical(contract.get(k)) == _canonical(v) for k, v in required.items()),
          '036 action/observation/timing identity differs')
    _canonical(record)
    return record


def record_for_checkpoint(index, checkpoint, initial_state, *, technical=False, study='event036'):
    selected_training = _study_module(study)
    cell = selected_training.cell_for_index(index)
    if technical:
        cell['updates'] = 1
    checkpoint, initial_state = Path(checkpoint).resolve(), Path(initial_state).resolve()
    _need(checkpoint.parent == initial_state.parent/'train',
          'checkpoint must belong to the initial-state run train directory')
    initial_sha = _sha(initial_state)
    initial = json.loads(initial_state.read_text())
    if technical:
        _need(initial.get('schema') == _schema(study, 'initial', technical=True), 'actual technical initialization required')
    _need(initial.get('cell') == cell and initial.get('phase') == 'BEFORE_UPDATE_0_AND_FIRST_ROLLOUT'
          and type(initial.get('actual_counter')) is int and initial['actual_counter'] == 0
          and initial.get('critic_width') == 47
          and initial.get('model_initialization_seed') == initial.get('collection_seed') == cell['seed'],
          'actual 036 fresh initialization evidence differs')
    hashes = initial.get('hashes', {})
    for key in ('actor', 'critic', *selected_training.COMMON_INITIAL_HASHES):
        value = hashes.get(key)
        _need((key == 'collection_cuda_rng' and value is None)
              or (type(value) is str and re.fullmatch('[0-9a-f]{64}', value)), 'full model/normalizer/RNG evidence required')
    observations = initial.get('initial_observations', {})
    _need(observations.get('actor_shape') == [1024,43] and observations.get('critic_shape') == [1024,47],
          'initial observation evidence differs')
    match = re.fullmatch(r'model_(0)\.pt' if technical else r'model_(99|499)\.pt', checkpoint.name)
    _need(match is not None, 'checkpoint filename differs from selected technical/study scope')
    label = int(match.group(1))
    checkpoint_sha = _sha(checkpoint)
    receipt_path = initial_state.parent/('technical-result.json' if technical else f'checkpoint-{label}-live.json')
    _need(receipt_path.is_file() and not receipt_path.is_symlink(), 'same-run checkpoint receipt required')
    receipt = json.loads(receipt_path.read_text())
    _need(type(receipt) is dict
          and type(receipt.get('common_step_counter')) is int
          and receipt['common_step_counter'] == (label+1)*24
          and receipt.get('checkpoint_sha256') == checkpoint_sha, 'checkpoint receipt clock/SHA differs')
    if technical:
        _need(receipt.get('schema') == _schema(study, 'result', technical=True)
              and receipt.get('status') == 'PASS_UPDATE_SAVE_FULL_RELOAD_ONLY'
              and _canonical(receipt.get('cell')) == _canonical(cell)
              and type(receipt.get('checkpoint_label')) is int and receipt['checkpoint_label'] == label
              and type(receipt.get('completed_updates')) is int and receipt['completed_updates'] == 1,
              'technical checkpoint receipt identity/status differs')
    else:
        _need(type(receipt.get('iteration')) is int and receipt['iteration'] == label,
              'checkpoint receipt iteration differs')
    record = dict(schema=_schema(study, 'policy', technical=technical),
                  policy_id=('technical-' if technical else '') + f'cell-{index:02d}-model-{label}',
                  training_cell=cell, checkpoint_label=label, completed_updates=label+1,
                  common_step_counter=(label+1)*24, task=_task_id(cell, study),
                  checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                  initial_state_sha256=initial_sha, impact_speed_contract=initial.get('contract'))
    _need(_sha(initial_state) == initial_sha, 'initial evidence changed while reading')
    return validate_record(record, technical=technical, study=study)


def validate_payload(payload, record, *, technical=False, study='event036'):
    validate_record(record, technical=technical, study=study)
    _need(isinstance(payload, Mapping) and type(payload.get('iter')) is int
          and payload['iter'] == record['checkpoint_label'], 'actual checkpoint label differs')
    infos = payload.get('infos')
    _need(isinstance(infos, Mapping) and isinstance(infos.get('env_state'), Mapping), 'saved runner state required')
    counter = infos['env_state'].get('common_step_counter')
    _need(type(counter) is int and counter == record['common_step_counter'], 'saved transition counter differs')
    _need(infos.get('cat_pressure_treatment') is None, 'CaT pressure scheduler forbidden')
    _need(_canonical(infos.get('impact_speed_contract')) == _canonical(record['impact_speed_contract']),
          'full initial/saved metadata differs')
    validate_state_shapes(payload.get('actor_state_dict'), 43, actor=True)
    validate_state_shapes(payload.get('critic_state_dict'), 47)
    finite_tree(payload)
    raw_std = payload['actor_state_dict']['distribution.std_param'].detach()
    return dict(status='PASS_PAYLOAD_ONLY', iteration=record['checkpoint_label'],
                common_step_counter=counter, actor_width=43, critic_width=47,
                checkpoint_sha256=record['checkpoint_sha256'],
                raw_gaussian_std_parameter=raw_std.cpu().tolist(),
                effective_gaussian_std_after_clamp=raw_std.clamp(1e-6,1e6).cpu().tolist())


def height_evaluation_protocol():
    return dict(schema='z1_reward037_nominal_height_bank_v1',training_offset_range_m=[-.005,.005],
                evaluation_offset_m=0.,body_names=['block','nail'],
                reset_rng='CPU_rng_preserved_around_native_zero_offset_height_event',
                height_robustness_evaluation=False)


@contextmanager
def nominal_height_reset(env):
    """Explicit bank overlay after strict loading; native reset math stays intact."""
    import torch
    from mjlab.envs import mdp
    from src.tasks.hammer.rl.spline_reward037_contract import _validate_training_height
    _need(str(env.device) == 'cpu', 'nominal height bank overlay requires CPU')
    _validate_training_height(env,True)
    cfg = env.event_manager.get_term_cfg('randomize_board_nail_height037')
    original = cfg.func
    _need(original is mdp.dr.body_pos, 'native height reset callback required')
    evidence = dict(height_evaluation_protocol(),reset_calls=0,world_reset_counts={},callback_restored=False)
    @wraps(original)  # Preserve native EventManager recompute/model-field attributes.
    def nominal(source, env_ids, **params):
        _need(source is env and cfg.func is nominal, 'nominal height overlay owner differs')
        _need(params.get('ranges') == (-.005,.005), 'declared training height range changed')
        worlds = range(env.num_envs) if env_ids is None else env_ids.detach().cpu().tolist()
        with torch.random.fork_rng(devices=[]):
            original(source,env_ids,**dict(params,ranges=(0.,0.)))
        evidence['reset_calls'] += 1
        for world in worlds:
            key = str(world)
            evidence['world_reset_counts'][key] = evidence['world_reset_counts'].get(key,0)+1
    cfg.func = nominal
    try:
        yield evidence
    finally:
        intact = cfg.func is nominal
        cfg.func = original
        evidence['callback_restored'] = True
        _need(intact, 'nominal height overlay callback changed')


def make_adapter_class(qualified_base, pilot, *, technical=False, study='event036'):
    class EventCriticHorizonAdapter(pilot.make_adapter_class(qualified_base)):
        def __enter__(self):
            result = super().__enter__()
            try:
                if study == 'reward037' and self.pilot_record['training_cell'].get('height_dr') is True:
                    self.height_evaluation = self.stack.enter_context(nominal_height_reset(self.env))
                return result
            except BaseException:
                self.stack.close()
                raise

        def bind(self, env, runner, saved, contract):
            validate_record(self.pilot_record, technical=technical, study=study)
            _need(_canonical(saved) == _canonical(contract) == _canonical(self.pilot_record['impact_speed_contract']),
                  '036 initial/saved/live contract differs')
            from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models
            validate_impact_speed_models(runner.alg.actor, runner.alg.critic, contract)
            _need(type(env.num_envs) is int and env.num_envs == 32 and str(env.device) == 'cpu', 'CPU32 required')
            _need(type(env.common_step_counter) is int and env.common_step_counter == self.pilot_record['common_step_counter'],
                  'loaded checkpoint transition clock differs')
            # The legacy pilot bind fixes counter2400 and its old marker. Its
            # bank/reset/observer lifecycle is reused; only this binding changes.
            qualified_base.bind(self, env, runner, saved, contract)
    return EventCriticHorizonAdapter


def _load_frontend():
    from scripts.z1_thirty_evaluation023.qualified import verify_copies, pilot_utilities, load_pinned
    pins = verify_copies()
    pilot = pilot_utilities()
    qualified_root = VENDOR / 'pilot-compat001/qualified007'
    sys.path.insert(0, str(qualified_root))
    qualified = load_pinned(qualified_root/'horizon_integration.py',
                            pins['pilot-compat001/qualified007/horizon_integration.py'], 'event036_qualified_horizon')
    evaluator = load_pinned(qualified_root/'current_evaluator_candidate.py',
                            pins['pilot-compat001/qualified007/current_evaluator_candidate.py'], 'event036_qualified_evaluator')
    bank = load_pinned(VENDOR/'reset-bank001/reset_bank.py', BANK_ADAPTER_SHA, 'event036_reset_bank')
    _need(_sha(BANK) == BANK_SHA, 'historical matched32 bank bytes differ')
    pilot.adapt_pilot_condition(evaluator, qualified_root/'current_evaluator_candidate.py')
    return pilot, qualified, evaluator, bank, pins


def _selected_configuration(record, *, study='event036'):
    task, cfg = _study_module(study).build_train_config(record['training_cell']['index'])
    if record['schema'] == _schema(study, 'policy', technical=True):
        cfg.agent.max_iterations = 1
    _need(task == record['task'], 'selected 036 factory task differs')
    called = []
    def selected(task_name):
        _need(task_name == task and not called, 'one selected training factory call required')
        called.append(True)
        return deepcopy(cfg.env), deepcopy(cfg.agent)
    return selected


def cap_technical_steps(evaluator):
    """One opt-in loop-bound seam; preserve all measured equations/callbacks."""
    _need(not hasattr(evaluator, 'technical_step_cap'), 'technical step cap already installed')
    tree = ast.parse(evaluator.thirty_condition_source)
    _need(len(tree.body) == 1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name == '_condition',
          'technical cap requires the adapted condition function alone')
    for name, expected in (('thirty_configuration_adaptation_counts',{'training_factory_config':1}),
                           ('pilot_condition_adaptation_counts',{'balanced_route_expected_signs':1})):
        _need(getattr(evaluator,name,None) == expected, 'qualified configuration/route adaptation missing')
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == 'step'
             and ast.dump(node.iter, include_attributes=False) == ast.dump(ast.parse('range(steps)', mode='eval').body, include_attributes=False)]
    _need(len(loops) == 1, 'exactly one qualified policy-step loop required')
    loops[0].iter = ast.copy_location(ast.parse('range(min(steps, 3))', mode='eval').body, loops[0].iter)
    ast.fix_missing_locations(tree)
    previous = dict(evaluator.__dict__)
    exec(compile(tree, '<event036-technical-step-cap-only>', 'exec'), evaluator.__dict__)
    _need(set(evaluator.__dict__) - set(previous) <= {'_condition','__builtins__'}
          and all(evaluator.__dict__[key] is value for key,value in previous.items() if key != '_condition'),
          'technical cap changed adapted evaluator globals')
    evaluator.technical_step_cap = 3
    evaluator.technical_step_cap_counts = {'policy_step_loop_cap':1}


def measurement_source_files():
    import inspect
    from src.tasks.hammer.mdp.impulse_bound import SubstepImpulseAccumulator
    path = Path(inspect.getfile(SubstepImpulseAccumulator)).resolve()
    return [dict(path=str(path), sha256=_sha(path))]


def write_numeric_audit(qualified, context, output, worker, *, technical=False, study='event036'):
    """Audit actual raw files once; technical binding probes remain nonqualifying."""
    qualified.durable(output/'worker-result.json', worker)
    receipt = dict(schema=_schema(study, 'numeric_audit'), scientific_qualification=False,
                   safety_certified=False, worker_result_sha256=_sha(output/'worker-result.json'))
    if technical:
        receipt.update(status='SKIPPED_TECHNICAL_BINDING_ONLY', study_trials=0)
    else:
        from scripts.z1_thirty_evaluation023.qualified import measurement_reader
        reader = measurement_reader()
        measured = reader.audit_measurements(reader.dependencies(), context, output, json.loads(BANK.read_text()))
        _need(measured.get('denominators', {}).get('all_first_episodes') == 32, 'numeric audit requires all32 starts')
        receipt.update(status='PASS_FILE_NUMERIC_AUDIT_ONLY', measurement=measured)
    qualified.durable(output/'numeric-audit.json', receipt)
    return receipt


def evaluate(index, checkpoint, initial_state, output, *, deadline_epoch, max_output_bytes=128*1024**2, technical=False, study='event036'):
    _study_module(study)
    study_kwargs = {'study':study} if study != 'event036' else {}
    now = time.time()
    _need(type(deadline_epoch) in (int,float) and math.isfinite(deadline_epoch)
          and now < deadline_epoch <= now + 7200, 'finite at-most-two-hour evaluation deadline required')
    _need(type(max_output_bytes) is int and 0 < max_output_bytes <= 128*1024**2, 'bounded128MiB output cap required')
    record = record_for_checkpoint(index, checkpoint, initial_state, **study_kwargs, **({'technical':True} if technical else {}))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    _need(not torch.cuda.is_available(), 'hide CUDA for the preserved CPU evaluation path')
    payload = torch.load(record['checkpoint_path'], map_location='cpu', weights_only=False)
    audit = validate_payload(payload, record, technical=technical, **study_kwargs)
    del payload
    _need(_sha(record['checkpoint_path']) == record['checkpoint_sha256'], 'checkpoint changed during validation')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    pilot, qualified, evaluator, bank, pins = _load_frontend()
    from scripts.z1_thirty_evaluation023.integration import adapt_family_configuration
    adapt_family_configuration(evaluator, VENDOR/'pilot-compat001/qualified007/current_evaluator_candidate.py',
                               _selected_configuration(record, **study_kwargs))
    if technical:
        cap_technical_steps(evaluator)
    context = dict(schema=_schema(study, 'measurement_context'), policy=record,
                   source_root=str(Path(__file__).resolve().parents[1]), source_files=measurement_source_files(),
                   output_root=str(output), evaluation_seed=EVALUATION_SEED,
                   reset_bank=dict(path=str(BANK.resolve()), sha256=BANK_SHA),
                   bank_adapter=dict(path=str((VENDOR/'reset-bank001/reset_bank.py').resolve()), sha256=BANK_ADAPTER_SHA),
                   deadline_epoch=deadline_epoch, max_output_bytes=max_output_bytes,
                   technical=technical, policy_step_cap=3 if technical else None,
                   technical_adaptation_counts=evaluator.technical_step_cap_counts if technical else {},
                   loaded_model_components=['actor'], critic_validation='payload_shape_finite_only',
                   measurement_fields=dict(whole_episode='native-boundary-records.json:native.boundary_measurement',
                       extended_physical_attempt='observer-records.json:whole_attempt',
                       contact_200ms='inclusive-safety-endpoints.json:worlds.*.windows.authoritative_0_200ms_inclusive',
                       literal_200ms_depth='observer-records.json:literal_depth.200ms',
                       native_30mm_attainment='native-boundary-records.json:native.boundary_measurement.depth_30mm_attained',
                       historical_productive='FirstStrikeTracker finalized advancement >0.5mm plus nonnull observer accepted_contact_sample',
                       geometric_strike_reward='separate AcceptedSplineStrikeReward 10mm geometric eligibility; not historical productive label',
                       legacy_report_window_ms=50, authoritative_contact_samples=801,
                       contact_safety_without_contact='UNKNOWN'))
    if record['training_cell'].get('height_dr') is True:
        context['height_evaluation_protocol'] = height_evaluation_protocol()
    qualified.durable(output/'policy.json', record)
    qualified.durable(output/'checkpoint-audit.json', audit)
    qualified.durable(output/'measurement-context.json', dict(context, preserved_source_sha256=pins))
    qualified.durable(output/'runtime.json', dict(python=sys.version, executable=sys.executable, platform=sys.platform,
                      packages={name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','mjlab')},
                      torch_cuda=torch.version.cuda, device='cpu', threads=2))
    case = output/'pilot-bank32'
    arguments = ['--task', record['task'], '--checkpoint', record['checkpoint_path'],
                 '--speed-bounds', '1.0', '1.5', '--speed-cost', '0.8493676', '--seed', str(EVALUATION_SEED),
                 '--num-envs', '32', '--mode', 'policy', '--output', str(case), '--device', 'cpu',
                 '--sampled-pd500', '--sampled-pd500-gain-mode', 'variable', '--measurement-protocol', 'native']
    args = evaluator.parse_args(arguments)
    case.mkdir(exist_ok=False)
    adapter = make_adapter_class(qualified.HorizonAdapter, pilot, technical=technical, **study_kwargs)(context, case, bank)
    args.horizon_adapter = adapter
    _need(time.time() < deadline_epoch, 'evaluation deadline expired before construction')
    rows, report = evaluator._condition(args, None, 0)
    _need(time.time() < deadline_epoch, 'evaluation deadline elapsed before completion')
    _need(adapter.started_counts == {world:1 for world in range(32)} and adapter.final_contract_restored is True,
          'exact first-start denominator/final contract evidence differs')
    _need(_sha(record['checkpoint_path']) == record['checkpoint_sha256'] and _sha(BANK) == BANK_SHA,
          'checkpoint or reset bank changed during evaluation')
    height_evaluation = getattr(adapter,'height_evaluation',None)
    if record['training_cell'].get('height_dr') is True:
        _need(height_evaluation is not None and height_evaluation['callback_restored'] is True
              and height_evaluation['world_reset_counts'] == {str(world):1 for world in range(32)},
              'nominal height overlay must cover each bank world once and restore native callback')
        qualified.durable(output/'height-evaluation.json',height_evaluation)
    for item in [*rows, report]:
        item.update(observer_mapping=adapter.mapping_binding, native_onset_trace=adapter.trace_binding,
                    safety_endpoint_summary=adapter.endpoint_binding, reset_bank=context['reset_bank'])
    qualified.durable(case/'episodes.json', rows)
    qualified.durable(case/'report.json', report)
    _need(qualified.tree_bytes(output) <= max_output_bytes, 'evaluation output cap exceeded')
    result = dict(status='PASS_BINDING_ONLY' if technical else 'RAW_REVIEW_REQUIRED',
                  policy_id=record['policy_id'], attempts=32, study_trials=0 if technical else 32,
                  technical=technical, policy_step_cap=3 if technical else None,
                  checkpoint_sha256=record['checkpoint_sha256'], reset_bank=context['reset_bank'],
                  report_sha256=_sha(case/'report.json'), final_contract_restored=True,
                  loaded_model_components=['actor'], critic_validation='payload_shape_finite_only',
                  measurement_fields=context['measurement_fields'], scientific_qualification=False, safety_certified=False,
                  reports=[dict(case_id='pilot-bank32',report_sha256=_sha(case/'report.json'),
                                observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,
                                safety_endpoint_summary=adapter.endpoint_binding)])
    if height_evaluation is not None:
        result['height_evaluation'] = dict(height_evaluation,path=str(output/'height-evaluation.json'),
                                         sha256=_sha(output/'height-evaluation.json'))
    write_numeric_audit(qualified, context, output, result, technical=technical, **study_kwargs)
    _need(time.time() < deadline_epoch, 'evaluation deadline elapsed during file audit')
    result['numeric_audit'] = dict(path=str(output/'numeric-audit.json'),sha256=_sha(output/'numeric-audit.json'))
    qualified.durable(output/'evaluation-result.json', result)
    _need(qualified.tree_bytes(output) <= max_output_bytes, 'evaluation output cap exceeded including numeric receipt')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('event036','reward037'), default='event036')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--max-output-bytes', type=int, default=128*1024**2)
    parser.add_argument('--technical', action='store_true', help='nonqualifying model0 binding check, capped at three policy steps')
    args = parser.parse_args(argv)
    return evaluate(args.index, args.checkpoint, args.initial_state, args.output,
                    deadline_epoch=args.deadline_epoch, max_output_bytes=args.max_output_bytes,
                    **({'study':args.study} if args.study != 'event036' else {}),
                    **({'technical':True} if args.technical else {}))


if __name__ == '__main__':
    main()

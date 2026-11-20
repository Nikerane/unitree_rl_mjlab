"""Four explicit human-reference windows; no timing or action imitation."""
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math

import torch

from src.tasks.hammer.mdp.ordered_reference import OrderedPolylineReference

REFERENCE_IDS = ('D22', 'U22', 'D20', 'U20')
FIVE_PATHS = ('F04', 'F13', 'F18', 'F20', 'F22')
FIVE_REFERENCE = '+'.join(FIVE_PATHS)
MIXED_REFERENCE_IDS = ('D22+D20', 'U22+U20', FIVE_REFERENCE)
FIVE_DATA_SHA256 = 'eaf6f1a04709f1db5123fbcd7871818eb5b357ffb2072ffaa3246ba7ad83396e'
DATA_PATH = Path(__file__).parents[1] / 'config/z1/data/human_first_strikes.json'


@lru_cache(maxsize=1)
def _data():
    raw = DATA_PATH.read_bytes()
    data = json.loads(raw)
    if data['schema'] != 'z1_human_reference_paths_v1' or data['scale'] != 1.:
        raise ValueError('human reference data schema or scale differs')
    return data, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _five_data():
    raw = DATA_PATH.with_name('five_full_training.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIVE_DATA_SHA256:
        raise ValueError('five-demo training reference file differs')
    bank = json.loads(raw)
    if (bank['schema'] != 'z1_five_complete_references_v1' or bank['scale'] != 1.
            or tuple(bank['paths']) != FIVE_PATHS or bank['training_demo_ids'] != [4, 13, 18, 20, 22]):
        raise ValueError('five-demo training membership or scale differs')
    return bank


def reference_spec(reference_id):
    """Return a declared source window, its raw coordinates and zero-speed start."""
    if reference_id in FIVE_PATHS:
        bank = _five_data()
        row = bank['paths'][reference_id]
        return deepcopy(dict(row, reference_id=reference_id, window='selected_start_to_first_contact',
            data_sha256=FIVE_DATA_SHA256, source_sha256=bank['source_sha256']))
    if type(reference_id) is not str or reference_id not in REFERENCE_IDS:
        raise ValueError('human reference must be D22, U22, D20 or U20')
    data, digest = _data()
    demo = data['demonstrations'][reference_id[1:]]
    begin = demo['apex_index'] if reference_id[0] == 'D' else 0
    points = demo['points_m'][begin:]
    q = demo['apex_q_rad'] if reference_id[0] == 'D' else demo['start_q_rad']
    return deepcopy(dict(reference_id=reference_id, demonstration=int(reference_id[1:]),
        window='apex_to_first_contact' if reference_id[0] == 'D' else 'selected_start_to_first_contact',
        points_m=points, source_frames=demo['source_frames'][begin:], start_q_rad=q,
        apex_index=demo['apex_index']-begin,
        data_sha256=digest, source_sha256=data['source_sha256'], scale=1.))


def path_ids(reference_id):
    """The paths a task uses: one, or two for a combined task."""
    if reference_id in MIXED_REFERENCE_IDS:
        return tuple(reference_id.split('+'))
    reference_spec(reference_id)
    return (reference_id,)


def reference_identity(reference_id):
    if reference_id == FIVE_REFERENCE:
        return dict(schema='z1_five_complete_reference_mixed_v1', reference_id=reference_id,
            paths=[reference_identity(path) for path in FIVE_PATHS],
            world_assignment='world_index_modulo_5_in_path_order',
            training_demo_ids=[4, 13, 18, 20, 22], training_world_counts=[205, 205, 205, 205, 204],
            training_data_sha256=FIVE_DATA_SHA256)
    if reference_id in MIXED_REFERENCE_IDS:
        return dict(schema='z1_human_reference_mixed_v1', reference_id=reference_id,
            paths=[reference_identity(path) for path in path_ids(reference_id)],
            world_assignment='even_worlds_first_path_odd_worlds_second')
    spec = reference_spec(reference_id)
    distances = [math.dist(a,b) for a,b in zip(spec['points_m'][:-1],spec['points_m'][1:])]
    apex_fraction = sum(distances[:spec['apex_index']])/sum(distances)
    result = dict(schema='z1_human_reference_v1', reference_id=reference_id,
        demonstration=spec['demonstration'], window=spec['window'],
        data_sha256=spec['data_sha256'], source_sha256=spec['source_sha256'],
        source_frame_range=[spec['source_frames'][0], spec['source_frames'][-1]],
        scale=1., coordinate='ordered_polyline_arc_length',
        point='Z1 hammer_head_site following registered source head-center path shape',
        reset_joint_positions_rad=spec['start_q_rad'], reset_joint_velocity_rad_s=[0.]*6,
        followthrough_m=.02, path_width_m=.01, lookahead_m=.01,
        phase='distance_including_followthrough', progress='distance_to_first_contact',
        apex_approach_fraction=apex_fraction,
        branch_search='sequential_segments_with_motion_budget_and_crossed_vertex_gate',
        displacement_budget_multiplier=1.25,
        original_timing=False, physical_tracking_qualified=False)
    if reference_id in FIVE_PATHS:
        result.update(start_ik_residual_m=spec['start_ik_residual_m'],
                      max_ik_residual_m=spec['max_ik_residual_m'])
    return result


def _single_reader(reference_id, num_envs, device):
    ref = OrderedPolylineReference(reference_spec(reference_id)['points_m'], num_envs, device)
    ref.reference_id = reference_id
    ref.apex_fraction = reference_identity(reference_id)['apex_approach_fraction']
    return ref


def _per_world(name):
    return property(lambda self: self._pick(*(getattr(reader, name) for reader in self.readers)))


def _per_world_call(name):
    def call(self, *args):
        return self._pick(*(getattr(reader, name)(*args) for reader in self.readers))
    return call


class MixedPolylineReference:
    """Unchanged ordered readers on one batch; each world follows one path.

    All readers see every world and a world only reads its own path's result,
    so single-path reader behaviour is reused rather than reimplemented.
    """
    path_shape = 'human_polyline'

    def __init__(self, reference_id, num_envs, device):
        self.reference_id, self.num_envs, self.device = reference_id, num_envs, torch.device(device)
        self.paths = path_ids(reference_id)
        self.readers = tuple(_single_reader(path, num_envs, device) for path in self.paths)
        self.axis_tol = self.readers[0].axis_tol
        self.path_id = torch.arange(num_envs, device=self.device) % len(self.paths)
        self._forced = None

    def _pick(self, *values):
        worlds = torch.arange(self.num_envs, device=self.device)
        return torch.stack(values, dim=1)[worlds, self.path_id]

    def force_path(self, path):
        """Evaluation only: every world runs one path, as in its single-path task."""
        if path not in self.paths:
            raise ValueError('path is not part of this combined task')
        self.path_id.fill_(self.paths.index(path))
        self._forced = self.readers[self.paths.index(path)]

    def __getattr__(self, name):
        # Per-path constants have no single value until evaluation selects a path.
        forced = self.__dict__.get('_forced')
        if forced is None:
            raise AttributeError(name)
        return getattr(forced, name)

    def reset(self, env_ids=None):
        for reader in self.readers:
            reader.reset(env_ids)

    def diagnose_ordered(self, head_w):
        results = [reader.diagnose_ordered(head_w) for reader in self.readers]
        return tuple(self._pick(*field) for field in zip(*results))

    _nail, _head0, _anchored, _distance = map(_per_world, ('_nail', '_head0', '_anchored', '_distance'))
    update, peek, guidance, approach_coordinate = map(
        _per_world_call, ('update', 'peek', 'guidance', 'approach_coordinate'))
    diagnose_current_height = diagnose_ordered
    approach_height_coordinate = approach_coordinate


def get_human_reference(env, reference_id):
    ref = getattr(env, '_strike_reference', None)
    if ref is None:
        build = MixedPolylineReference if reference_id in MIXED_REFERENCE_IDS else _single_reader
        ref = env._strike_reference = build(reference_id, env.num_envs, env.device)
    if (type(ref) not in (OrderedPolylineReference, MixedPolylineReference) or ref.reference_id != reference_id
            or ref.num_envs != env.num_envs or ref.device != torch.device(env.device)):
        raise ValueError('cached reference differs from selected human task')
    return ref


def reset_human_reference(env, env_ids, *, reference_id):
    get_human_reference(env, reference_id).reset(env_ids)
    clock = getattr(env, '_u22_reference_clock', None)
    if clock is not None:
        clock.reset(env_ids)


def reset_human_joints(env, env_ids, *, reference_id, asset_cfg):
    """Explicit reset at the selected IK start; no hidden connector motion."""
    ref = get_human_reference(env, reference_id)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    starts = torch.tensor([reference_spec(path)['start_q_rad'] for path in path_ids(reference_id)],
                          device=env.device, dtype=torch.float32)
    q = starts[ref.path_id[ids]] if len(starts) > 1 else starts.expand(len(ids), -1)
    env.scene[asset_cfg.name].write_joint_state_to_sim(
        q, torch.zeros_like(q), joint_ids=asset_cfg.joint_ids, env_ids=ids)


def _geometry(env, robot_cfg, nail_cfg, reference_id):
    head = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    nail = env.scene[nail_cfg.name].data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)
    ref = get_human_reference(env, reference_id)
    ref.update(head, nail, env.episode_length_buf)
    return ref, head


def human_phase(env, robot_cfg, nail_cfg, *, reference_id):
    ref, _ = _geometry(env, robot_cfg, nail_cfg, reference_id)
    phase = ref.peek()
    clock = getattr(env, '_u22_reference_clock', None)
    if clock is not None and clock.mode != 'path':
        phase = torch.where(env._hammer_first_strike.started, phase, clock.phase(env, ref))
    return phase.unsqueeze(-1)


def human_error(env, robot_cfg, nail_cfg, *, reference_id):
    ref, head = _geometry(env, robot_cfg, nail_cfg, reference_id)
    waypoint = ref.guidance(head)
    clock = getattr(env, '_u22_reference_clock', None)
    if clock is not None and clock.mode != 'path':
        nominal = ref.waypoint(clock.phase(env, ref) + ref.lookahead_m / ref._total_length)
        use_native = env._hammer_first_strike.started | (env.episode_length_buf == 0)
        waypoint = torch.where(use_native[:, None], waypoint, nominal)
    return waypoint - head


def human_metadata(env):
    """Bind the used path and task ID; shared contracts validate the native recipe."""
    event = env.cfg.events.get('sample_strike_route_signs')
    if event is None or event.func is not reset_human_reference:
        if isinstance(getattr(env,'_strike_reference',None),(OrderedPolylineReference,MixedPolylineReference)):
            raise ValueError('partial human reference wiring cannot claim a legacy task')
        return None
    reference_id = event.params.get('reference_id')
    identity = reference_identity(reference_id)
    if event.params != {'reference_id':reference_id}:
        raise ValueError('human reset reference parameters differ')
    joint = env.cfg.events['reset_strike_route_joints']
    if joint.func is not reset_human_joints or joint.params.get('reference_id') != reference_id:
        raise ValueError('human reset pose differs from reference')
    for group in ('actor', 'critic'):
        for name, func in (('strike_phase',human_phase), ('strike_ref_error',human_error)):
            term = env.observation_manager.get_term_cfg(group,name)
            if (term.func is not func or term.params.get('reference_id') != reference_id
                    or set(term.params) != {'reference_id','robot_cfg','nail_cfg'}):
                raise ValueError('human observation geometry differs')
    ref = get_human_reference(env,reference_id)
    if (type(ref) is MixedPolylineReference and ref._forced is None and not torch.equal(
            ref.path_id,torch.arange(ref.num_envs,device=ref.device)%len(ref.paths))):
        raise ValueError('combined task world assignment differs from its identity')
    for path, reader in zip(path_ids(reference_id), getattr(ref,'readers',(ref,))):
        expected = torch.tensor(reference_spec(path)['points_m'],device=reader.device,
                                dtype=reader.original_points.dtype)
        if not torch.equal(reader.original_points,expected):
            raise ValueError('live human path differs from saved source coordinates')
        if (reader.overshoot != .02 or reader.axis_tol != .01 or reader.lookahead_m != .01
                or reader.motion_budget_multiplier != 1.25):
            raise ValueError('human path reader geometry parameters differ')
    if (env.reward_manager.get_term_cfg('r_imit').weight != 0.
            or env.cfg.rewards['r_imit'].weight != 0. or 'r_imit_anneal' in env.cfg.curriculum):
        raise ValueError('human tasks retain the existing disabled imitation prior')
    clock = getattr(env, '_u22_reference_clock', None)
    if clock is not None:
        if reference_id != 'U22':
            raise ValueError('timed guidance requires U22')
        identity['timing'] = clock.identity.copy()
    return identity


def human_initial_state(env):
    """Bind the actual first reset to the declared path before collection starts."""
    identity = human_metadata(env)
    if identity is None:
        return None
    ref = get_human_reference(env,identity['reference_id'])
    asset = env.observation_manager.get_term_cfg('actor','strike_phase').params['robot_cfg']
    head = env.scene[asset.name].data.site_pos_w[:,asset.site_ids].squeeze(1)
    error = (head-ref._head0).norm(dim=-1)
    if not bool(ref._anchored.all()) or not bool(torch.isfinite(error).all()) or bool((error > ref.axis_tol).any()):
        raise ValueError('human initial head does not match the declared path start within tracking width')
    if identity['reference_id'] == FIVE_REFERENCE:
        # Verify the actual five reset states once before rollout. The scratch
        # forward uses the running environment's MuJoCo model without stepping it.
        import mujoco
        import numpy as np
        controller = env._sampled_pd500
        expected_q = torch.tensor([reference_spec(path)['start_q_rad'] for path in FIVE_PATHS],
                                  dtype=torch.float32, device=env.device)[ref.path_id]
        actual_q = env.sim.data.qpos[:, controller.qids[controller.ids]]
        actual_v = env.sim.data.qvel[:, controller.vids[controller.ids]]
        if (not torch.equal(actual_q, expected_q) or not torch.equal(controller.reset_joint_position, expected_q)
                or not bool(torch.isfinite(actual_v).all()) or torch.count_nonzero(actual_v).item() != 0):
            raise ValueError('five-demo live reset joint positions or velocities differ')
        model = env.sim.mj_model
        scratch = mujoco.MjData(model)
        unique_q = np.unique(env.sim.data.qpos.detach().cpu().numpy(), axis=0)
        if len(unique_q) != len(FIVE_PATHS) or not np.isfinite(unique_q).all():
            raise ValueError('expected five unique finite physical reset states')
        for state in unique_q:
            scratch.qpos[:] = state
            mujoco.mj_forward(model, scratch)
            if scratch.ncon:
                raise ValueError('five-demo initial state has a physical contact')
        return dict(reference_id=FIVE_REFERENCE, training_world_counts=torch.bincount(
            ref.path_id, minlength=len(FIVE_PATHS)).cpu().tolist(), tracking_width_m=ref.axis_tol,
            max_anchor_error_m=float(error.max()), reset_contacts=0, reset_qvel_zero=True,
            reset_offsets_m={path: float(error[ref.path_id == i].max()) for i, path in enumerate(FIVE_PATHS)},
            anchor_error_m=error.detach().cpu().tolist())
    return dict(reference_id=identity['reference_id'],anchor_error_m=error.detach().cpu().tolist(),
                max_anchor_error_m=float(error.max()),tracking_width_m=ref.axis_tol)


def validate_human_identity(metadata):
    value = metadata.get('human_reference')
    if value is not None:
        if type(value) is not dict:
            raise ValueError('human reference checkpoint identity differs')
        expected = reference_identity(value.get('reference_id'))
        if 'timing' in value:
            from src.tasks.hammer.mdp.u22_timing import timing_identity
            if value.get('reference_id') != 'U22' or type(value['timing']) is not dict:
                raise ValueError('timing contract requires U22')
            expected['timing'] = timing_identity(value['timing'].get('mode'))
        if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError('human reference checkpoint identity differs')

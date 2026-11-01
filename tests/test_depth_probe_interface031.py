"""CPU publisher-to-native-tape contract; no physical environment or model."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scripts.z1_depth_campaign031 import probe
from src.tasks.hammer.cat.hook import CatSoftHook
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY, CAT_R_POS_KEY


class HookVectorStep:
    """Fake physical step with the real hook and the wrapper's four-item result."""

    def __init__(self, raw, *, missing=None):
        self.raw = raw
        self.missing = missing
        self.steps = 0
        self.publications = []
        self.hook = CatSoftHook(SimpleNamespace(params=dict(
            robot_cfg=SceneEntityCfg('robot'), limit=1., max_p=.5, tau=.5,
        )), raw)

    def step(self, action):
        assert action.device.type == 'cpu' and action.shape == (2, 13)
        self.steps += 1
        raw = self.raw
        with torch.no_grad():
            raw.scene['robot'].data.joint_vel[:] = torch.tensor(
                [[3. if self.steps % 2 else 2.] * 6, [.5] * 6])
            raw.reward_manager._step_reward[:, 0] = torch.tensor([2., 3.]) * self.steps
            raw.reset_terminated[:] = torch.tensor([self.steps % 2 == 1, False])
            raw.reset_time_outs[:] = torch.tensor([False, self.steps % 2 == 0])
        raw.extras = {}
        self.hook(raw)
        # Keep actual publisher tensors so later mutation detects a reader alias.
        published = raw.extras
        self.publications.append(published)
        returned_extras = dict(published)
        if self.missing is not None:
            del returned_extras[self.missing]
        # Raw extras deliberately become stale: only the step's returned extras
        # own this transition. Reset fields in extras are unrelated decoys too.
        raw.extras = {CAT_DELTA_KEY: torch.full((2,), -11.),
                      CAT_R_POS_KEY: torch.full((2,), -12.)}
        returned_extras['reset_terminated'] = ~raw.reset_terminated
        returned_extras['reset_time_outs'] = ~raw.reset_time_outs
        reward = raw.reward_manager._step_reward.sum(dim=1) * raw.step_dt
        done = (raw.reset_terminated | raw.reset_time_outs).long()
        return raw.obs_buf, reward, done, returned_extras


def raw_fixture():
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(2, 6), joint_vel=torch.zeros(2, 6)))
    raw = SimpleNamespace(device='cpu', num_envs=2, scene={'robot': robot}, step_dt=.02,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2, 6), qvel=torch.zeros(2, 6))),
        obs_buf={'actor': torch.zeros(2, 43)}, extras={},
        action_manager=SimpleNamespace(action=torch.zeros(2, 13)),
        reset_terminated=torch.zeros(2, dtype=torch.bool),
        reset_time_outs=torch.zeros(2, dtype=torch.bool),
        reward_manager=SimpleNamespace(active_terms=['approach', 'action_rate'],
            _step_reward=torch.tensor([[0., -.5], [0., -.25]], requires_grad=True),
            _scale_by_dt=True,
            get_term_cfg=lambda name: SimpleNamespace(weight=-.01 if name == 'action_rate' else 1.)))
    raw._sampled_pd500 = SimpleNamespace(**{key: torch.zeros(2, 6) for key in (
        'applied_kp', 'applied_kd', 'applied_joint_position', 'requested_joint_position',
        'torque', 'requested_torque', 'saturated', 'physics_substeps_since_reset')})
    return raw


@pytest.fixture
def cpu_tape(monkeypatch):
    # Only the tape's action allocation and RNG observation require CUDA seams.
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get('device') == 'cuda:0':
            kwargs['device'] = 'cpu'
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, 'zeros', cpu_zeros)
    monkeypatch.setattr(probe, 'rng_hash', lambda: {'cpu_fixture': 'unchanged'})
    return probe.native_tape


@pytest.mark.parametrize('raw_aliases', [False, True])
def test_tape_reads_real_same_step_hook_publications_and_detaches_snapshots(cpu_tape, raw_aliases):
    raw = raw_fixture()
    assert not hasattr(raw, 'delta') and not hasattr(raw, 'r_pos')
    if raw_aliases:
        raw.delta = torch.full((2,), -21.)
        raw.r_pos = torch.full((2,), -22.)
    vec = HookVectorStep(raw)

    rows = cpu_tape(raw, vec)

    assert len(rows) == vec.steps == 16
    torch.testing.assert_close(rows[0]['delta'], torch.tensor([.5, 0.]))
    torch.testing.assert_close(rows[1]['delta'], torch.tensor([1. / 3., 0.]))
    torch.testing.assert_close(rows[0]['r_pos'], torch.tensor([.04, .06]))
    torch.testing.assert_close(rows[1]['r_pos'], torch.tensor([.08, .12]))
    for index, (row, publication) in enumerate(zip(rows, vec.publications), start=1):
        for field, key in (('delta', CAT_DELTA_KEY), ('r_pos', CAT_R_POS_KEY)):
            torch.testing.assert_close(row[field], publication[key])
            assert row[field].data_ptr() != publication[key].data_ptr()
            assert not row[field].requires_grad and row[field].grad_fn is None
        assert row['reset_terminated'].tolist() == [index % 2 == 1, False]
        assert row['reset_time_outs'].tolist() == [False, index % 2 == 0]
    saved = [{field: row[field].clone() for field in ('delta', 'r_pos')} for row in rows]
    with torch.no_grad():
        for publication in vec.publications:
            publication[CAT_DELTA_KEY].fill_(91.)
            publication[CAT_R_POS_KEY].fill_(92.)
        raw.reset_terminated.fill_(False)
        raw.reset_time_outs.fill_(False)
    for row, prior in zip(rows, saved):
        for field in prior:
            torch.testing.assert_close(row[field], prior[field])
    assert rows[0]['reset_terminated'].tolist() == [True, False]
    assert rows[1]['reset_time_outs'].tolist() == [False, True]


@pytest.mark.parametrize('missing,field', [(CAT_DELTA_KEY, 'delta'), (CAT_R_POS_KEY, 'r_pos')])
def test_tape_rejects_missing_step_publication_even_with_raw_aliases(cpu_tape, missing, field):
    raw = raw_fixture()
    raw.delta = torch.zeros(2)
    raw.r_pos = torch.zeros(2)
    vec = HookVectorStep(raw, missing=missing)

    with pytest.raises(ValueError, match='required native parity field missing: ' + field):
        cpu_tape(raw, vec)
    assert vec.steps == 1

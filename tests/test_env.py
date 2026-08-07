"""Layer 4 — Full environment construction and physics sanity tests.

These tests require the complete mjlab + Warp stack and will compile
GPU/CPU kernels on the first run (slow). Run with:

    pytest tests/test_env.py -v                    # requires Warp
    pytest tests/ -v -m "not integration"          # skip this file

All tests share a single env instance (module scope) to avoid recompiling
Warp kernels between tests.
"""

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.integration


_GUIDELINE_TASK_IDS = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0",
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate",
)

_JOINT_POSITION_PARENT_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
    "CProgress-Vel-Delivered4"
)
_JOINT_POSITION_FIC0_TASK = f"{_JOINT_POSITION_PARENT_TASK}-JointPosition-Fixed"
_JOINT_POSITION_FICTT_TASK = f"{_JOINT_POSITION_FIC0_TASK}-TT"
_JOINT_POSITION_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


@pytest.fixture(scope="module")
def guideline_envs_cpu():
    """Construct both registered Cartesian guideline arms on CPU."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks  # noqa: F401  # populate the task registry in isolated runs

    envs = {}
    try:
        for task_id in _GUIDELINE_TASK_IDS:
            cfg = load_env_cfg(task_id)
            cfg.scene.num_envs = 1
            envs[task_id] = ManagerBasedRlEnv(cfg, device="cpu")
        yield envs
    finally:
        for env in envs.values():
            env.close()


@pytest.fixture(scope="module")
def joint_position_envs_cpu():
    """Construct the joint study arms and their matched Cartesian parent on CPU."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    import src.tasks  # noqa: F401  # populate the task registry in isolated runs

    if _JOINT_POSITION_FIC0_TASK not in list_tasks():
        # Intentional RED: keep each consumer as a normal assertion failure rather
        # than turning the whole module into a fixture-setup KeyError.
        yield {}
        return

    envs = {}
    try:
        task_ids = [
            _JOINT_POSITION_PARENT_TASK,
            _JOINT_POSITION_FIC0_TASK,
        ]
        if _JOINT_POSITION_FICTT_TASK in list_tasks():
            task_ids.append(_JOINT_POSITION_FICTT_TASK)
        for task_id in task_ids:
            cfg = load_env_cfg(task_id, play=True)
            cfg.scene.num_envs = 2
            envs[task_id] = ManagerBasedRlEnv(cfg, device="cpu")
        yield envs
    finally:
        for env in envs.values():
            env.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nail_depth(env) -> float:
    """Return current nail slide qpos for env 0."""
    nail_entity = env.scene["nail_block"]
    joint_ids, _ = nail_entity.find_joints(("nail_slide",))
    return nail_entity.data.joint_pos[0, joint_ids[0]].item()


def _get_site_z(entity, site_name: str) -> float:
    """Return world-frame Z position of a named site for env 0."""
    site_ids, _ = entity.find_sites((site_name,))
    return entity.data.site_pos_w[0, site_ids[0], 2].item()


def _joint_position_env(envs):
    assert _JOINT_POSITION_FIC0_TASK in envs, (
        f"missing registered task {_JOINT_POSITION_FIC0_TASK}"
    )
    return envs[_JOINT_POSITION_FIC0_TASK]


def _joint_position_trackability_env(envs):
    assert _JOINT_POSITION_FICTT_TASK in envs, (
        f"missing registered task {_JOINT_POSITION_FICTT_TASK}"
    )
    return envs[_JOINT_POSITION_FICTT_TASK]


def _apply_joint_action(env, action: torch.Tensor) -> torch.Tensor:
    env.action_manager.process_action(action)
    env.action_manager.apply_action()
    term = env.action_manager.get_term("joint_position")
    robot = env.scene["robot"]
    return robot.data.joint_pos_target[:, term.target_ids].clone()


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


class TestEnvConstruction:
    # (2026-07-14 hygiene: dropped test_env_is_not_none — the fixture failing covers it — and
    # test_num_envs, which asserted the value the fixture itself set: a tautology.)
    def test_action_shape(self, env_cpu):
        """DifferentialIK produces 3D delta-position actions."""
        assert env_cpu.action_manager.action.shape == (1, 3)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class TestObservations:
    # (2026-07-14 hygiene: dropped test_reset_returns_dict — subsumed by the obs_after_reset
    # fixture + the key/shape tests below.)
    def test_obs_has_actor_key(self, obs_after_reset):
        assert "actor" in obs_after_reset

    def test_obs_has_critic_key(self, obs_after_reset):
        assert "critic" in obs_after_reset

    # 33 -> 37 (2026-06-10, T1): + strike_phase (1) + strike_ref_error (3).
    def test_actor_obs_shape(self, obs_after_reset):
        # 7 joint_pos + 7 joint_vel + 3 ee_pos + 3 ee_vel
        # + 3 head_pos + 3 head_vel + 3 nail_top_pos + 1 nail_depth + 3 actions = 33
        assert obs_after_reset["actor"].shape == (1, 37), (
            f"Expected actor shape (1, 37), got {obs_after_reset['actor'].shape}"
        )

    def test_critic_obs_shape(self, obs_after_reset):
        assert obs_after_reset["critic"].shape == (1, 37)

    def test_obs_finite(self, obs_after_reset):
        for key, tensor in obs_after_reset.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in obs['{key}']"
            )


class TestCartesianGuidelineConstruction:
    @pytest.mark.parametrize("task_id", _GUIDELINE_TASK_IDS)
    def test_registered_arm_shapes_and_finite_observations(
        self, guideline_envs_cpu, task_id
    ):
        env = guideline_envs_cpu[task_id]
        obs, _ = env.reset(seed=20260801)
        assert env.action_manager.action.shape == (1, 3)
        # 37 base + next_gate_vector(3) + completed_gate_fraction(1)
        # + guideline_perpendicular_error(1) + waypoint_progress_state(2)
        assert obs["actor"].shape == (1, 44)
        assert obs["critic"].shape == (1, 44)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(obs["critic"]).all()

    def test_same_seed_initial_guideline_observations_match(self, guideline_envs_cpu):
        initial = {}
        for task_id, env in guideline_envs_cpu.items():
            obs, _ = env.reset(seed=20260801)
            initial[task_id] = {
                group: obs[group][:, -5:].clone() for group in ("actor", "critic")
            }

        c0, cgate = _GUIDELINE_TASK_IDS
        torch.testing.assert_close(initial[c0]["actor"], initial[cgate]["actor"])
        torch.testing.assert_close(initial[c0]["critic"], initial[cgate]["critic"])

    def test_training_reset_observation_previews_post_forward_geometry(
        self, guideline_envs_cpu
    ):
        env = guideline_envs_cpu[_GUIDELINE_TASK_IDS[0]]
        obs, _ = env.reset(seed=20260802)
        robot = env.scene["robot"]
        nail = env.scene["nail_block"]
        head_ids, _ = robot.find_sites(("hammer_head_site",))
        nail_ids, _ = nail.find_sites(("nail_top",))
        head = robot.data.site_pos_w[:, head_ids].squeeze(1)
        nail_top = nail.data.site_pos_w[:, nail_ids].squeeze(1)
        # The guideline block is the last SEVEN columns: next_gate_vector(3),
        # completed_gate_fraction(1), guideline_perpendicular_error(1) and the
        # dense waypoint_progress_state(2) = [d_start/reference_length, f_best].
        # Every one of them previews the same post-forward geometry: at reset the
        # head sits on the entry, so its distance to the first target is exactly
        # one seventh of the entry->nail reference and no best fraction is banked.
        first_target = head + (nail_top - head) / 7.0
        reference_length = torch.linalg.vector_norm(nail_top - head, dim=-1, keepdim=True)
        d_start = torch.linalg.vector_norm(first_target - head, dim=-1, keepdim=True)
        expected = torch.cat(
            (
                (nail_top - head) / 7.0,
                torch.zeros((1, 1), device=env.device),
                torch.zeros((1, 1), device=env.device),
                d_start / reference_length,
                torch.zeros((1, 1), device=env.device),
            ),
            dim=-1,
        )
        torch.testing.assert_close(obs["actor"][:, -7:], expected)
        torch.testing.assert_close(obs["critic"][:, -7:], expected)

    @pytest.mark.parametrize("task_id", _GUIDELINE_TASK_IDS)
    def test_actor_and_critic_reads_cannot_advance_tracker(
        self, guideline_envs_cpu, task_id
    ):
        from src.tasks.hammer.mdp.guideline import _ENV_GUIDELINE_ATTR

        env = guideline_envs_cpu[task_id]
        env.reset(seed=20260801)
        tracker = getattr(env, _ENV_GUIDELINE_ATTR)
        state_names = (
            "initialized",
            "entry",
            "nail",
            "previous_head",
            "next_gate",
            "newly_crossed",
            "disarmed",
        )
        before = {name: getattr(tracker, name).clone() for name in state_names}

        for _ in range(3):
            env.observation_manager.compute_group("actor")
            env.observation_manager.compute_group("critic")

        for name, expected in before.items():
            torch.testing.assert_close(getattr(tracker, name), expected)

    def test_runtime_observation_still_fails_closed_without_tracker(
        self, guideline_envs_cpu
    ):
        from src.tasks.hammer.mdp.guideline import _ENV_GUIDELINE_ATTR

        env = guideline_envs_cpu[_GUIDELINE_TASK_IDS[0]]
        env.reset(seed=20260801)
        tracker = getattr(env, _ENV_GUIDELINE_ATTR)
        delattr(env, _ENV_GUIDELINE_ATTR)
        try:
            with pytest.raises(RuntimeError, match="needs WaypointProgressTracker"):
                env.observation_manager.compute_group("actor")
        finally:
            setattr(env, _ENV_GUIDELINE_ATTR, tracker)

    def test_progress_state_tracks_live_tracker_geometry_after_motion(
        self, guideline_envs_cpu
    ):
        """The reset-time assertion above is 1/7 by construction, so it cannot tell a
        real reader from one returning a hardcoded 1/7. Once the head has moved and
        gates have been crossed, target_start_distance is re-anchored to the NEW
        active target and the 1/7 identity no longer holds, so this comparison
        against the tracker's own live state does discriminate.
        """
        from src.tasks.hammer.mdp.guideline import (
            GUIDELINE_NUM_GATES,
            _ENV_GUIDELINE_ATTR,
        )

        env = guideline_envs_cpu[_GUIDELINE_TASK_IDS[0]]
        env.reset(seed=20260803)
        tracker = getattr(env, _ENV_GUIDELINE_ATTR)
        descend = torch.tensor([[0.0, 0.0, -1.0]])
        # Stop while the descent is MID-corridor. After all six gates complete the
        # reader zeroes normalized_start (guideline.py), which would make a
        # hardcoded-constant reader indistinguishable from the real one again.
        for _ in range(3):
            obs, *_ = env.step(descend)

        reference_length = torch.linalg.vector_norm(
            tracker.nail - tracker.entry, dim=-1
        )
        expected = torch.stack(
            (
                (tracker.target_start_distance / reference_length).clamp(0.0, 1.0),
                tracker.best_target_fraction,
            ),
            dim=-1,
        )
        for group in ("actor", "critic"):
            torch.testing.assert_close(obs[group][:, -2:], expected)

        # Non-vacuity: the head has left the entry and the active target has been
        # re-anchored, so the observed value is no longer the reset-time 1/7.
        assert 0 < int(tracker.next_gate[0]) < GUIDELINE_NUM_GATES
        assert not torch.allclose(
            obs["actor"][0, -2],
            torch.tensor(1.0 / (GUIDELINE_NUM_GATES + 1)),
            atol=1e-4,
        )

    def test_gate_crossing_tape_changes_only_one_shot_reward_payout(
        self, guideline_envs_cpu
    ):
        from src.tasks.hammer.mdp.guideline import _ENV_GUIDELINE_ATTR

        state_names = (
            "initialized",
            "entry",
            "nail",
            "previous_head",
            "next_gate",
            "newly_crossed",
            "disarmed",
        )
        action_tape = [torch.tensor([[0.0, 0.0, -1.0]])] * 10
        traces = {}
        for task_id, env in guideline_envs_cpu.items():
            env.reset(seed=20260811)
            tracker = getattr(env, _ENV_GUIDELINE_ATTR)
            trace = []
            for action in action_tape:
                obs, reward, terminated, truncated, extras = env.step(action)
                raw_rewards = {
                    name: env.reward_manager._step_reward[0, index].clone()
                    for index, name in enumerate(env.reward_manager.active_terms)
                }
                trace.append(
                    {
                        "actor": obs["actor"].clone(),
                        "critic": obs["critic"].clone(),
                        "reward": reward.clone(),
                        "terminated": terminated.clone(),
                        "truncated": truncated.clone(),
                        "raw_rewards": raw_rewards,
                        "cat_delta": extras["cat_delta"].clone(),
                        "cat_r_pos": extras["cat_r_pos"].clone(),
                        "tracker": {
                            name: getattr(tracker, name).clone()
                            for name in state_names
                        },
                        "robot_joint_pos": env.scene["robot"].data.joint_pos.clone(),
                        "robot_joint_vel": env.scene["robot"].data.joint_vel.clone(),
                        "robot_joint_target": (
                            env.scene["robot"].data.joint_pos_target.clone()
                        ),
                        "nail_joint_pos": env.scene["nail_block"].data.joint_pos.clone(),
                        "nail_joint_vel": env.scene["nail_block"].data.joint_vel.clone(),
                    }
                )
            traces[task_id] = trace

        c0, cgate = _GUIDELINE_TASK_IDS
        first_episode_gate_rate = 0.0
        for c0_step, cgate_step in zip(traces[c0], traces[cgate], strict=True):
            for name in (
                "actor",
                "critic",
                "terminated",
                "truncated",
                "cat_delta",
                "robot_joint_pos",
                "robot_joint_vel",
                "robot_joint_target",
                "nail_joint_pos",
                "nail_joint_vel",
            ):
                torch.testing.assert_close(
                    c0_step[name], cgate_step[name], rtol=0.0, atol=0.0
                )
            for name in state_names:
                torch.testing.assert_close(
                    c0_step["tracker"][name],
                    cgate_step["tracker"][name],
                    rtol=0.0,
                    atol=0.0,
                )

            c0_raw = c0_step["raw_rewards"]
            cgate_raw = cgate_step["raw_rewards"]
            assert set(cgate_raw) == set(c0_raw) | {"r_gate"}
            for name in c0_raw:
                torch.testing.assert_close(
                    c0_raw[name], cgate_raw[name], rtol=0.0, atol=0.0
                )

            gate_rate = cgate_raw["r_gate"]
            expected_delta = gate_rate * guideline_envs_cpu[cgate].step_dt
            torch.testing.assert_close(
                cgate_step["reward"] - c0_step["reward"],
                expected_delta.unsqueeze(0),
                rtol=0.0,
                atol=2e-7,
            )
            torch.testing.assert_close(
                cgate_step["cat_r_pos"] - c0_step["cat_r_pos"],
                expected_delta.unsqueeze(0),
                rtol=0.0,
                atol=2e-7,
            )
            if not bool(cgate_step["terminated"][0] or cgate_step["truncated"][0]):
                expected_rate = (
                    8.0 * cgate_step["tracker"]["newly_crossed"].float() / 6.0
                )
                torch.testing.assert_close(gate_rate.unsqueeze(0), expected_rate)

            if first_episode_gate_rate < 8.0:
                first_episode_gate_rate += gate_rate.item()
                assert first_episode_gate_rate <= 8.0 + 1e-6

        assert first_episode_gate_rate == pytest.approx(8.0)
        first_done = next(
            index
            for index, step in enumerate(traces[cgate])
            if bool(step["terminated"][0] or step["truncated"][0])
        )
        first_episode_rates = [
            step["raw_rewards"]["r_gate"].item()
            for step in traces[cgate][: first_done + 1]
        ]
        assert sum(first_episode_rates) == pytest.approx(8.0)
        assert sum(first_episode_rates) * guideline_envs_cpu[cgate].step_dt == pytest.approx(
            0.16
        )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    # (2026-07-14 hygiene: dropped test_step_runs — subsumed by test_step_returns_5_tuple.)
    def test_step_returns_5_tuple(self, env_cpu):
        env_cpu.reset()
        action = torch.zeros(1, 3)
        result = env_cpu.step(action)
        assert len(result) == 5

    def test_reward_shape(self, env_cpu):
        env_cpu.reset()
        _, reward, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert reward.shape == (1,), f"Expected reward shape (1,), got {reward.shape}"

    def test_done_is_bool_tensor(self, env_cpu):
        env_cpu.reset()
        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert done.dtype == torch.bool

    def test_reward_is_finite(self, env_cpu):
        env_cpu.reset()
        _, reward, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert torch.isfinite(reward).all()

    def test_obs_after_step_is_finite(self, env_cpu):
        env_cpu.reset()
        obs, _, _, _, _ = env_cpu.step(torch.zeros(1, 3))
        for key, tensor in obs.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in obs['{key}'] after step"
            )


# ---------------------------------------------------------------------------
# Fixed-gain joint-position production action
# ---------------------------------------------------------------------------


class TestJointPositionFixedConstruction:
    def test_fictt_live_manager_exposes_raw_cost_and_external_dt_weighting(
        self, joint_position_envs_cpu
    ):
        """The live TT term is a nonnegative rate; manager sign and dt stay external."""
        env = _joint_position_trackability_env(joint_position_envs_cpu)
        obs, _ = env.reset(seed=20260807)
        assert obs["actor"].shape == (env.num_envs, 47)
        assert obs["critic"].shape == (env.num_envs, 47)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(obs["critic"]).all()
        assert env.action_manager.total_action_dim == 6

        action = torch.full(
            (env.num_envs, 6), 0.5, dtype=torch.float32, device=env.device
        )
        step_obs, _, _, _, _ = env.step(action)
        assert step_obs["actor"].shape == (env.num_envs, 47)
        assert step_obs["critic"].shape == (env.num_envs, 47)
        assert torch.isfinite(step_obs["actor"]).all()
        assert torch.isfinite(step_obs["critic"]).all()

        term = env.reward_manager.get_term_cfg("r_tt")
        raw_cost = term.func(env, **term.params)
        assert torch.isfinite(raw_cost).all()
        assert torch.all(raw_cost >= 0.0)
        assert torch.any(raw_cost > 0.0)
        term_index = env.reward_manager.active_terms.index("r_tt")
        weighted_rate = env.reward_manager._step_reward[:, term_index]
        assert torch.isfinite(weighted_rate).all()
        assert torch.all(weighted_rate <= 0.0)
        torch.testing.assert_close(weighted_rate, raw_cost * term.weight)
        torch.testing.assert_close(
            env.reward_manager._episode_sums["r_tt"],
            weighted_rate * env.step_dt,
        )

        fic0 = _joint_position_env(joint_position_envs_cpu)
        assert "r_tt" not in fic0.reward_manager.active_terms
        with pytest.raises(ValueError, match="r_tt"):
            fic0.reward_manager.get_term_cfg("r_tt")

    def test_joint_and_cartesian_runtime_dimensions_are_isolated(
        self, joint_position_envs_cpu
    ):
        joint = _joint_position_env(joint_position_envs_cpu)
        parent = joint_position_envs_cpu[_JOINT_POSITION_PARENT_TASK]

        joint_obs, _ = joint.reset(seed=20260807)
        joint_step_obs, _, _, _, _ = joint.step(
            torch.zeros((joint.num_envs, 6), device=joint.device)
        )
        for observations in (joint_obs, joint_step_obs):
            assert observations["actor"].shape == (joint.num_envs, 47)
            assert observations["critic"].shape == (joint.num_envs, 47)
            assert torch.isfinite(observations["actor"]).all()
            assert torch.isfinite(observations["critic"]).all()
        assert joint.action_manager.total_action_dim == 6

        parent_obs, _ = parent.reset(seed=20260807)
        assert parent.action_manager.total_action_dim == 3
        assert parent_obs["actor"].shape == (parent.num_envs, 44)
        assert parent_obs["critic"].shape == (parent.num_envs, 44)
        assert torch.isfinite(parent_obs["actor"]).all()
        assert torch.isfinite(parent_obs["critic"]).all()

    def test_live_reference_and_p_observation_widths_stay_four_and_seven(
        self, joint_position_envs_cpu
    ):
        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        expected_widths = {
            "strike_phase": 1,
            "strike_ref_error": 3,
            "next_gate_vector": 3,
            "completed_gate_fraction": 1,
            "guideline_perpendicular_error": 1,
            "waypoint_progress_state": 2,
        }
        for group_name in ("actor", "critic"):
            term_names = env.observation_manager.active_terms[group_name]
            term_dims = env.observation_manager.group_obs_term_dim[group_name]
            widths = {
                name: term_dims[term_names.index(name)][0]
                for name in expected_widths
            }
            assert widths == expected_widths
            assert widths["strike_phase"] + widths["strike_ref_error"] == 4
            assert sum(
                widths[name]
                for name in (
                    "next_gate_vector",
                    "completed_gate_fraction",
                    "guideline_perpendicular_error",
                    "waypoint_progress_state",
                )
            ) == 7

    def test_live_targets_ids_bias_and_exact_affine_map(
        self, joint_position_envs_cpu
    ):
        from src.tasks.hammer.config.z1.joint_position_contract import (
            load_joint_position_contract,
        )

        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        term = env.action_manager.get_term("joint_position")
        robot = env.scene["robot"]
        live_ids, live_names = robot.find_joints(_JOINT_POSITION_NAMES)
        assert tuple(term.target_names) == _JOINT_POSITION_NAMES
        assert tuple(term.target_ids.tolist()) == tuple(live_ids)
        assert tuple(live_names) == _JOINT_POSITION_NAMES
        assert torch.count_nonzero(robot.data.encoder_bias[:, term.target_ids]) == 0

        artifact = (
            Path(__file__).resolve().parents[1]
            / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
        )
        contract = load_joint_position_contract(artifact)
        default = robot.data.default_joint_pos[:, term.target_ids]
        scale = torch.as_tensor(
            contract.scale_rad.tolist(),
            dtype=default.dtype,
            device=default.device,
        ).unsqueeze(0)
        zero = torch.zeros((env.num_envs, 6), device=env.device)
        torch.testing.assert_close(_apply_joint_action(env, zero), default)
        for sign in (-1.0, 1.0):
            raw = torch.full_like(zero, sign)
            torch.testing.assert_close(
                _apply_joint_action(env, raw),
                default + sign * scale,
                rtol=0.0,
                atol=2e-7,
            )

    def test_unwrapped_oversized_actions_hit_only_the_physical_target_clip(
        self, joint_position_envs_cpu
    ):
        from src.tasks.hammer.config.z1.joint_position_contract import (
            load_joint_position_contract,
        )

        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        raw_row = torch.tensor(
            (1e6, -1e6, 1e6, -1e6, 1e6, -1e6),
            dtype=torch.float32,
            device=env.device,
        )
        raw = raw_row.unsqueeze(0).repeat(env.num_envs, 1)
        target = _apply_joint_action(env, raw)
        artifact = (
            Path(__file__).resolve().parents[1]
            / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
        )
        clips = load_joint_position_contract(artifact).physical_clip_rad
        expected_row = torch.as_tensor(
            tuple(
                bounds[1] if action > 0 else bounds[0]
                for action, bounds in zip(raw_row.tolist(), clips, strict=True)
            ),
            dtype=target.dtype,
            device=target.device,
        )
        torch.testing.assert_close(
            env.action_manager.action, raw, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            target,
            expected_row.unsqueeze(0).repeat(env.num_envs, 1),
            rtol=0.0,
            atol=2e-7,
        )

    def test_rsl_wrapper_clips_raw_actions_before_the_physical_action_map(
        self, joint_position_envs_cpu
    ):
        from mjlab.rl import RslRlVecEnvWrapper
        from src.tasks.hammer.config.z1.joint_position_contract import (
            load_joint_position_contract,
        )

        env = _joint_position_env(joint_position_envs_cpu)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        raw_row = torch.tensor(
            (5.0, -5.0, 5.0, -5.0, 5.0, -5.0),
            dtype=torch.float32,
            device=env.device,
        )
        wrapped.step(raw_row.unsqueeze(0).repeat(env.num_envs, 1))
        clipped = raw_row.clamp(-1.0, 1.0).unsqueeze(0).repeat(env.num_envs, 1)
        torch.testing.assert_close(
            env.action_manager.action, clipped, rtol=0.0, atol=0.0
        )

        term = env.action_manager.get_term("joint_position")
        robot = env.scene["robot"]
        default = robot.data.default_joint_pos[:, term.target_ids]
        artifact = (
            Path(__file__).resolve().parents[1]
            / "src/tasks/hammer/config/z1/data/z1_joint_position_stage1.json"
        )
        scale = torch.as_tensor(
            load_joint_position_contract(artifact).scale_rad.tolist(),
            dtype=default.dtype,
            device=default.device,
        ).unsqueeze(0)
        target = robot.data.joint_pos_target[:, term.target_ids]
        torch.testing.assert_close(
            target,
            default + clipped * scale,
            rtol=0.0,
            atol=2e-7,
        )

    def test_desired_target_is_independent_of_current_joint_state(
        self, joint_position_envs_cpu
    ):
        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        term = env.action_manager.get_term("joint_position")
        robot = env.scene["robot"]
        default = robot.data.default_joint_pos[:, term.target_ids]
        action = torch.tensor(
            (0.25, -0.5, 0.75, -0.25, 0.5, -0.75),
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0).repeat(env.num_envs, 1)

        robot.write_joint_position_to_sim(default + 0.02, joint_ids=term.target_ids)
        target_a = _apply_joint_action(env, action)
        robot.write_joint_position_to_sim(default - 0.02, joint_ids=term.target_ids)
        target_b = _apply_joint_action(env, action)
        torch.testing.assert_close(target_a, target_b, rtol=0.0, atol=0.0)
        env.reset(seed=20260807)

    def test_reset_clears_last_action_then_zero_restores_the_default_target(
        self, joint_position_envs_cpu
    ):
        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        env.step(torch.ones((env.num_envs, 6), device=env.device))
        assert torch.count_nonzero(env.action_manager.action) == env.num_envs * 6

        env.reset(seed=20260807)
        observation_term = env.cfg.observations["actor"].terms["actions"]
        last_action = observation_term.func(env, **observation_term.params)
        assert torch.count_nonzero(last_action) == 0
        term = env.action_manager.get_term("joint_position")
        robot = env.scene["robot"]
        default = robot.data.default_joint_pos[:, term.target_ids]
        zero = torch.zeros((env.num_envs, 6), device=env.device)
        torch.testing.assert_close(_apply_joint_action(env, zero), default)

    def test_joint_action_never_changes_the_gripper_target(
        self, joint_position_envs_cpu
    ):
        env = _joint_position_env(joint_position_envs_cpu)
        env.reset(seed=20260807)
        robot = env.scene["robot"]
        gripper_ids, gripper_names = robot.find_joints(("jointGripper",))
        assert tuple(gripper_names) == ("jointGripper",)
        gripper_ids_tensor = torch.tensor(
            gripper_ids, dtype=torch.long, device=env.device
        )
        sentinel = torch.full(
            (env.num_envs, 1), 0.123, dtype=torch.float32, device=env.device
        )
        robot.set_joint_position_target(sentinel, joint_ids=gripper_ids_tensor)
        before = robot.data.joint_pos_target[:, gripper_ids_tensor].clone()
        _apply_joint_action(
            env, torch.ones((env.num_envs, 6), device=env.device)
        )
        torch.testing.assert_close(
            robot.data.joint_pos_target[:, gripper_ids_tensor],
            before,
            rtol=0.0,
            atol=0.0,
        )

    def test_cartesian_checkpoint_shape_fails_loudly_on_joint_runner(
        self, joint_position_envs_cpu, tmp_path
    ):
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

        joint = _joint_position_env(joint_position_envs_cpu)
        parent = joint_position_envs_cpu[_JOINT_POSITION_PARENT_TASK]
        runner_cls = load_runner_cls(_JOINT_POSITION_FIC0_TASK)
        assert runner_cls is not None
        parent_runner = runner_cls(
            RslRlVecEnvWrapper(parent, clip_actions=1.0),
            asdict(load_rl_cfg(_JOINT_POSITION_PARENT_TASK)),
            log_dir=None,
            device="cpu",
        )
        joint_runner = runner_cls(
            RslRlVecEnvWrapper(joint, clip_actions=1.0),
            asdict(load_rl_cfg(_JOINT_POSITION_FIC0_TASK)),
            log_dir=None,
            device="cpu",
        )
        checkpoint = parent_runner.alg.save()
        checkpoint.update(
            {
                "iter": 0,
                "infos": {"env_state": {"common_step_counter": 0}},
            }
        )
        checkpoint_path = tmp_path / "cartesian_44x3.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(RuntimeError, match=r"size mismatch|shape"):
            joint_runner.load(str(checkpoint_path), map_location="cpu")


# ---------------------------------------------------------------------------
# Physics sanity at reset
# ---------------------------------------------------------------------------


class TestPhysicsAtReset:
    def test_nail_depth_zero_at_reset(self, env_cpu):
        env_cpu.reset()
        depth = _get_nail_depth(env_cpu)
        assert abs(depth) < 1e-4, (
            f"Nail depth should be ~0 at reset, got {depth:.6f}"
        )

    def test_hammer_head_above_nail_at_reset(self, env_cpu):
        """Hammer head must start above the nail so it can strike downward."""
        env_cpu.reset()
        robot = env_cpu.scene["robot"]
        nail = env_cpu.scene["nail_block"]

        head_z = _get_site_z(robot, "hammer_head_site")
        nail_top_z = _get_site_z(nail, "nail_top")

        assert head_z > nail_top_z, (
            f"Hammer head z={head_z:.4f} is not above nail top z={nail_top_z:.4f}"
        )

    def test_reset_joint_positions_near_init_pose(self, env_cpu):
        """Joints after reset must sit at NEAR_NAIL_JOINT_POS (the configured
        INIT_STATE) within the ±0.05 rad T0 reset-randomization band.

        (Renamed/fixed 2026-06-10: the old test compared against
        NEUTRAL_JOINT_POS — not the configured init pose — with a 1e-3
        exactness that T0's reset randomization intentionally breaks.)
        """
        from src.assets.robots.unitree_z1.z1_constants import NEAR_NAIL_JOINT_POS

        env_cpu.reset()
        robot = env_cpu.scene["robot"]
        joint_pos = robot.data.joint_pos[0]  # shape: (njoint,)

        rand_band = 0.05 + 1e-3
        for i, name in enumerate(robot.joint_names):
            expected = NEAR_NAIL_JOINT_POS[name]
            actual = joint_pos[i].item()
            assert abs(actual - expected) < rand_band, (
                f"Joint '{name}': expected {expected:.6f} ± {rand_band} rad, "
                f"got {actual:.6f} rad"
            )

    def test_nail_position_in_valid_range(self, env_cpu):
        """Nail qpos must be within its slide joint range [0, NAIL_GOAL_DEPTH]."""
        from src.tasks.hammer.nail_block import NAIL_GOAL_DEPTH

        env_cpu.reset()
        depth = _get_nail_depth(env_cpu)
        assert 0.0 <= depth <= NAIL_GOAL_DEPTH, (
            f"Nail depth {depth:.6f} is outside valid range [0, {NAIL_GOAL_DEPTH}]"
        )


# ---------------------------------------------------------------------------
# Termination logic
# ---------------------------------------------------------------------------


class TestTermination:
    def test_no_termination_at_reset(self, env_cpu):
        """Episode should not be done immediately after reset."""
        env_cpu.reset()
        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        # At rest with nail at 0, done should be False (nail_driven condition)
        # (time_out won't trigger on step 1 of a 20 s episode)
        assert not done[0].item(), (
            "Episode terminated on step 1 — check nail_fully_driven threshold"
        )

    def test_termination_when_nail_fully_driven(self, env_cpu):
        """Manually driving the nail past the success threshold must trigger done."""
        from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

        env_cpu.reset()
        nail_entity = env_cpu.scene["nail_block"]
        joint_ids, _ = nail_entity.find_joints(("nail_slide",))

        # Set nail qpos just above the success threshold
        target_depth = NAIL_SUCCESS_THRESHOLD + 0.001
        pos = torch.tensor(
            [[target_depth]], dtype=torch.float32, device="cpu"
        )
        nail_entity.data.write_joint_position(
            pos, joint_ids=torch.tensor(joint_ids, dtype=torch.long)
        )

        _, _, done, _, _ = env_cpu.step(torch.zeros(1, 3))
        assert done[0].item(), (
            f"Expected done=True with nail at {target_depth:.4f} m "
            f"(threshold={NAIL_SUCCESS_THRESHOLD})"
        )


def test_pv_env_fails_closed_at_BUILD_when_the_substep_tracker_is_missing():
    """The eager resolve in CatSoftHook.__init__ must be load-bearing, not decorative.

    Every other hook test builds via helpers.stub, which bypasses __init__ entirely, so the
    construction-time guard had no coverage: deleting it left the suite green while a
    mis-wired arm would only fail deep into a GPU run. This exercises the real __init__.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks.hammer.config.z1  # noqa: F401

    cfg = load_env_cfg(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel", play=True
    )
    cfg.scene.num_envs = 1
    assert cfg.metrics.pop("substep_peak_qv") is not None  # the tracker the hook requires
    with pytest.raises(RuntimeError, match="SubstepPeakJointVel"):
        ManagerBasedRlEnv(cfg, device="cpu")


def test_pv_env_fails_closed_at_BUILD_on_a_joint_set_mismatch():
    """A same-width but DIFFERENT joint set must be rejected by identity, not by width.

    (A mere permutation is harmless: SceneEntityCfg.resolve normalises names to model order,
    so reversed names resolve to the same ids. The reachable hazard is a different six-joint
    selection -- the Z1 has seven joints, so swapping joint6 for jointGripper still gives six
    columns while shifting what every margin column means.)
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks.hammer.config.z1  # noqa: F401

    cfg = load_env_cfg(
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel", play=True
    )
    cfg.scene.num_envs = 1
    # Still six columns, but the gripper replaces joint6: a pure column-count check waves
    # this through and every margin column is then attributed to the wrong joint.
    cfg.metrics["cat_soft"].params["robot_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "jointGripper"),
    )
    with pytest.raises(RuntimeError, match="joint"):
        ManagerBasedRlEnv(cfg, device="cpu")


def test_wave3_pv_smoke_gate_passes_on_cpu():
    """Rehearse the Vega launch gate locally: every check must pass before it reaches CUDA.

    The smoke script is the last thing standing between a mis-wired arm and six GPU runs, so
    it must itself be exercised. If any check here fails, the gate would have failed on Vega.
    """
    from scripts.smoke_wave3_pv import run_checks

    results = run_checks(device="cpu", num_envs=4, steps=2)
    failed = [name for name, ok, _ in results if not ok]
    assert not failed, failed
    assert len(results) >= 20  # the gate must not silently shrink to a trivial pass


def test_presentation3_pvd0_smoke_gate_passes_on_cpu():
    """The zero reward dose keeps physical instrumentation but has no manager payout."""
    from scripts.smoke_wave3_pv import PVD0_TASK, run_checks

    results = run_checks(task=PVD0_TASK, device="cpu", num_envs=4, steps=2)
    failed = [name for name, ok, _ in results if not ok]
    assert not failed, failed
    assert len(results) >= 20


def test_pvd0_standalone_smoke_anchors_this_checkout_without_pythonpath():
    """The deployment command must not import src from another editable checkout."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    task = (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered0"
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_wave3_pv.py",
            "--task",
            task,
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--steps",
            "1",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[smoke] task   : {task}" in result.stdout
    assert "23/23 checks passed" in result.stdout


def test_pvd0_standalone_smoke_puts_this_checkout_ahead_of_a_decoy(tmp_path):
    """A later worktree entry must be promoted ahead of another checkout."""
    root = Path(__file__).resolve().parents[1]
    decoy_src = tmp_path / "src"
    decoy_src.mkdir()
    (decoy_src / "__init__.py").write_text(
        'raise RuntimeError("decoy src imported")\n', encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(root)))
    task = (
        "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
        "CProgress-Vel-Delivered0"
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_wave3_pv.py",
            "--task",
            task,
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--steps",
            "1",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[smoke] task   : {task}" in result.stdout
    assert "23/23 checks passed" in result.stdout


@pytest.mark.parametrize(
    "task, expected_use_vel",
    (
        (
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
            "CProgress-Delivered4",
            False,
        ),
        (
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
            "CProgress-Vel-Delivered4",
            True,
        ),
        (
            "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-"
            "CProgress-Vel-Delivered0",
            True,
        ),
    ),
)
def test_presentation_phase_one_new_tasks_build_live_and_stay_impulse_log_only(
    task, expected_use_vel
):
    """The two new registered configs must survive real manager construction."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    import src.tasks.hammer.config.z1  # noqa: F401

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        hook_index = list(env.metrics_manager.active_terms).index("cat_soft")
        hook = env.metrics_manager._term_cfgs[hook_index].func
        assert hook._use_vel is expected_use_vel
        assert hook._imp_max_p == 0.0
        expected_delivered = 0.0 if task.endswith("Delivered0") else 4.0
        assert cfg.rewards["delivered_impulse"].weight == expected_delivered
        if expected_delivered:
            assert env.reward_manager.get_term_cfg("delivered_impulse").weight == 4.0
        else:
            # mjlab intentionally skips zero-weight readers at runtime. The configured
            # FirstStrikeDeliveredRewardTerm stays present and physical impulse is measured
            # by first-strike instrumentation, not by a reward-manager payout.
            assert env.reward_manager.get_term_cfg("delivered_impulse").weight == 0.0
        assert set(env.action_manager.active_terms) == {"ik_hammer_head"}
    finally:
        env.close()

"""Integrity gate for the compact direct-reference FIC result package."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation/results/2026-08-12_z1_fic_direct_reference"
RESULT_RECORD = ROOT / "docs/results/2026-08-12_z1_fic_direct_reference.md"
RESULT_INDEX = ROOT / "docs/results/README.md"

EXPECTED_CODE_REVISION = "fe9ff8b8c3debba91219b982a5252b61cd6b67b2"
EXPECTED_ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
FIC0_TASK = (
    "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-"
    "JointPosition-Fixed"
)
FICTT_TASK = f"{FIC0_TASK}-TT"
ARMS = {"fic0": FIC0_TASK, "fictt": FICTT_TASK}
SEEDS = (2, 3, 4)
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
OBSERVATION_TERMS = (
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "ee_vel",
    "head_pos",
    "head_vel",
    "nail_top_pos",
    "nail_depth",
    "strike_phase",
    "strike_ref_error",
    "actions",
)
OBSERVATION_WIDTH = 40
IMITATION_CURRICULUM = (
    (0, 0.10),
    (1200, 0.08),
    (2400, 0.06),
    (3600, 0.04),
    (4800, 0.02),
    (6000, 0.00),
)
EVALUATION_SEED = 2026081202
NUM_ENVS = 64
EPISODE_LENGTH_S = 4.0
MAX_EPISODE_STEPS = 200
I_REF_N_S = 0.2799950838088989
JOINT_VELOCITY_LIMIT_RAD_S = 3.1415
JOINT_IMPULSE_CAP_N_M_S = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64)
EXPECTED_POPULATION_SHA256 = (
    "320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb"
)
EVAL_NAMES = tuple(f"{arm}_seed{seed}.json" for arm in ARMS for seed in SEEDS)
CURRICULUM_NAMES = tuple(
    f"{arm}_seed{seed}_curriculum.json" for arm in ARMS for seed in SEEDS
)
PNG_NAMES = (
    "fic0_seed2_montage.png",
    "fic0_seed2_trajectory.png",
    "fictt_seed2_montage.png",
    "fictt_seed2_trajectory.png",
)
EXPECTED_PNG_PROPERTIES = {
    "fic0_seed2_montage.png": ((5760, 720), "RGB"),
    "fic0_seed2_trajectory.png": ((1600, 768), "RGBA"),
    "fictt_seed2_montage.png": ((5760, 720), "RGB"),
    "fictt_seed2_trajectory.png": ((1600, 768), "RGBA"),
}
EXPECTED_PNG_SHA256 = {
    "fic0_seed2_montage.png": (
        "450b04c974727b2a91146f9c50217fdade96c93a4fc88f89892a8e460f797d18"
    ),
    "fic0_seed2_trajectory.png": (
        "d087d7dab0adcda168044c13ee70669ac64db26fd25e8d8d3ef79093e94520d9"
    ),
    "fictt_seed2_montage.png": (
        "98eedbb4accd216ff1d10689a57e75891f675c9da0ef564ff5a69d84b898cf49"
    ),
    "fictt_seed2_trajectory.png": (
        "90eaccb85df001bc458f5a544562ecb08ef708b86cd548ec36515886b4529ca2"
    ),
}
EXPECTED_IMAGE_EMBEDS = (
    "![FIC-0 seed-2 fixed-reset rollout montage]"
    "(../../evaluation/results/2026-08-12_z1_fic_direct_reference/"
    "fic0_seed2_montage.png)",
    "![FIC-0 seed-2 fixed-reset trajectory]"
    "(../../evaluation/results/2026-08-12_z1_fic_direct_reference/"
    "fic0_seed2_trajectory.png)",
    "![FIC-TT seed-2 fixed-reset rollout montage]"
    "(../../evaluation/results/2026-08-12_z1_fic_direct_reference/"
    "fictt_seed2_montage.png)",
    "![FIC-TT seed-2 fixed-reset trajectory]"
    "(../../evaluation/results/2026-08-12_z1_fic_direct_reference/"
    "fictt_seed2_trajectory.png)",
)
RENDER_ROOT = (
    "/ceph/hpc/home/eunikhilr/campaigns/z1-fic-direct-reference/renders/"
    f"{EXPECTED_CODE_REVISION}/{EXPECTED_ASSET_REVISION}"
)
EXPECTED_MP4_SHA256 = {
    (
        f"{RENDER_ROOT}/fic0_seed2_"
        "92499dac28305f1f406b8f4307114f1cb493290ef8e99771bd766a5f19d1dc0e/"
        "policy.mp4"
    ): "04b9aea4c274dd6e54a179baf03f28dec46ff5bf27591cf2d69db53d2946c617",
    (
        f"{RENDER_ROOT}/fictt_seed2_"
        "98795c91592b9950e37cca6e8f5dfb0a50e20a9291cff89b98c63225681eb931/"
        "policy.mp4"
    ): "a6fd0035193a0cd7719dcd06897f7524945a969b9b92de9e5a43b0a48f7228cb",
}
REQUIRED_CLAIM_BOUNDARIES = (
    "The independent training unit is the training seed (`n=3` per treatment). "
    "The 64 fixed evaluation worlds are paired measurement conditions, not training "
    "replicates.",
    "All comparisons are descriptive; no p-value, confidence interval, or `n=192` "
    "treatment claim is made.",
    "Impulse CaT was log-only, so below-cap observations do not show active constraint "
    "protection or establish hardware safety.",
    "Velocity legality is empirical for these evaluated first episodes, not a hard "
    "guarantee.",
    "Reference-error measurements cover reward-eligible ante-contact samples. The "
    "`0.05 m` bandwidth is an approved permissive reward scale, not a physically "
    "calibrated constant.",
    "The seed-2 renders are qualitative fixed-reset mean-policy examples, not outcome "
    "distributions.",
    "No Cartesian-versus-joint or waypoint-versus-reference causal comparison is made.",
    "The campaign stops before active impulse-CaT, VIC, and further controlled-drop "
    "experiments.",
)
DATA_NAMES = tuple(sorted((*EVAL_NAMES, *CURRICULUM_NAMES, *PNG_NAMES)))
PACKAGE_NAMES = {*DATA_NAMES, "SHA256SUMS"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EPISODE_KEYS = {
    "env_id",
    "success",
    "timeout",
    "episode_steps",
    "first_strike_started",
    "first_strike_finalized",
    "first_strike_productive",
    "first_strike_reason",
    "precontact_velocity_m_s",
    "first_event_impulse_n_s",
    "nail_depth_m",
    "joint_impulse_peak_n_m_s",
    "joint_impulse_utilization",
    "joint_velocity_peak_rad_s",
    "joint_velocity_legal",
    "joint_target_rmse_rad",
    "joint_target_error_max_rad",
    "reference_error_mean_m",
    "reference_error_max_m",
    "reference_error_samples",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise AssertionError(f"non-finite JSON token: {value}")


def _load_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite_json,
    )
    assert isinstance(payload, dict)
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    assert raw == canonical, f"non-canonical JSON bytes: {path.name}"
    return payload


def _finite_float(value: object, label: str) -> float:
    assert type(value) is float, f"{label} must be a JSON float"
    assert math.isfinite(value), f"{label} must be finite"
    return value


def _assert_exact_typed(actual: object, expected: object, label: str) -> None:
    assert type(actual) is type(expected), f"{label} has the wrong JSON type"
    if isinstance(expected, dict):
        assert set(actual) == set(expected), label
        for key, value in expected.items():
            _assert_exact_typed(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        assert len(actual) == len(expected), label
        for index, value in enumerate(expected):
            _assert_exact_typed(actual[index], value, f"{label}[{index}]")
        return
    assert actual == expected, label


def _validate_png(path: Path) -> None:
    assert path.is_file() and not path.is_symlink(), f"missing banked PNG: {path.name}"
    assert path.name in EXPECTED_PNG_PROPERTIES
    expected_size, expected_mode = EXPECTED_PNG_PROPERTIES[path.name]
    try:
        with Image.open(path) as image:
            assert image.format == "PNG", f"wrong image format: {path.name}"
            assert image.size == expected_size, f"wrong PNG dimensions: {path.name}"
            assert image.mode == expected_mode, f"wrong PNG mode: {path.name}"
            image.verify()
        with Image.open(path) as image:
            image.load()
            assert image.size == expected_size, f"wrong decoded dimensions: {path.name}"
            assert image.mode == expected_mode, f"wrong decoded mode: {path.name}"
    except (OSError, SyntaxError) as exc:
        raise AssertionError(f"undecodable PNG: {path.name}") from exc
    assert _sha256(path) == EXPECTED_PNG_SHA256[path.name]


def _expected_treatment_contract(arm: str) -> dict[str, object]:
    task = ARMS[arm]
    return {
        "task": task,
        "action_term": "joint_position",
        "joint_names": list(JOINT_NAMES),
        "observation_names": list(OBSERVATION_TERMS),
        "observation_width": OBSERVATION_WIDTH,
        "actor_obs_normalization": True,
        "critic_obs_normalization": True,
        "algorithm_class": "src.tasks.hammer.rl.cat_ppo:CatPPO",
        "d4_weight": 4.0,
        "delivered_impulse_i_ref_n_s": I_REF_N_S,
        "r_imit_weight": 0.10,
        "r_imit_sigma_m": 0.05,
        "action_rate_weight": -0.01,
        "success_termination": "nail_driven",
        "r_imit_curriculum": [
            {"step": step, "weight": weight}
            for step, weight in IMITATION_CURRICULUM
        ],
        "velocity_cat_substep": True,
        "impulse_cat_log_only": True,
        "row_attribution_enabled": False,
        "endpoint_producers_substep": True,
        "r_tt_enabled": arm == "fictt",
        "r_tt_k_tt": 1.0 if arm == "fictt" else None,
    }


def _validate_episode(row: dict[str, object], env_id: int) -> None:
    assert set(row) == EPISODE_KEYS
    assert type(row["env_id"]) is int and row["env_id"] == env_id
    assert type(row["episode_steps"]) is int
    assert 1 <= row["episode_steps"] <= MAX_EPISODE_STEPS
    assert type(row["reference_error_samples"]) is int
    assert 0 < row["reference_error_samples"] <= row["episode_steps"]
    for key in (
        "success",
        "timeout",
        "first_strike_started",
        "first_strike_finalized",
        "first_strike_productive",
        "joint_velocity_legal",
    ):
        assert type(row[key]) is bool, f"{key} must be bool"
    assert row["success"] or row["timeout"], "row must be a terminal first episode"
    if row["timeout"]:
        assert row["episode_steps"] == MAX_EPISODE_STEPS
    assert row["first_strike_reason"] in {"none", "success", "window"}
    started = row["first_strike_started"]
    finalized = row["first_strike_finalized"]
    productive = row["first_strike_productive"]
    reason = row["first_strike_reason"]
    assert not finalized or started
    assert not productive or finalized
    if finalized:
        assert reason in {"success", "window"}
    else:
        assert reason == "none" and not productive
    if not started:
        assert not finalized and not productive and reason == "none"

    scalar_keys = (
        "precontact_velocity_m_s",
        "first_event_impulse_n_s",
        "nail_depth_m",
        "joint_target_rmse_rad",
        "joint_target_error_max_rad",
        "reference_error_mean_m",
        "reference_error_max_m",
    )
    scalars = {key: _finite_float(row[key], key) for key in scalar_keys}
    assert scalars["first_event_impulse_n_s"] >= 0.0
    assert scalars["nail_depth_m"] >= 0.0
    assert 0.0 <= scalars["joint_target_rmse_rad"] <= scalars[
        "joint_target_error_max_rad"
    ]
    assert 0.0 <= scalars["reference_error_mean_m"] <= scalars[
        "reference_error_max_m"
    ]

    vectors: dict[str, list[float]] = {}
    for key in (
        "joint_impulse_peak_n_m_s",
        "joint_impulse_utilization",
        "joint_velocity_peak_rad_s",
    ):
        value = row[key]
        assert type(value) is list and len(value) == len(JOINT_NAMES)
        vectors[key] = [
            _finite_float(item, f"{key}[{joint}]")
            for joint, item in enumerate(value)
        ]
        assert all(item >= 0.0 for item in vectors[key])
    expected_utilization = [
        peak / cap
        for peak, cap in zip(
            vectors["joint_impulse_peak_n_m_s"],
            JOINT_IMPULSE_CAP_N_M_S,
            strict=True,
        )
    ]
    np.testing.assert_allclose(
        vectors["joint_impulse_utilization"],
        expected_utilization,
        rtol=0.0,
        atol=1e-15,
    )
    assert row["joint_velocity_legal"] is all(
        peak <= JOINT_VELOCITY_LIMIT_RAD_S
        for peak in vectors["joint_velocity_peak_rad_s"]
    )


def _independent_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    impulses = np.asarray([row["first_event_impulse_n_s"] for row in rows], dtype=float)
    impulse_peaks = np.asarray(
        [row["joint_impulse_peak_n_m_s"] for row in rows], dtype=float
    )
    impulse_utilization = impulse_peaks / np.asarray(
        JOINT_IMPULSE_CAP_N_M_S, dtype=float
    )
    qvel_peaks = np.asarray(
        [row["joint_velocity_peak_rad_s"] for row in rows], dtype=float
    )
    target_rmse = np.asarray(
        [row["joint_target_rmse_rad"] for row in rows], dtype=float
    )
    target_max = np.asarray(
        [row["joint_target_error_max_rad"] for row in rows], dtype=float
    )
    reference_mean = np.asarray(
        [row["reference_error_mean_m"] for row in rows], dtype=float
    )
    reference_max = np.asarray(
        [row["reference_error_max_m"] for row in rows], dtype=float
    )
    qvel_legal = np.all(qvel_peaks <= JOINT_VELOCITY_LIMIT_RAD_S, axis=1)
    impulse_legal = np.all(impulse_utilization <= 1.0, axis=1)

    def scalar(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    return {
        "task_success_n": sum(bool(row["success"]) for row in rows),
        "task_success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "productive_first_strike_n": sum(
            bool(row["first_strike_productive"]) for row in rows
        ),
        "productive_first_strike_rate": float(
            np.mean([bool(row["first_strike_productive"]) for row in rows])
        ),
        "first_event_impulse_n_s": {
            "mean": float(np.mean(impulses)),
            "std": float(np.std(impulses)),
            "min": float(np.min(impulses)),
            "max": float(np.max(impulses)),
        },
        "joint_impulse_peak_n_m_s": {
            "p95": np.quantile(impulse_peaks, 0.95, axis=0).tolist(),
            "max": np.max(impulse_peaks, axis=0).tolist(),
        },
        "joint_impulse_utilization": {
            "p95": np.quantile(impulse_utilization, 0.95, axis=0).tolist(),
            "max": np.max(impulse_utilization, axis=0).tolist(),
            "all_joints_at_or_below_cap_n": int(np.sum(impulse_legal)),
            "all_joints_at_or_below_cap_rate": float(np.mean(impulse_legal)),
        },
        "joint_velocity_peak_rad_s": {
            "p95": np.quantile(qvel_peaks, 0.95, axis=0).tolist(),
            "max": np.max(qvel_peaks, axis=0).tolist(),
            "all_joints_legal_n": int(np.sum(qvel_legal)),
            "all_joints_legal_rate": float(np.mean(qvel_legal)),
        },
        "joint_target_rmse_rad": scalar(target_rmse),
        "joint_target_error_max_rad": scalar(target_max),
        "reference_error_mean_m": scalar(reference_mean),
        "reference_error_max_m": scalar(reference_max),
    }


def _assert_nested_close(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), label
        for key, value in expected.items():
            _assert_nested_close(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), label
        for index, value in enumerate(expected):
            _assert_nested_close(actual[index], value, f"{label}[{index}]")
        return
    if type(expected) is int:
        assert type(actual) is int and actual == expected, label
        return
    assert type(expected) is float
    actual_number = _finite_float(actual, label)
    assert math.isclose(
        actual_number, float(expected), rel_tol=1e-12, abs_tol=1e-12
    ), label


def _validate_evaluation(path: Path, arm: str, seed: int) -> dict[str, object]:
    payload = _load_canonical_json(path)
    assert set(payload) == {
        "schema_version",
        "task",
        "training_seed",
        "protocol",
        "checkpoint",
        "code_git",
        "asset_git",
        "treatment_contract",
        "initial_population_sha256",
        "episodes",
        "summary",
    }
    assert type(payload["schema_version"]) is int and payload["schema_version"] == 2
    assert payload["task"] == ARMS[arm]
    assert type(payload["training_seed"]) is int and payload["training_seed"] == seed
    _assert_exact_typed(payload["protocol"], {
        "evaluation_seed": EVALUATION_SEED,
        "num_envs": NUM_ENVS,
        "episodes_per_env": 1,
        "episode_length_s": EPISODE_LENGTH_S,
        "auto_reset": False,
        "policy_mode": "mean",
    }, f"{path.name}.protocol")
    _assert_exact_typed(payload["code_git"], {
        "revision": EXPECTED_CODE_REVISION,
        "status": "",
    }, f"{path.name}.code_git")
    _assert_exact_typed(payload["asset_git"], {
        "revision": EXPECTED_ASSET_REVISION,
        "status": "",
    }, f"{path.name}.asset_git")
    _assert_exact_typed(
        payload["treatment_contract"],
        _expected_treatment_contract(arm),
        f"{path.name}.treatment_contract",
    )
    assert payload["initial_population_sha256"] == EXPECTED_POPULATION_SHA256

    checkpoint = payload["checkpoint"]
    assert isinstance(checkpoint, dict) and set(checkpoint) == {"path", "sha256"}
    assert isinstance(checkpoint["path"], str)
    assert Path(checkpoint["path"]).is_absolute()
    assert Path(checkpoint["path"]).name == "model_499.pt"
    assert (
        f"/campaigns/z1-fic-direct-reference/runs/{EXPECTED_CODE_REVISION}/"
        f"{EXPECTED_ASSET_REVISION}/"
    ) in checkpoint["path"]
    assert f"fic_direct_reference_{arm}_seed{seed}" in checkpoint["path"]
    assert isinstance(checkpoint["sha256"], str)
    assert SHA256_RE.fullmatch(checkpoint["sha256"])

    rows = payload["episodes"]
    assert type(rows) is list and len(rows) == NUM_ENVS
    for env_id, row in enumerate(rows):
        assert type(row) is dict
        _validate_episode(row, env_id)
    _assert_nested_close(
        payload["summary"],
        _independent_summary(rows),
        f"{path.name}.summary",
    )
    return payload


def _validate_curriculum(path: Path, arm: str, seed: int) -> dict[str, object]:
    payload = _load_canonical_json(path)
    assert set(payload) == {
        "schema_version",
        "task",
        "arm",
        "training_seed",
        "scalar",
        "expected_control_step_stages",
        "observed",
        "final_weight",
    }
    assert type(payload["schema_version"]) is int and payload["schema_version"] == 1
    assert payload["task"] == ARMS[arm]
    assert payload["arm"] == arm
    assert type(payload["training_seed"]) is int and payload["training_seed"] == seed
    assert payload["scalar"] == "Curriculum/r_imit_anneal/weight"
    _assert_exact_typed(
        payload["expected_control_step_stages"],
        [list(stage) for stage in IMITATION_CURRICULUM],
        f"{path.name}.expected_control_step_stages",
    )
    assert _finite_float(payload["final_weight"], "final_weight") == 0.0

    observed = payload["observed"]
    assert isinstance(observed, list) and len(observed) == 500
    assert [row.get("iteration") for row in observed] == list(range(500))
    weights: list[float] = []
    for row in observed:
        assert isinstance(row, dict) and set(row) == {"iteration", "weight"}
        assert type(row["iteration"]) is int
        weights.append(_finite_float(row["weight"], "curriculum weight"))
    assert all(-1e-6 <= weight <= 0.1 + 1e-6 for weight in weights)
    assert all(
        weights[index] >= weights[index + 1] - 1e-6
        for index in range(len(weights) - 1)
    )
    assert all(
        any(abs(weight - stage_weight) <= 1e-6 for weight in weights)
        for _, stage_weight in IMITATION_CURRICULUM
    )
    plateau_windows = (
        (0, 49, 0.10),
        (50, 99, 0.08),
        (100, 149, 0.06),
        (150, 199, 0.04),
        (200, 249, 0.02),
        (250, 500, 0.00),
    )
    for start, stop, expected_weight in plateau_windows:
        assert all(
            abs(weight - expected_weight) <= 1e-6
            for weight in weights[start:stop]
        ), f"curriculum plateau drift at iterations {start}:{stop}"
    boundary_transitions = {
        49: (0.10, 0.08),
        99: (0.08, 0.06),
        149: (0.06, 0.04),
        199: (0.04, 0.02),
        249: (0.02, 0.00),
    }
    for iteration, (before, after) in boundary_transitions.items():
        assert after - 1e-6 <= weights[iteration] <= before + 1e-6, (
            f"curriculum boundary drift at iteration {iteration}"
        )
    return payload


def _validate_manifest() -> None:
    lines = (RESULTS / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == len(DATA_NAMES)
    entries: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\n]+)", line)
        assert match is not None, f"invalid SHA256SUMS line: {line!r}"
        entries.append((match.group(1), match.group(2)))
    assert [name for _, name in entries] == list(DATA_NAMES)
    for expected_hash, name in entries:
        assert _sha256(RESULTS / name) == expected_hash, name


def _summary_value(
    summary: dict[str, object], metric: str, statistic: str
) -> float:
    metric_summary = summary[metric]
    assert type(metric_summary) is dict
    return _finite_float(metric_summary[statistic], f"{metric}.{statistic}")


def _primary_result_line(
    summary: dict[str, object], *, arm: str, seed: int
) -> str:
    success_n = summary["task_success_n"]
    productive_n = summary["productive_first_strike_n"]
    assert type(success_n) is int and type(productive_n) is int
    impulse_mean = _summary_value(summary, "first_event_impulse_n_s", "mean")
    impulse_std = _summary_value(summary, "first_event_impulse_n_s", "std")
    target_rmse = _summary_value(summary, "joint_target_rmse_rad", "mean")
    target_max = _summary_value(summary, "joint_target_error_max_rad", "mean")
    reference_mean = _summary_value(summary, "reference_error_mean_m", "mean")
    label = {"fic0": "FIC-0", "fictt": "FIC-TT"}[arm]
    return (
        f"| {seed} | {label} | {success_n}/{NUM_ENVS} | "
        f"{productive_n}/{NUM_ENVS} | {impulse_mean:.6f} ± {impulse_std:.6f} | "
        f"{impulse_mean / I_REF_N_S:.4f} | {target_rmse:.6f} | "
        f"{target_max:.6f} | {reference_mean:.6f} |"
    )


def _paired_delta_values(
    fic0_summary: dict[str, object], fictt_summary: dict[str, object]
) -> np.ndarray:
    fic0_rmse = _summary_value(fic0_summary, "joint_target_rmse_rad", "mean")
    fictt_rmse = _summary_value(fictt_summary, "joint_target_rmse_rad", "mean")
    return np.asarray(
        (
            fictt_rmse - fic0_rmse,
            100.0 * (fictt_rmse - fic0_rmse) / fic0_rmse,
            _summary_value(fictt_summary, "joint_target_error_max_rad", "mean")
            - _summary_value(fic0_summary, "joint_target_error_max_rad", "mean"),
            _summary_value(fictt_summary, "first_event_impulse_n_s", "mean")
            - _summary_value(fic0_summary, "first_event_impulse_n_s", "mean"),
            _summary_value(fictt_summary, "reference_error_mean_m", "mean")
            - _summary_value(fic0_summary, "reference_error_mean_m", "mean"),
            100.0
            * (
                _finite_float(fictt_summary["task_success_rate"], "task success rate")
                - _finite_float(fic0_summary["task_success_rate"], "task success rate")
            ),
            100.0
            * (
                _finite_float(
                    fictt_summary["productive_first_strike_rate"],
                    "productive first-strike rate",
                )
                - _finite_float(
                    fic0_summary["productive_first_strike_rate"],
                    "productive first-strike rate",
                )
            ),
        ),
        dtype=float,
    )


def _paired_delta_line(seed: int, values: np.ndarray) -> str:
    assert values.shape == (7,) and np.isfinite(values).all()
    return (
        f"| {seed} | {values[0]:+.6f} | {values[1]:+.2f}% | "
        f"{values[2]:+.6f} | {values[3]:+.6f} | {values[4]:+.6f} | "
        f"{values[5]:+.2f} | {values[6]:+.2f} |"
    )


def _paired_delta_aggregate_line(deltas: np.ndarray) -> str:
    assert deltas.shape == (len(SEEDS), 7) and np.isfinite(deltas).all()
    means = np.mean(deltas, axis=0)
    sample_std = np.std(deltas, axis=0, ddof=1)
    return (
        "| Mean ± seed SD | "
        f"{means[0]:+.6f} ± {sample_std[0]:.6f} | "
        f"{means[1]:+.2f}% ± {sample_std[1]:.2f}% | "
        f"{means[2]:+.6f} ± {sample_std[2]:.6f} | "
        f"{means[3]:+.6f} ± {sample_std[3]:.6f} | "
        f"{means[4]:+.6f} ± {sample_std[4]:.6f} | "
        f"{means[5]:+.2f} ± {sample_std[5]:.2f} | "
        f"{means[6]:+.2f} ± {sample_std[6]:.2f} |"
    )


def _constraint_result_line(
    summary: dict[str, object], *, arm: str, seed: int
) -> str:
    qvel = summary["joint_velocity_peak_rad_s"]
    utilization = summary["joint_impulse_utilization"]
    assert type(qvel) is dict and type(utilization) is dict
    qvel_max = np.asarray(qvel["max"], dtype=float)
    utilization_max = np.asarray(utilization["max"], dtype=float)
    assert qvel_max.shape == utilization_max.shape == (len(JOINT_NAMES),)
    qvel_joint = int(np.argmax(qvel_max))
    utilization_joint = int(np.argmax(utilization_max))
    label = {"fic0": "FIC-0", "fictt": "FIC-TT"}[arm]
    return (
        f"| {seed} | {label} | {qvel['all_joints_legal_n']}/{NUM_ENVS} | "
        f"{qvel_max[qvel_joint]:.6f}; {JOINT_NAMES[qvel_joint]} | "
        f"{utilization['all_joints_at_or_below_cap_n']}/{NUM_ENVS} | "
        f"{100.0 * utilization_max[utilization_joint]:.2f}%; "
        f"{JOINT_NAMES[utilization_joint]} |"
    )


def _validate_result_record(payloads: list[dict[str, object]]) -> None:
    assert RESULT_RECORD.is_file(), f"missing result record: {RESULT_RECORD}"
    record = RESULT_RECORD.read_text()
    index = RESULT_INDEX.read_text()
    assert (
        "[2026-08-12_z1_fic_direct_reference.md]"
        "(2026-08-12_z1_fic_direct_reference.md)"
    ) in index
    for embed in EXPECTED_IMAGE_EMBEDS:
        assert embed in record
    for payload in payloads:
        checkpoint = payload["checkpoint"]
        checkpoint_lines = [
            line for line in record.splitlines() if checkpoint["path"] in line
        ]
        assert len(checkpoint_lines) == 1
        assert checkpoint["sha256"] in checkpoint_lines[0]
    mp4_lines = [line for line in record.splitlines() if "/policy.mp4" in line]
    assert len(mp4_lines) == len(EXPECTED_MP4_SHA256)
    for path, digest in EXPECTED_MP4_SHA256.items():
        matching_lines = [line for line in mp4_lines if path in line]
        assert len(matching_lines) == 1
        assert digest in matching_lines[0]

    by_treatment: dict[tuple[str, int], dict[str, object]] = {}
    for payload in payloads:
        task = payload["task"]
        arm = next(
            name for name, expected_task in ARMS.items() if task == expected_task
        )
        seed = payload["training_seed"]
        assert type(seed) is int
        key = (arm, seed)
        assert key not in by_treatment
        by_treatment[key] = _independent_summary(payload["episodes"])
    assert set(by_treatment) == {(arm, seed) for arm in ARMS for seed in SEEDS}

    deltas: list[np.ndarray] = []
    for seed in SEEDS:
        for arm in ARMS:
            assert _primary_result_line(
                by_treatment[(arm, seed)], arm=arm, seed=seed
            ) in record
            assert _constraint_result_line(
                by_treatment[(arm, seed)], arm=arm, seed=seed
            ) in record
        delta = _paired_delta_values(
            by_treatment[("fic0", seed)], by_treatment[("fictt", seed)]
        )
        assert _paired_delta_line(seed, delta) in record
        deltas.append(delta)
    assert _paired_delta_aggregate_line(np.stack(deltas)) in record
    normalized_record = " ".join(record.split())
    for statement in REQUIRED_CLAIM_BOUNDARIES:
        assert " ".join(statement.split()) in normalized_record


def test_direct_reference_fic_result_package_is_complete_and_self_consistent() -> None:
    assert RESULTS.is_dir(), f"missing banked result directory: {RESULTS}"
    assert {path.name for path in RESULTS.iterdir()} == PACKAGE_NAMES
    _validate_manifest()
    for name in PNG_NAMES:
        _validate_png(RESULTS / name)

    evaluations: list[dict[str, object]] = []
    populations: set[str] = set()
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for arm in ARMS:
        for seed in SEEDS:
            evaluation = _validate_evaluation(
                RESULTS / f"{arm}_seed{seed}.json", arm, seed
            )
            curriculum = _validate_curriculum(
                RESULTS / f"{arm}_seed{seed}_curriculum.json", arm, seed
            )
            assert curriculum["task"] == evaluation["task"]
            assert curriculum["training_seed"] == evaluation["training_seed"]
            populations.add(evaluation["initial_population_sha256"])
            checkpoint_paths.add(evaluation["checkpoint"]["path"])
            checkpoint_hashes.add(evaluation["checkpoint"]["sha256"])
            evaluations.append(evaluation)

    assert len(populations) == 1, "all six evaluations must share one population"
    assert len(checkpoint_paths) == len(checkpoint_hashes) == 6
    _validate_result_record(evaluations)

"""Contracts for active direct-reference scientific scripts.

These tests deliberately avoid constructing a MuJoCo environment.  They pin the
fail-closed configuration/provenance boundary that must hold before the CPU
measurement scripts are allowed to build one.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from evaluation.whip import lambda_feasibility
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT


REPO = Path(__file__).resolve().parents[1]
REWARD_DESIGN = REPO / "docs" / "research" / "reward-design"
ACTIVE_SCRIPTS = (
    REWARD_DESIGN / "playback_reference.py",
    REWARD_DESIGN / "reward_design_util.py",
    REWARD_DESIGN / "derive_impulse_thresholds.py",
    REWARD_DESIGN / "c2_enforcement_gate.py",
)


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def reward_util() -> ModuleType:
    return _load_script("test_reward_design_util", REWARD_DESIGN / "reward_design_util.py")


@pytest.fixture(scope="module")
def quantity_script() -> ModuleType:
    return _load_script(
        "test_derive_impulse_thresholds",
        REWARD_DESIGN / "derive_impulse_thresholds.py",
    )


@pytest.fixture(scope="module")
def c2_script() -> ModuleType:
    return _load_script("test_c2_enforcement_gate", REWARD_DESIGN / "c2_enforcement_gate.py")


@pytest.fixture(scope="module")
def playback_script() -> ModuleType:
    return _load_script("test_playback_reference", REWARD_DESIGN / "playback_reference.py")


def _cfg(*, imp_max_p: float = 0.0, imp_limit=IMP_J_LIMIT):
    return SimpleNamespace(
        metrics={
            "cat_soft": SimpleNamespace(
                params={"imp_max_p": imp_max_p, "imp_limit": imp_limit}
            )
        }
    )


def test_reference_provenance_is_exactly_the_direct_controller_contract() -> None:
    spec = getattr(lambda_feasibility, "_reference_controller_spec")()

    assert spec == {
        "class": "src.tasks.hammer.mdp.references.SingleStrikeReference",
        "parameters": {
            "overshoot": 0.15,
            "descent_speed": 0.05,
            "axis_tol": 0.05,
        },
        "geometry": {
            "start": "frozen reset hammer-head position",
            "target_xy": "frozen nail x/y",
            "target_z": (
                "min(frozen nail z - overshoot, frozen reset hammer-head z)"
            ),
            "path": "single direct segment with distance-paced linear interpolation",
        },
        "playback": {
            "target": "playback_target(min(control_index, playback_length))",
            "action": "clip((target - live_head) / delta, -1, 1)",
            "recorded_commands": 30,
            "post_zero_commands": 10,
        },
    }
    serialized = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    assert not any(
        retired in serialized
        for retired in (
            "approach_height",
            "min_windup_clearance",
            "windup_speed",
            "descent_margin",
            "apex",
        )
    )
    assert lambda_feasibility._reference_controller_digest() == hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def test_c2_defers_nonzero_before_the_environment_probe(
    c2_script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("environment probe")
        raise AssertionError("C2 must defer before constructing an environment")

    monkeypatch.setattr(c2_script, "build_cfg", forbidden, raising=False)
    monkeypatch.setattr(c2_script, "run_reference_strikes", forbidden, raising=False)

    code = c2_script.main()

    captured = capsys.readouterr()
    assert code != 0
    assert "DEFERRED: direct reference not enforcement-qualified" in captured.err
    assert "imp_max_p must remain 0.0" in captured.err
    assert calls == []


@pytest.mark.parametrize(
    ("cfg", "live_limit", "reason"),
    (
        (_cfg(imp_max_p=0.25), IMP_J_LIMIT, "imp_max_p"),
        (_cfg(imp_limit=[1.64, 3.28, 1.64, 1.64, 1.64, 1.63]), IMP_J_LIMIT, "configured cap"),
        (_cfg(), [1.64, 3.28, 1.64, 1.64, 1.64, 1.63], "live cap"),
    ),
)
def test_direct_reference_guard_rejects_enforcement_or_cap_drift(
    reward_util: ModuleType, cfg, live_limit, reason: str
) -> None:
    guard = getattr(reward_util, "assert_log_only_reference_contract")
    with pytest.raises(RuntimeError, match=reason):
        guard(cfg, live_imp_limit=live_limit)


def test_direct_reference_guard_accepts_only_the_frozen_log_only_contract(
    reward_util: ModuleType,
) -> None:
    guard = getattr(reward_util, "assert_log_only_reference_contract")
    frozen = guard(_cfg(), live_imp_limit=torch.tensor(IMP_J_LIMIT))
    assert torch.equal(frozen, torch.tensor(IMP_J_LIMIT))


def test_direct_reference_guard_rejects_live_imp_max_p_divergence(
    reward_util: ModuleType,
) -> None:
    guard = getattr(reward_util, "assert_log_only_reference_contract")
    with pytest.raises(RuntimeError, match="live imp_max_p"):
        guard(
            _cfg(),
            live_imp_limit=torch.tensor(IMP_J_LIMIT),
            live_imp_max_p=0.25,
        )


def test_live_guard_call_sites_check_both_hook_values() -> None:
    required = {
        "playback_reference.py": "main",
        "reward_design_util.py": "run_reference_strikes",
        "derive_impulse_thresholds.py": "main",
    }

    for filename, function_name in required.items():
        tree = ast.parse((REWARD_DESIGN / filename).read_text(), filename=filename)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        live_guard_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "assert_log_only_reference_contract"
            and "live_imp_limit" in {keyword.arg for keyword in node.keywords}
        ]
        assert len(live_guard_calls) == 1, filename
        assert {
            "live_imp_limit",
            "live_imp_max_p",
        } <= {keyword.arg for keyword in live_guard_calls[0].keywords}


def test_production_measurement_config_is_fixed_reset_fixed_impedance_c0(
    reward_util: ModuleType,
) -> None:
    cfg = getattr(reward_util, "load_direct_reference_c0_cfg")()

    assert cfg.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)
    assert set(cfg.actions) == {"ik_hammer_head"}
    assert "set_gains" not in cfg.actions
    assert cfg.metrics["cat_soft"].params["imp_max_p"] == 0.0
    assert cfg.metrics["cat_soft"].params["imp_limit"] == IMP_J_LIMIT


def test_reference_helper_exposes_repeatability_not_intensity_semantics(
    reward_util: ModuleType,
) -> None:
    run = reward_util.run_reference_strikes
    assert tuple(inspect.signature(run).parameters) == ("cfg", "repeat_count")

    labels = getattr(reward_util, "direct_reference_repeat_indices")(3)
    assert labels == (0, 1, 2)
    with pytest.raises(ValueError, match="positive"):
        getattr(reward_util, "direct_reference_repeat_indices")(0)


def test_reference_provenance_tracks_actual_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    original_digest = lambda_feasibility._reference_controller_digest()
    defaults = SingleStrikeReference.__init__.__defaults__
    assert defaults is not None
    monkeypatch.setattr(
        SingleStrikeReference.__init__,
        "__defaults__",
        (0.123, defaults[1], defaults[2]),
    )

    spec = lambda_feasibility._reference_controller_spec()

    assert spec["parameters"]["overshoot"] == 0.123
    assert lambda_feasibility._reference_controller_digest() != original_digest


def test_direct_reference_provenance_rejects_nonzero_horizontal_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    keyword_defaults = dict(SingleStrikeReference.__init__.__kwdefaults__ or {})
    keyword_defaults["horizontal_detour_m"] = 0.020
    monkeypatch.setattr(
        SingleStrikeReference.__init__, "__kwdefaults__", keyword_defaults
    )
    with pytest.raises(RuntimeError, match="horizontal_detour_m.*0.0"):
        lambda_feasibility._reference_controller_spec()


def test_direct_reference_provenance_rejects_nonhistorical_followthrough_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    keyword_defaults = dict(SingleStrikeReference.__init__.__kwdefaults__ or {})
    keyword_defaults["followthrough_mode"] = "strike_axis"
    monkeypatch.setattr(
        SingleStrikeReference.__init__, "__kwdefaults__", keyword_defaults
    )
    with pytest.raises(RuntimeError, match="followthrough_mode.*vertical"):
        lambda_feasibility._reference_controller_spec()


def test_playback_verdict_explicitly_defers_full_rate_certification(
    playback_script: ModuleType,
) -> None:
    scope = getattr(playback_script, "CONTROL_RATE_FEASIBILITY_SCOPE")
    assert scope == (
        "control-rate sampled feasibility only; authoritative 500 Hz "
        "qvel/contact/finite certification is delegated to "
        "evaluation/guideline/qualify_reference.py"
    )
    main_source = inspect.getsource(playback_script.main)
    assert "CONTROL_RATE_FEASIBILITY_SCOPE" in main_source
    assert "legal finite state" not in main_source


def test_horizontal_playback_requires_three_productive_routes_and_shared_geometry(
    playback_script: ModuleType,
) -> None:
    from src.tasks.hammer.mdp.references import SingleStrikeReference

    head = torch.tensor([[0.50, 0.00, 0.20]])
    nail = torch.tensor([[0.50, 0.00, 0.102]])
    ref = SingleStrikeReference(
        1,
        "cpu",
        overshoot=0.05,
        descent_speed=0.03,
        horizontal_detour_m=0.020,
    )
    ref.update(head, nail, torch.zeros(1, dtype=torch.long))
    tapes = {
        sign: playback_script.route_target_tape(ref, sign)
        for sign in (-1, 0, 1)
    }
    results = {
        sign: {"drove_nail": True, "sampled_feasible": True}
        for sign in (-1, 0, 1)
    }

    assert playback_script.horizontal_route_geometry_matches(tapes)
    assert playback_script.horizontal_route_qualification_passes(results, tapes)
    assert torch.equal(tapes[-1][:, :, 2], tapes[0][:, :, 2])
    assert torch.equal(tapes[0][:, :, 2], tapes[1][:, :, 2])
    assert torch.equal(tapes[-1][0], tapes[0][0])
    assert torch.equal(tapes[0][0], tapes[1][0])
    assert torch.equal(tapes[-1][-1], tapes[0][-1])
    assert torch.equal(tapes[0][-1], tapes[1][-1])

    failed_results = dict(results)
    failed_results[1] = {"drove_nail": False, "sampled_feasible": False}
    assert not playback_script.horizontal_route_qualification_passes(
        failed_results, tapes
    )
    assert not playback_script.horizontal_route_qualification_passes(
        {sign: results[sign] for sign in (-1, 0)}, tapes
    )
    drifted_tapes = {sign: tape.clone() for sign, tape in tapes.items()}
    drifted_tapes[1][1, 0, 2] += 0.001
    assert not playback_script.horizontal_route_geometry_matches(drifted_tapes)


def test_horizontal_playback_orchestrator_runs_each_route_exactly_once(
    playback_script: ModuleType,
) -> None:
    calls: list[int] = []
    tape = torch.tensor(
        [[[0.50, 0.00, 0.20]], [[0.50, 0.00, 0.05]]], dtype=torch.float32
    )

    def run_route(sign: int):
        calls.append(sign)
        return {"drove_nail": True, "sampled_feasible": True}, tape.clone()

    passed, results, tapes = playback_script.qualify_horizontal_routes(run_route)
    assert passed
    assert calls == [-1, 0, 1]
    assert tuple(results) == (-1, 0, 1)
    assert tuple(tapes) == (-1, 0, 1)


def test_measured_contact_duration_cannot_rederive_the_frozen_cap(
    quantity_script: ModuleType,
) -> None:
    summarize = getattr(quantity_script, "direct_reference_measurement_summary")
    measured = torch.tensor(
        [
            [0.10, 0.20, 0.10, 0.10, 0.10, 0.10],
            [0.11, 0.21, 0.11, 0.11, 0.11, 0.11],
        ],
        dtype=torch.float64,
    )

    short = summarize(measured, contact_duration_s=[0.010, 0.012])
    long = summarize(measured, contact_duration_s=[0.100, 0.120])

    assert short["frozen_cap_nms"] == long["frozen_cap_nms"] == IMP_J_LIMIT
    assert short["lambda_to_frozen_cap"] == long["lambda_to_frozen_cap"]
    assert short["sample_kind"] == "deterministic direct-reference repeatability"
    assert "p95" not in short
    assert "intensity" not in short


def test_active_scientific_scripts_have_no_retired_reference_or_probe_controls() -> None:
    allowed_reference_keywords = {
        "overshoot",
        "descent_speed",
        "axis_tol",
        "horizontal_detour_m",
    }
    forbidden_assignments = {
        "APPROACH_HEIGHTS",
        "HEIGHTS",
        "WINDUP_CLEARANCES",
        "MAX_PS",
        "PROBE_SCALE",
    }
    retired_constructor_keywords: list[tuple[str, str]] = []
    retired_controls: list[tuple[str, str]] = []

    for path in ACTIVE_SCRIPTS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "SingleStrikeReference":
                    for keyword in node.keywords:
                        if keyword.arg not in allowed_reference_keywords:
                            retired_constructor_keywords.append((path.name, str(keyword.arg)))
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in forbidden_assignments:
                        retired_controls.append((path.name, target.id))

    assert retired_constructor_keywords == []
    assert retired_controls == []

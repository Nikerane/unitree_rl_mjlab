"""Integrity checks for the compact seed-2 VIC engineering-canary bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "docs/results/assets/2026-08-14_z1_vic_seed2_canary"
RESULT_RECORD = ROOT / "docs/results/2026-08-14_z1_vic_seed2_canary.md"
RESULT_INDEX = ROOT / "docs/results/README.md"

CODE_REVISION = "ca5e83bc84bc979bac857088a9330f2a2bd8034b"
ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
CHECKPOINT_SHA256 = (
    "c1544b779e78e7323bf02ce7b0f165745ee63b5aad9f930f9eea64eb6ea4ea77"
)
EVALUATION_SHA256 = (
    "3203f6db0ebdcf2cd5249d2d57f47b75d97161cba48b7b391394053dfb8915eb"
)
TELEMETRY_SHA256 = (
    "402d5dbe92125facf301cbfcf84c0b36109dd01c6b7978060fcb5a379dd5f62a"
)
KP_TRACE_SHA256 = (
    "34bf03cee93fd295be5d8b5238440b053f4f169bc7b762e1619f72bb7aed0937"
)
POLICY_VIDEO_SHA256 = (
    "cbf61beb73415f3456cfad57dff8fbc93522785b5b2abb00cce02c827f234351"
)

PACKAGE_FILES = {
    "model_499.pt",
    "presentation/metadata.json",
    "presentation/montage.png",
    "presentation/policy.mp4",
    "presentation/trace.npz",
    "presentation/trajectory.png",
    "victt_seed2_eval.json",
    "victt_seed2_kp_through_strike.png",
    "victt_seed2_kp_trace.npz",
    "victt_seed2_kp_trace_eval.json",
    "victt_seed2_telemetry.json",
    "z1_vic_canary_eval.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(relative_path: str) -> dict[str, object]:
    payload = json.loads((PACKAGE / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_vic_canary_bank_has_exact_hash_verified_inventory() -> None:
    manifest = PACKAGE / "SHA256SUMS"
    package_entries = list(PACKAGE.rglob("*"))
    actual_files = {
        path.relative_to(PACKAGE).as_posix()
        for path in package_entries
        if path.is_file()
    }
    assert actual_files == PACKAGE_FILES | {"SHA256SUMS"}
    assert not any(path.is_symlink() for path in package_entries)

    rows = manifest.read_text(encoding="utf-8").splitlines()
    parsed = {}
    for row in rows:
        digest, relative_path = row.split("  ", 1)
        assert len(digest) == 64
        assert relative_path not in parsed
        parsed[relative_path] = digest

    assert list(parsed) == sorted(PACKAGE_FILES)
    assert set(parsed) == PACKAGE_FILES
    for relative_path, digest in parsed.items():
        path = PACKAGE / relative_path
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path) == digest

    assert parsed["model_499.pt"] == CHECKPOINT_SHA256
    assert parsed["victt_seed2_eval.json"] == EVALUATION_SHA256
    assert parsed["victt_seed2_telemetry.json"] == TELEMETRY_SHA256
    assert parsed["victt_seed2_kp_trace.npz"] == KP_TRACE_SHA256
    assert parsed["presentation/policy.mp4"] == POLICY_VIDEO_SHA256


def test_vic_canary_bank_preserves_the_frozen_training_and_behavior_result() -> None:
    result = _load_json("victt_seed2_eval.json")
    assert result["code_git"] == {"revision": CODE_REVISION, "status": ""}
    assert result["asset_git"] == {"revision": ASSET_REVISION, "status": ""}
    assert result["training_seed"] == 2
    assert result["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert result["protocol"] == {
        "auto_reset": False,
        "control_decimation": 10,
        "episode_length_s": 4.0,
        "episodes_per_env": 1,
        "evaluation_seed": 2026081202,
        "max_control_steps": 200,
        "num_envs": 64,
        "physics_dt_s": 0.002,
        "physics_sample_hz": 500,
        "policy_mode": "mean",
    }
    assert result["validation"]["canary_all_gates_pass"] is True

    summary = result["summary"]
    assert summary["task_success_n"] == 64
    assert summary["productive_first_strike_n"] == 64
    assert summary["first_event_impulse_n_s"]["mean"] == 0.4518213882111013
    assert summary["joint_target_rmse_rad"]["mean"] == 0.17455326431080354
    assert max(summary["joint_velocity_peak_rad_s"]["max"]) == 3.105227470397949
    assert max(summary["joint_impulse_utilization"]["max"]) == 0.4946253285175417
    assert result["treatment_contract"]["impulse_cat_log_only"] is True


def test_vic_canary_kp_trace_is_bound_to_the_checkpoint_and_contact_population() -> None:
    with np.load(PACKAGE / "victt_seed2_kp_trace.npz", allow_pickle=False) as trace:
        assert str(trace["checkpoint_sha256"]) == CHECKPOINT_SHA256
        assert str(trace["code_revision"]) == CODE_REVISION
        assert str(trace["asset_revision"]) == ASSET_REVISION
        assert trace["kp_n_m_per_rad"].shape == (64, 80, 6)
        assert trace["sample_seen"].all()
        assert np.array_equal(trace["substep_count"], np.full(64, 80))
        assert set(trace["first_strike_substep"].tolist()) == {69, 70}

        kp = trace["kp_n_m_per_rad"]
        strike = trace["first_strike_substep"]
        at_contact = np.asarray([kp[env_id, index] for env_id, index in enumerate(strike)])
        assert np.array_equal(
            np.median(at_contact, axis=0),
            np.asarray([800.0, 1200.0, 1250.0, 1250.0, 1250.0, 800.0]),
        )


def test_vic_canary_result_is_indexed_with_the_one_seed_claim_boundary() -> None:
    record = RESULT_RECORD.read_text(encoding="utf-8")
    assert "one-seed engineering result" in record
    assert "not evidence that VIC is superior" in record
    assert "impulse CaT was log-only" in record
    assert "64/64" in record
    assert "41119011" in record
    assert "`victt_seed2_eval.json` is the authority" in record
    assert "separate instrumented replay" in record

    index = RESULT_INDEX.read_text(encoding="utf-8")
    assert "2026-08-14_z1_vic_seed2_canary.md" in index

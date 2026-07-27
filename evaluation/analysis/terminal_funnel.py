"""Provenance-bound terminal approach geometry for the first-strike bank.

This module is deliberately offline and NumPy-only.  The schema-v5 bank stores
one physical rollout per episode and counterfactually replays four reward
readers over it; geometry is therefore computed once per episode, never once
per reward reader.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


WINDOW_SAMPLES_BEFORE = 30
EXPECTED_SCHEMA_VERSION = 5
ASSET_MANIFEST_KEY = "assets:hammer_z1_env/assets/nail_block_scene.xml"

NPZ_PAYLOAD_KEY = "payload_json"
# numpy's fixed-width Unicode ("<U") dtype stores itemsize = 4 * char_count in a
# field that must fit a signed 32-bit int, so np.asarray(text, dtype=np.str_)
# raises TypeError("string too large to store inside array") past this many
# characters. A weak seed's un-terminated episodes can serialize well past it.
LEGACY_UNICODE_MAX_CHARS = 536_870_911


def encode_payload_json(text: str) -> np.ndarray:
    """Encode JSON text as a 1-D ``uint8`` array of its UTF-8 bytes.

    Replaces the legacy scalar fixed-width-Unicode (``<U``) encoding, whose
    itemsize ceiling (see ``LEGACY_UNICODE_MAX_CHARS``) a long-episode weak
    seed's payload can exceed. A byte array has no such ceiling.
    """

    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def decode_payload_json(npz) -> str:
    """Decode a loaded NPZ's ``payload_json`` member back to JSON text.

    Accepts both the current 1-D ``uint8`` UTF-8-bytes encoding and the
    legacy scalar ``<U`` encoding it replaces, so artifacts written before
    this change keep loading unchanged. Fails closed (raises ``ValueError``,
    never warns) on any other member set, dtype, shape, or invalid UTF-8.
    """

    files = list(npz.files)
    if files != [NPZ_PAYLOAD_KEY]:
        raise ValueError(
            f"NPZ must contain exactly {[NPZ_PAYLOAD_KEY]!r}, found {files!r}"
        )
    array = npz[NPZ_PAYLOAD_KEY]
    if array.dtype.kind == "U":
        if array.shape != ():
            raise ValueError("legacy payload_json must be a scalar Unicode value")
        return str(array)
    if array.dtype == np.dtype("uint8"):
        if array.ndim != 1:
            raise ValueError("payload_json byte array must be 1-D uint8")
        try:
            return array.tobytes().decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"payload_json is not valid UTF-8: {error}") from error
    raise ValueError(f"payload_json has an unsupported dtype: {array.dtype!r}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def first_contact_window(
    contact: Sequence[bool],
    positions: Sequence[Sequence[float]],
    *,
    samples_before: int = WINDOW_SAMPLES_BEFORE,
) -> tuple[int | None, np.ndarray | None]:
    """Return the inclusive pre-contact window at the first contact onset."""

    contact_array = np.asarray(contact, dtype=bool)
    position_array = np.asarray(positions, dtype=float)
    if contact_array.ndim != 1:
        raise ValueError("contact must be one-dimensional")
    if (
        position_array.ndim != 2
        or position_array.shape[1] != 3
        or position_array.shape[0] != contact_array.size
    ):
        raise ValueError("positions must have shape [len(contact), 3]")
    if samples_before < 1:
        raise ValueError("samples_before must be positive")
    onset_indices = np.flatnonzero(contact_array)
    if not onset_indices.size:
        return None, None
    onset = int(onset_indices[0])
    if onset < samples_before:
        return onset, None
    return onset, position_array[onset - samples_before : onset + 1].copy()


def classify_terminal_funnel(
    transverse_error_m: Sequence[float],
    *,
    tolerance_m: float,
    time_ms: Sequence[float],
) -> dict:
    """Apply the frozen descriptive terminal-funnel classification."""

    error = np.asarray(transverse_error_m, dtype=float)
    time = np.asarray(time_ms, dtype=float)
    if error.ndim != 1 or time.ndim != 1 or error.shape != time.shape:
        raise ValueError("error and time must be same-length one-dimensional arrays")
    if not np.isfinite(error).all() or np.any(error < 0):
        raise ValueError("transverse errors must be finite and non-negative")
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive")
    if not np.all(np.diff(time) > 0) or time[-1] != pytest_approx_zero():
        raise ValueError("time must increase strictly and end at first contact (0 ms)")

    baseline_mask = (time >= -60.0 - 1e-9) & (time <= -20.0 + 1e-9)
    terminal_contraction_mask = (time >= -20.0 - 1e-9) & (time <= -4.0 + 1e-9)
    reexpansion_mask = (time >= -10.0 - 1e-9) & (time <= 1e-9)
    if not (
        baseline_mask.any()
        and terminal_contraction_mask.any()
        and reexpansion_mask.any()
    ):
        raise ValueError("time does not cover the frozen terminal masks")

    baseline = float(np.median(error[baseline_mask]))
    terminal = float(error[-1])
    ratio = terminal / baseline if baseline > 0 else 0.0
    late_median = float(np.median(error[terminal_contraction_mask]))
    late_max = float(np.max(error[reexpansion_mask]))

    if baseline < 2.0 * tolerance_m:
        label = "already_aligned"
    elif terminal > tolerance_m:
        label = "off_target"
    elif ratio > 0.5:
        label = "weak_contraction"
    elif late_median > 0.75 * baseline:
        label = "weak_terminal_contraction"
    elif late_max > 1.25 * tolerance_m:
        label = "late_reexpansion"
    else:
        label = "strong_funnel"

    return {
        "baseline_error_m": baseline,
        "terminal_error_m": terminal,
        "contraction_ratio": ratio,
        "late_median_error_m": late_median,
        "late_max_error_m": late_max,
        "classification": label,
        "strong_funnel": label == "strong_funnel",
        "already_aligned": label == "already_aligned",
    }


def pytest_approx_zero() -> float:
    """Named zero keeps the time validation readable without a pytest import."""

    return 0.0


def compute_episode_metrics(
    position_window_m: Sequence[Sequence[float]],
    *,
    nail_xy: Sequence[float],
    dt_s: float,
    tolerance_m: float,
) -> dict:
    """Compute transverse targeting and final pre-contact kinematics."""

    positions = np.asarray(position_window_m, dtype=float)
    nail = np.asarray(nail_xy, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("position_window_m must have shape [N>=2, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("position_window_m must be finite")
    if nail.shape != (2,) or not np.isfinite(nail).all():
        raise ValueError("nail_xy must contain two finite values")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    offsets = positions[:, :2] - nail
    error = np.linalg.norm(offsets, axis=1)
    time_ms = (np.arange(positions.shape[0]) - positions.shape[0] + 1) * dt_s * 1000.0
    classification = classify_terminal_funnel(
        error, tolerance_m=tolerance_m, time_ms=time_ms
    )
    delta = positions[-1] - positions[-2]
    downward_speed = float(-delta[2] / dt_s)
    lateral_speed = float(np.linalg.norm(delta[:2]) / dt_s)
    approach_angle = (
        float(np.degrees(np.arctan2(lateral_speed, downward_speed)))
        if downward_speed > 0
        else None
    )
    return {
        **classification,
        "time_ms": time_ms.tolist(),
        "transverse_error_m": error.tolist(),
        "terminal_offset_x_m": float(offsets[-1, 0]),
        "terminal_offset_y_m": float(offsets[-1, 1]),
        "downward_speed_m_s": downward_speed,
        "lateral_speed_m_s": lateral_speed,
        "approach_angle_deg": approach_angle,
    }


def _parse_nail_geometry(asset_path: str | Path, expected_sha256: str) -> dict:
    path = Path(asset_path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"nail asset hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    root = ET.parse(path).getroot()
    body = root.find(".//body[@name='nail']")
    head = root.find(".//geom[@name='nail_head']")
    site = root.find(".//site[@name='nail_top']")
    if body is None or head is None or site is None:
        raise ValueError("nail body/head/top site missing from scene XML")
    body_pos = np.fromstring(body.attrib["pos"], sep=" ")
    head_size = np.fromstring(head.attrib["size"], sep=" ")
    site_pos = np.fromstring(site.attrib["pos"], sep=" ")
    if body_pos.shape != (3,) or head_size.size < 1 or site_pos.shape != (3,):
        raise ValueError("unexpected nail geometry vectors")
    return {
        "asset_path": str(path),
        "asset_sha256": actual_sha256,
        "nail_xy_m": body_pos[:2].astype(float).tolist(),
        "nail_top_initial_z_m": float(body_pos[2] + site_pos[2]),
        "tolerance_m": float(head_size[0]),
        "tolerance_basis": "nail_head_radius_center_over_head",
    }


def _validate_physical_trace(trace: Mapping) -> tuple[np.ndarray, np.ndarray]:
    physical = trace["physical"]
    contact = np.asarray(physical["contact"], dtype=bool)
    positions = np.asarray(physical["head_position_m"], dtype=float)
    if contact.ndim != 1 or positions.shape != (contact.size, 3):
        raise ValueError(f"invalid contact/position shape for {trace['episode_id']}")
    if not np.isfinite(positions).all():
        raise ValueError(f"non-finite head positions for {trace['episode_id']}")
    for key in ("clamped_depth_m", "net_axial_force_n"):
        values = np.asarray(physical[key], dtype=float)
        if values.shape != (contact.size,) or not np.isfinite(values).all():
            raise ValueError(f"invalid {key} for {trace['episode_id']}")
    joint_speed = np.asarray(physical["joint_speed_rad_s"], dtype=float)
    if (
        joint_speed.ndim != 2
        or joint_speed.shape[0] != contact.size
        or not np.isfinite(joint_speed).all()
    ):
        raise ValueError(f"invalid joint_speed_rad_s for {trace['episode_id']}")
    return contact, positions


def load_qualified_bank(
    raw_path: str | Path,
    manifest_path: str | Path,
    qualification_summary_path: str | Path,
    asset_path: str | Path,
) -> dict:
    """Load and validate the persisted schema-v5 analysis bank."""

    raw_path = Path(raw_path)
    external_manifest = _load_json(manifest_path)
    qualification = _load_json(qualification_summary_path)
    if not qualification.get("valid"):
        raise ValueError("qualification summary is not valid")
    expected_hashes = qualification["artifact_sha256"]
    if _sha256(raw_path) != expected_hashes["raw"]:
        raise ValueError("raw artifact SHA-256 mismatch")
    if _sha256(manifest_path) != expected_hashes["manifest"]:
        raise ValueError("manifest artifact SHA-256 mismatch")

    with np.load(raw_path, allow_pickle=False) as saved:
        raw = json.loads(decode_payload_json(saved))

    if raw.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("unsupported raw schema version")
    if raw.get("manifest") != external_manifest:
        raise ValueError("embedded and external manifests differ")
    if raw.get("crash") is not None:
        raise ValueError("bank records a crash")

    expected_asset_hash = external_manifest["relevant_provenance"]["file_sha256"][
        ASSET_MANIFEST_KEY
    ]
    geometry = _parse_nail_geometry(asset_path, expected_asset_hash)

    traces = raw["physical_traces"]
    records = raw["reward_records"]
    trace_by_id = {row["episode_id"]: row for row in traces}
    record_by_id = {row["episode_id"]: row for row in records}
    if len(trace_by_id) != len(traces) or len(record_by_id) != len(records):
        raise ValueError("duplicate episode_id in bank")
    if trace_by_id.keys() != record_by_id.keys():
        raise ValueError("physical/reward episode sets differ")
    expected_count = int(external_manifest["expected_episode_count"])
    if len(traces) != expected_count:
        raise ValueError(f"expected {expected_count} episodes, found {len(traces)}")

    checkpoint_hashes = {
        row["path"]: row["sha256"] for row in external_manifest["checkpoints"]
    }
    dt_s = float(external_manifest["physics_dt_s"])
    episodes: list[dict] = []
    shared_reward_trace_count = 0
    contact_count = 0
    eligible_count = 0
    status_counts: dict[str, int] = defaultdict(int)

    for episode_id in sorted(trace_by_id):
        trace = trace_by_id[episode_id]
        record = record_by_id[episode_id]
        contact, positions = _validate_physical_trace(trace)
        checkpoint_path = record["checkpoint_path"]
        if checkpoint_path not in checkpoint_hashes:
            raise ValueError(f"checkpoint not bound in manifest: {checkpoint_path}")
        trace_digests = record["physical_trace_digests"]
        if set(trace_digests) != {"C", "D-prime", "E", "F"}:
            raise ValueError(f"unexpected reward readers for {episode_id}")
        if set(trace_digests.values()) != {trace["trace_digest"]}:
            raise ValueError(f"reward readers do not share physical trace for {episode_id}")
        shared_reward_trace_count += 1

        event_views = [record["returns"][name] for name in ("D-prime", "E", "F")]
        event_metadata = {
            (
                view["finalization_step"],
                view["finalization_reason"],
                bool(view["productive"]),
            )
            for view in event_views
        }
        if len(event_metadata) != 1:
            raise ValueError(f"event readers disagree for {episode_id}")
        finalization_step, finalization_reason, productive = event_metadata.pop()
        if finalization_reason not in {"success", "window"}:
            raise ValueError(f"unexpected finalization reason for {episode_id}")
        status_counts[finalization_reason] += 1

        onset, window = first_contact_window(
            contact, positions, samples_before=WINDOW_SAMPLES_BEFORE
        )
        if onset is not None:
            contact_count += 1
        if window is not None:
            eligible_count += 1
        checkpoint_id = f"{record['stratum']}:seed{int(record['training_seed'])}"
        episodes.append(
            {
                "episode_id": episode_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_hashes[checkpoint_path],
                "stratum": record["stratum"],
                "training_seed": int(record["training_seed"]),
                "split": record["split"],
                "reset_seed": int(record["reset_seed"]),
                "action_seed": int(record["action_seed"]),
                "first_contact_index": onset,
                "position_window_m": window,
                "finalization_step": finalization_step,
                "finalization_reason": finalization_reason,
                "productive": bool(productive),
            }
        )

    return {
        "schema_version": raw["schema_version"],
        "input_paths": {
            "raw": str(raw_path),
            "manifest": str(Path(manifest_path)),
            "qualification_summary": str(Path(qualification_summary_path)),
        },
        "input_sha256": {
            "raw": _sha256(raw_path),
            "manifest": _sha256(manifest_path),
            "qualification_summary": _sha256(qualification_summary_path),
        },
        "manifest": external_manifest,
        "nail_geometry": geometry,
        "validity": {
            "qualification_valid": True,
            "physical_trace_count": len(traces),
            "reward_record_count": len(records),
            "joined_episode_count": len(episodes),
            "shared_reward_trace_count": shared_reward_trace_count,
            "contact_count": contact_count,
            "eligible_window_count": eligible_count,
            "success_count": status_counts["success"],
            "window_count": status_counts["window"],
        },
        "episodes": episodes,
    }


def hierarchical_summary(rows: Sequence[Mapping], *, metric: str) -> dict:
    """Average resets within checkpoints, then checkpoints equally in strata."""

    by_checkpoint: dict[str, list[float]] = defaultdict(list)
    checkpoint_stratum: dict[str, str] = {}
    for row in rows:
        checkpoint = str(row["checkpoint_id"])
        stratum = str(row["stratum"])
        checkpoint_stratum.setdefault(checkpoint, stratum)
        if checkpoint_stratum[checkpoint] != stratum:
            raise ValueError(f"checkpoint {checkpoint} belongs to multiple strata")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric}")
        by_checkpoint[checkpoint].append(value)
    checkpoint_means = {
        checkpoint: float(np.mean(values))
        for checkpoint, values in sorted(by_checkpoint.items())
    }
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for checkpoint, mean in checkpoint_means.items():
        by_stratum[checkpoint_stratum[checkpoint]].append(mean)
    return {
        "checkpoint_means": checkpoint_means,
        "stratum_equal_checkpoint_mean": {
            stratum: float(np.mean(means))
            for stratum, means in sorted(by_stratum.items())
        },
    }


def analyze_bank(bank: Mapping) -> dict:
    """Compute episode, checkpoint, and equal-checkpoint stratum summaries."""

    dt_s = float(bank["manifest"]["physics_dt_s"])
    nail_xy = bank["nail_geometry"]["nail_xy_m"]
    tolerance = float(bank["nail_geometry"]["tolerance_m"])
    episode_metrics: list[dict] = []
    for episode in bank["episodes"]:
        base = {
            key: episode[key]
            for key in (
                "episode_id",
                "checkpoint_id",
                "checkpoint_path",
                "checkpoint_sha256",
                "stratum",
                "training_seed",
                "split",
                "reset_seed",
                "action_seed",
                "first_contact_index",
                "finalization_step",
                "finalization_reason",
                "productive",
            )
        }
        window = episode["position_window_m"]
        if window is None:
            episode_metrics.append(
                {
                    **base,
                    "eligible": False,
                    "classification": (
                        "no_contact"
                        if episode["first_contact_index"] is None
                        else "incomplete_support"
                    ),
                    "strong_funnel": False,
                    "already_aligned": False,
                }
            )
            continue
        episode_metrics.append(
            {
                **base,
                "eligible": True,
                **compute_episode_metrics(
                    window,
                    nail_xy=nail_xy,
                    dt_s=dt_s,
                    tolerance_m=tolerance,
                ),
            }
        )

    eligible = [row for row in episode_metrics if row["eligible"]]
    by_checkpoint: dict[str, list[dict]] = defaultdict(list)
    for row in episode_metrics:
        by_checkpoint[row["checkpoint_id"]].append(row)

    checkpoint_summaries: list[dict] = []
    status_by_checkpoint: list[dict] = []
    for checkpoint_id, rows in sorted(by_checkpoint.items()):
        valid = [row for row in rows if row["eligible"]]
        first = rows[0]
        curves = np.asarray(
            [row["transverse_error_m"] for row in valid], dtype=float
        )
        if valid:
            terminal_error_mean = float(
                np.mean([row["terminal_error_m"] for row in valid])
            )
            terminal_error_median = float(
                np.median([row["terminal_error_m"] for row in valid])
            )
            terminal_offset_mean_x = float(
                np.mean([row["terminal_offset_x_m"] for row in valid])
            )
            terminal_offset_mean_y = float(
                np.mean([row["terminal_offset_y_m"] for row in valid])
            )
            contraction_ratio_mean = float(
                np.mean([row["contraction_ratio"] for row in valid])
            )
            contraction_ratio_median = float(
                np.median([row["contraction_ratio"] for row in valid])
            )
            mean_curve = np.mean(curves, axis=0).tolist()
            q25_curve = np.quantile(curves, 0.25, axis=0).tolist()
            q75_curve = np.quantile(curves, 0.75, axis=0).tolist()
            time_ms = valid[0]["time_ms"]
            at_risk = [int(curves.shape[0])] * curves.shape[1]
        else:
            terminal_error_mean = None
            terminal_error_median = None
            terminal_offset_mean_x = None
            terminal_offset_mean_y = None
            contraction_ratio_mean = None
            contraction_ratio_median = None
            mean_curve = []
            q25_curve = []
            q75_curve = []
            time_ms = (
                (np.arange(WINDOW_SAMPLES_BEFORE + 1) - WINDOW_SAMPLES_BEFORE)
                * dt_s
                * 1000.0
            ).tolist()
            at_risk = [0] * (WINDOW_SAMPLES_BEFORE + 1)
        summary = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": first["checkpoint_path"],
            "checkpoint_sha256": first["checkpoint_sha256"],
            "stratum": first["stratum"],
            "training_seed": first["training_seed"],
            "split": first["split"],
            "episode_count": len(rows),
            "contact_rate": float(
                np.mean([row["first_contact_index"] is not None for row in rows])
            ),
            "eligible_rate": float(np.mean([row["eligible"] for row in rows])),
            "terminal_error_mean_m": terminal_error_mean,
            "terminal_error_median_m": terminal_error_median,
            "terminal_offset_mean_x_m": terminal_offset_mean_x,
            "terminal_offset_mean_y_m": terminal_offset_mean_y,
            "contraction_ratio_mean": contraction_ratio_mean,
            "contraction_ratio_median": contraction_ratio_median,
            "strong_funnel_fraction": float(
                np.mean([row["strong_funnel"] for row in rows])
            ),
            "already_aligned_fraction": float(
                np.mean([row["already_aligned"] for row in rows])
            ),
            "mean_transverse_error_m": mean_curve,
            "q25_transverse_error_m": q25_curve,
            "q75_transverse_error_m": q75_curve,
            "time_ms": time_ms,
            "at_risk_n": at_risk,
        }
        checkpoint_summaries.append(summary)
        status_by_checkpoint.append(
            {
                "checkpoint_id": checkpoint_id,
                "success": sum(
                    row["finalization_reason"] == "success" for row in rows
                ),
                "window": sum(
                    row["finalization_reason"] == "window" for row in rows
                ),
            }
        )

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in checkpoint_summaries:
        by_stratum[row["stratum"]].append(row)
    stratum_summaries = []
    for stratum, rows in sorted(by_stratum.items()):
        terminal_values = [
            row["terminal_error_mean_m"]
            for row in rows
            if row["terminal_error_mean_m"] is not None
        ]
        contraction_values = [
            row["contraction_ratio_mean"]
            for row in rows
            if row["contraction_ratio_mean"] is not None
        ]
        stratum_summaries.append(
            {
                "stratum": stratum,
                "checkpoint_count": len(rows),
                "terminal_error_equal_checkpoint_mean_m": (
                    float(np.mean(terminal_values)) if terminal_values else None
                ),
                "contraction_ratio_equal_checkpoint_mean": (
                    float(np.mean(contraction_values))
                    if contraction_values
                    else None
                ),
                "strong_funnel_equal_checkpoint_fraction": float(
                    np.mean([row["strong_funnel_fraction"] for row in rows])
                ),
            }
        )

    return {
        "episode_count": len(episode_metrics),
        "eligible_episode_count": len(eligible),
        "contact_rate": float(
            np.mean(
                [row["first_contact_index"] is not None for row in episode_metrics]
            )
        ),
        "strong_funnel_episode_fraction": float(
            np.mean([row["strong_funnel"] for row in episode_metrics])
        ),
        "already_aligned_episode_fraction": float(
            np.mean([row["already_aligned"] for row in episode_metrics])
        ),
        "episode_metrics": episode_metrics,
        "checkpoint_summaries": checkpoint_summaries,
        "stratum_summaries": stratum_summaries,
        "status_by_checkpoint": status_by_checkpoint,
        "status_inference_allowed": False,
        "time_window_ms": [-60.0, 0.0],
        "time_points": 31,
        "primary_metric": "transverse nail-axis error",
    }

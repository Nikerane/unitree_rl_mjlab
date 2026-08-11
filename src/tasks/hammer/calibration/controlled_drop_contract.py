"""Immutable result contract for the controlled-drop calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class ControlledDropTrial:
    trial_index: int
    h0_m: float
    release_velocity_m_s: float
    precontact_velocity_m_s: float
    impulse_n_s: float
    contacted: bool
    finalized: bool
    productive: bool
    reason: Literal["success", "window"]
    depth_at_contact_m: float
    peak_depth_m: float


@dataclass(frozen=True)
class ControlledDropSummary:
    trials: tuple[ControlledDropTrial, ...]
    i_ref_mean_n_s: float


@dataclass(frozen=True)
class ControlledDropExecution:
    code_revision: str
    asset_revision: str
    device: str
    backend: str
    mujoco_version: str
    mujoco_warp_version: str
    mjlab_version: str
    physics_dt_s: float
    h0_m: float
    mass_kg: float
    radius_m: float
    half_height_m: float
    friction: tuple[float, float, float]
    slide_axis: tuple[float, float, float]
    slide_damping: float
    slide_frictionloss: float
    tracker_axis: tuple[float, float, float]
    tracker_window_substeps: int
    tracker_progress_eps: float


@dataclass(frozen=True)
class ControlledDropResult:
    execution: ControlledDropExecution
    summary: ControlledDropSummary
    schema_version: int = 1

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution": _json_value(self.execution),
            "summary": _json_value(self.summary),
        }


def summarize_primary_trials(
    trials: Sequence[ControlledDropTrial],
) -> ControlledDropSummary:
    """Preserve the five recorded trials and compute their arithmetic mean."""
    rows = tuple(trials)
    if len(rows) != 5:
        raise ValueError("primary calibration requires exactly five trials")
    for expected_index, row in enumerate(rows, start=1):
        _validate_primary_trial(row, expected_index=expected_index)
    if len({row.reason for row in rows}) != 1:
        raise ValueError("primary trials must have one finalization reason")
    return ControlledDropSummary(
        trials=rows,
        i_ref_mean_n_s=math.fsum(row.impulse_n_s for row in rows) / 5,
    )


def _validate_primary_trial(
    row: ControlledDropTrial,
    *,
    expected_index: int,
) -> None:
    """Validate one primary row without requiring the remaining four rows."""
    if isinstance(row.trial_index, bool):
        raise ValueError("trial indices must use a numeric type, not bool")
    if row.trial_index != expected_index:
        raise ValueError("trial indices must be exactly 1 through 5")
    if (
        isinstance(row.h0_m, bool)
        or not isinstance(row.h0_m, (int, float))
        or not math.isfinite(row.h0_m)
    ):
        raise ValueError("h0 must be a finite numeric value")
    if row.h0_m != 0.150:
        raise ValueError("h0 must be exactly 0.150 m")
    if (
        isinstance(row.release_velocity_m_s, bool)
        or not isinstance(row.release_velocity_m_s, (int, float))
        or not math.isfinite(row.release_velocity_m_s)
        or row.release_velocity_m_s != 0.0
    ):
        raise ValueError("trial must be released from rest")
    if (
        isinstance(row.precontact_velocity_m_s, bool)
        or not isinstance(row.precontact_velocity_m_s, (int, float))
        or not math.isfinite(row.precontact_velocity_m_s)
        or row.precontact_velocity_m_s <= 0.0
    ):
        raise ValueError("pre-contact velocity must be finite and positive")
    if (
        isinstance(row.impulse_n_s, bool)
        or not isinstance(row.impulse_n_s, (int, float))
        or not math.isfinite(row.impulse_n_s)
        or row.impulse_n_s <= 0.0
    ):
        raise ValueError("impulse must be finite and positive")
    if row.contacted is not True:
        raise ValueError("all primary trials must be contacted")
    if row.finalized is not True:
        raise ValueError("all primary trials must be finalized")
    if row.productive is not True:
        raise ValueError("all primary trials must be productive")
    if row.reason not in ("success", "window"):
        raise ValueError("reason must be success or window")
    for field, value in (
        ("depth_at_contact_m", row.depth_at_contact_m),
        ("peak_depth_m", row.peak_depth_m),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{field} depth must be a finite numeric value")
    if row.peak_depth_m - row.depth_at_contact_m <= 5e-4:
        raise ValueError("trial must show depth progress above 5e-4 m")


def build_primary_result(
    execution: ControlledDropExecution,
    trials: Sequence[ControlledDropTrial],
) -> ControlledDropResult:
    """Build the versioned primary result from one execution and five trials."""
    return ControlledDropResult(execution=execution, summary=summarize_primary_trials(trials))


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value

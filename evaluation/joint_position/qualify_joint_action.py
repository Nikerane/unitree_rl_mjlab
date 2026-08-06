"""Pure helpers for causal joint-target tape qualification."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np


def _finite_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def reduce_first_target_per_interval(
    targets_500hz: object, decimation: int = 10
) -> np.ndarray:
    """Keep the target applied at the first substep of each control interval."""
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation <= 0:
        raise ValueError("decimation must be a positive integer")
    targets = _finite_array(targets_500hz, name="targets_500hz", ndim=2)
    if targets.shape[0] == 0:
        raise ValueError("targets_500hz must contain at least one target")
    if targets.shape[0] % decimation != 0:
        raise ValueError("targets_500hz must contain complete control intervals")
    return targets[::decimation].copy()


def derive_scale_by_joint(
    tape: object,
    q_default: object,
    exploration_floor: float = 0.05,
    margin: float = 1.10,
) -> np.ndarray:
    """Derive per-joint action authority from target deviation plus exploration room."""
    target_tape = _finite_array(tape, name="tape", ndim=2)
    default = _finite_array(q_default, name="q_default", ndim=1)
    if target_tape.shape[0] == 0 or target_tape.shape[1] != default.shape[0]:
        raise ValueError("tape must be nonempty and match q_default width")
    if not math.isfinite(exploration_floor) or exploration_floor < 0.0:
        raise ValueError("exploration_floor must be finite and nonnegative")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    return margin * (np.max(np.abs(target_tape - default), axis=0) + exploration_floor)


def normalized_action_for_target(
    tape: object, q_default: object, scale: object
) -> np.ndarray:
    """Return absolute default-offset normalized actions for a target tape."""
    target_tape = _finite_array(tape, name="tape", ndim=2)
    default = _finite_array(q_default, name="q_default", ndim=1)
    scale_array = _finite_array(scale, name="scale", ndim=1)
    if (
        target_tape.shape[0] == 0
        or target_tape.shape[1] != default.shape[0]
        or scale_array.shape != default.shape
    ):
        raise ValueError("tape, q_default, and scale must have compatible widths")
    if np.any(scale_array <= 0.0):
        raise ValueError("scale must be positive")
    return (target_tape - default) / scale_array


def qualification_passes(
    rows: Sequence[dict[str, Any]], expected_seeds: tuple[int, ...] = tuple(range(1000, 1016))
) -> bool:
    """Return whether the replay rows are complete, ordered, and all explicitly passed."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    if len(rows) != len(expected_seeds):
        return False
    for row, expected_seed in zip(rows, expected_seeds, strict=True):
        if not isinstance(row, dict):
            return False
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != expected_seed:
            return False
        if row.get("passed") is not True:
            return False
    return True

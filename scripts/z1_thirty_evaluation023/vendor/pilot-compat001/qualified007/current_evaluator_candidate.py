"""Bounded, safety-on impact-speed evaluation; no learning or qualification claim.

Each requested goal gets a fresh, identically seeded environment. Comparisons use
only the first episode of each world, not asynchronously diverging reset streams.
Legacy tracking remains at reward time. Evaluation-only evidence separately
reports substep precontact path coverage and onset-aligned impact/load windows.
"""

from __future__ import annotations

import sys

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.impact_speed_interventions import (
    override_latched_goal,
    set_goal_before_first_action,
)
from scripts.impact_speed_observer import PreResetSnapshot, physical_snapshot


def compare_goal_responses(low: dict, high: dict, seed: int) -> dict:
    """Paired common-hit bootstrap, always accompanied by unconditional coverage.

    ``responsive`` is a diagnostic speed contrast, NOT a promotion/safety gate.
    IDs must already be in paired order; never silently reorder or drop misses.
    """
    n = len(low["hit"])
    for data in (low, high):
        if any(len(data[key]) != n for key in ("episode_id", "reset_hash", "hit", "speed")):
            raise ValueError("paired fields must have equal lengths")
        if len(set(data["episode_id"])) != n or any(not x for x in data["reset_hash"]):
            raise ValueError("paired IDs must be unique and reset hashes present")
    if low["episode_id"] != high["episode_id"] or low["reset_hash"] != high["reset_hash"]:
        raise ValueError("paired episode IDs and reset hashes must match in order")
    valid, coverage = [], {}
    for name, data in (("low", low), ("high", high)):
        available = np.array([hit and speed is not None and math.isfinite(speed)
                              for hit, speed in zip(data["hit"], data["speed"])], dtype=bool)
        valid.append(available)
        hits = sum(bool(x) for x in data["hit"])
        coverage[name] = dict(episodes=n, hits=hits, available=int(available.sum()), misses=n - hits)
    common = valid[0] & valid[1]
    differences = np.array([high["speed"][i] - low["speed"][i] for i in np.flatnonzero(common)])
    ci = None
    if len(differences) >= 2:
        rng = np.random.default_rng(seed)
        means = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return {"seed": seed, "bootstrap_samples": 2000, "common_valid_pairs": int(common.sum()),
            "paired_speed_difference": float(differences.mean()) if len(differences) else None,
            "ci95": ci, "responsive": bool(ci is not None and ci[0] > 0), "coverage": coverage,
            "interpretation": "common-valid speed contrast only; inspect miss coverage; not a qualification gate"}


def compare_paired_outcomes(
    low_rows: list[dict], high_rows: list[dict], seed: int, *, _include_routes: bool = True,
) -> dict:
    """Compare first-episode task, path, and joint-load outcomes by matched world."""
    def first_by_world(rows):
        first = sorted((row for row in rows if row.get("episode") == 0), key=lambda row: row["world"])
        worlds = [row["world"] for row in first]
        ids = [row.get("episode_id") for row in first]
        hashes = [row.get("reset_hash") for row in first]
        if (not first or len(set(worlds)) != len(worlds) or len(set(ids)) != len(ids)
                or any(not value for value in hashes)):
            raise ValueError("paired outcomes require unique first-episode worlds and reset hashes")
        return first

    low = first_by_world(low_rows)
    high = first_by_world(high_rows)
    if len(low) != len(high):
        raise ValueError("paired outcomes require the same number of first-episode worlds")
    for left, right in zip(low, high):
        if (left["world"] != right["world"] or left["episode_id"] != right["episode_id"]
                or left["reset_hash"] != right["reset_hash"]
                or left.get("route_sign") != right.get("route_sign")):
            raise ValueError("paired outcomes require matching worlds, reset hashes, and routes")
        if (isinstance(left.get("route_sign"), bool)
                or left.get("route_sign") not in _ROUTE_LABELS):
            raise ValueError("paired outcomes require valid route signs")

    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def bootstrap(differences, offset):
        if len(differences) < 2:
            return None
        rng = np.random.default_rng(seed + offset)
        means = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        return np.quantile(means, [.025, .975]).tolist()

    def metric_summary(extract, offset):
        paired_low, paired_high = [], []
        for left, right in zip(low, high):
            a, b = finite(extract(left)), finite(extract(right))
            if a is not None and b is not None:
                paired_low.append(a)
                paired_high.append(b)
        differences = np.asarray(paired_high) - np.asarray(paired_low)
        return {
            "common_pairs": len(differences),
            "low_mean": float(np.mean(paired_low)) if paired_low else None,
            "high_mean": float(np.mean(paired_high)) if paired_high else None,
            "mean_difference_high_minus_low": float(np.mean(differences)) if len(differences) else None,
            "ci95": bootstrap(differences, offset),
            "bootstrap_samples": 2000,
        }

    def outcome_value(row, path):
        value = row.get("measurement", {})
        for component in path:
            if not isinstance(value, dict):
                return None
            value = value.get(component)
        return value

    def speed(row):
        return row.get("measured_mps") if row.get("hit") else None

    def absolute_speed_error(row):
        measured, goal = finite(speed(row)), finite(row.get("goal_mps"))
        return None if measured is None or goal is None else abs(measured - goal)

    def axial_impulse(row):
        if not outcome_value(row, ("common_window", "window_complete")):
            return None
        return outcome_value(row, ("common_window", "axial_positive_ns"))

    def path_rms_mm(row):
        coverage = finite(outcome_value(row, ("precontact_path", "coverage")))
        if coverage is None or not math.isclose(coverage, 1., rel_tol=0., abs_tol=1e-12):
            return None
        value = finite(outcome_value(row, ("precontact_path", "observed_bin_rms_m")))
        return None if value is None else value * 1000

    def path_coverage(row):
        return outcome_value(row, ("precontact_path", "coverage"))

    def peak_sampled_nail_depth_mm(row):
        if row.get("complete") is not True:
            return None
        value = finite(row.get("nail_depth_peak_m"))
        return None if value is None else value * 1000

    def observed_samples(row):
        value = finite(outcome_value(row, ("loads", "episode", "observed_samples")))
        return value is not None and value > 0

    complete_pairs = []
    for left, right in zip(low, high):
        complete_pairs.append(
            left.get("complete") is True and right.get("complete") is True
            and observed_samples(left) and observed_samples(right)
        )
    full_path_pairs = sum(
        (finite(path_coverage(left)) is not None
         and math.isclose(finite(path_coverage(left)), 1., rel_tol=0., abs_tol=1e-12)
         and finite(path_coverage(right)) is not None
         and math.isclose(finite(path_coverage(right)), 1., rel_tol=0., abs_tol=1e-12))
        for left, right in zip(low, high)
    )

    joint_results = []
    for joint in range(6):
        low_peaks, high_peaks, low_violations, high_violations = [], [], [], []
        low_durations, high_durations = [], []
        low_observed_exceedances = high_observed_exceedances = 0
        low_observed_attempts = high_observed_attempts = 0
        for pair_index, (left, right) in enumerate(zip(low, high)):
            left_load = outcome_value(left, ("loads", "episode"))
            right_load = outcome_value(right, ("loads", "episode"))
            if not isinstance(left_load, dict) or not isinstance(right_load, dict):
                raise ValueError("paired outcomes require episode-level joint-load evidence")
            for load in (left_load, right_load):
                for key in ("peak_nms", "exceeded", "exceedance_duration_s"):
                    value = load.get(key)
                    if not isinstance(value, list) or len(value) != 6:
                        raise ValueError("joint-load evidence must contain six per-joint values")
            low_observed, high_observed = observed_samples(left), observed_samples(right)
            low_flag, high_flag = left_load["exceeded"][joint], right_load["exceeded"][joint]
            if low_observed:
                if not isinstance(low_flag, bool):
                    raise ValueError("observed joint-load exceedance flags must be boolean")
                low_observed_attempts += 1
                low_observed_exceedances += int(low_flag)
            if high_observed:
                if not isinstance(high_flag, bool):
                    raise ValueError("observed joint-load exceedance flags must be boolean")
                high_observed_attempts += 1
                high_observed_exceedances += int(high_flag)
            if complete_pairs[pair_index]:
                low_peak, high_peak = finite(left_load["peak_nms"][joint]), finite(right_load["peak_nms"][joint])
                low_duration = finite(left_load["exceedance_duration_s"][joint])
                high_duration = finite(right_load["exceedance_duration_s"][joint])
                if None in (low_peak, high_peak, low_duration, high_duration):
                    raise ValueError("complete paired joint-load evidence must be finite")
                if not isinstance(low_flag, bool) or not isinstance(high_flag, bool):
                    raise ValueError("complete paired joint-load exceedance flags must be boolean")
                low_peaks.append(low_peak)
                high_peaks.append(high_peak)
                low_violations.append(low_flag)
                high_violations.append(high_flag)
                low_durations.append(low_duration)
                high_durations.append(high_duration)
        added = sum(not a and b for a, b in zip(low_violations, high_violations))
        removed = sum(a and not b for a, b in zip(low_violations, high_violations))
        joint_results.append({
            "joint": joint + 1,
            "complete_paired_attempts": len(low_peaks),
            "censored_paired_attempts": len(low) - len(low_peaks),
            "low_peak_mean_nms": float(np.mean(low_peaks)) if low_peaks else None,
            "high_peak_mean_nms": float(np.mean(high_peaks)) if high_peaks else None,
            "mean_peak_difference_high_minus_low_nms": float(np.mean(np.asarray(high_peaks) - np.asarray(low_peaks))) if low_peaks else None,
            "peak_difference_ci95": bootstrap(np.asarray(high_peaks) - np.asarray(low_peaks), 100 + joint),
            "low_exceedance_attempts": sum(low_violations),
            "high_exceedance_attempts": sum(high_violations),
            "low_exceedance_rate": float(np.mean(low_violations)) if low_violations else None,
            "high_exceedance_rate": float(np.mean(high_violations)) if high_violations else None,
            "violation_rate_difference_high_minus_low": float(np.mean(high_violations) - np.mean(low_violations)) if low_violations else None,
            "added_violation_pairs": added,
            "removed_violation_pairs": removed,
            "mean_exceedance_duration_difference_high_minus_low_s": float(np.mean(np.asarray(high_durations) - np.asarray(low_durations))) if low_durations else None,
            "low_observed_exceedance_attempts": low_observed_exceedances,
            "high_observed_exceedance_attempts": high_observed_exceedances,
            "low_observed_attempts": low_observed_attempts,
            "high_observed_attempts": high_observed_attempts,
            "inferential_contrast_semantics": "paired peak, duration, and violation-rate contrasts use only pairs whose two first episodes are complete with positive episode-load samples",
            "observed_exceedance_semantics": "lower-bound counts from any attempt with positive observed samples; incomplete attempts are not treated as non-violations",
        })

    route_counts = {str(sign): sum(row.get("route_sign") == sign for row in low)
                    for sign in (-1, 0, 1)}
    report = {
        "paired_episode_count": len(low),
        "route_counts": route_counts,
        "coverage": {
            "attempts": len(low),
            "low_hits": sum(bool(row.get("hit")) for row in low),
            "low_misses": sum(bool(row.get("miss")) for row in low),
            "high_hits": sum(bool(row.get("hit")) for row in high),
            "high_misses": sum(bool(row.get("miss")) for row in high),
            "low_valid_contact_speeds": sum(finite(speed(row)) is not None for row in low),
            "high_valid_contact_speeds": sum(finite(speed(row)) is not None for row in high),
            "absolute_speed_error_common_pairs": sum(
                finite(speed(left)) is not None and finite(speed(right)) is not None
                for left, right in zip(low, high)
            ),
            "low_complete_common_windows": sum(bool(outcome_value(row, ("common_window", "window_complete"))) for row in low),
            "high_complete_common_windows": sum(bool(outcome_value(row, ("common_window", "window_complete"))) for row in high),
            "full_path_pairs": int(full_path_pairs),
            "incomplete_path_pairs": len(low) - int(full_path_pairs),
            "complete_paired_attempts": sum(complete_pairs),
            "censored_paired_attempts": len(low) - sum(complete_pairs),
            "speed_error_semantics": "absolute error conditional on both arms having valid contact-speed measurement; hit/miss coverage reported separately",
            "path_coverage_rule": "precontact spline RMS uses only matched pairs with coverage == 1.0 (all 16 bins); partial paths are included in coverage, not RMS",
        },
        "paired_metrics": {
            "terminal_speed_mps": metric_summary(speed, 1),
            "absolute_speed_error_mps": metric_summary(absolute_speed_error, 2),
            "positive_axial_impulse_50ms_ns": metric_summary(axial_impulse, 3),
            "path_coverage_fraction": metric_summary(path_coverage, 5),
            "precontact_spline_rms_mm": metric_summary(path_rms_mm, 4),
            "peak_sampled_nail_depth_mm": metric_summary(peak_sampled_nail_depth_mm, 6),
        },
        "per_joint_loads": joint_results,
        "difference_direction": "high condition minus low condition",
        "uncertainty": "paired whole-world percentile bootstrap; descriptive, not a qualification or global-optimum test",
    }
    if _include_routes:
        report["by_route"] = {}
        for sign, label in _ROUTE_LABELS.items():
            route_low = [row for row in low if row.get("route_sign") == sign]
            route_high = [row for row in high if row.get("route_sign") == sign]
            if route_low:
                report["by_route"][label] = compare_paired_outcomes(
                    route_low, route_high, seed + sign + 2, _include_routes=False
                )
    return report


class EpisodeAccumulator:
    """Consume policy-step snapshots, keeping first-episode evidence until done."""

    def __init__(self, num_envs: int, *, gamma: float, coefficient: float):
        self.gamma, self.coefficient = gamma, coefficient
        self.episodes = [0] * num_envs
        self.active = [self._new(i) for i in range(num_envs)]
        self.rows: list[dict] = []

    def _new(self, world):
        return dict(world=world, episode=self.episodes[world], steps=0, charge_count=0,
                    goal_mps=None, measured_mps=None, finalized=False, productive=False,
                    hit=False, miss=False, incomplete=False, timeout=False, complete=False,
                    route_sign=None,
                    first_contact_time_s=None, cost=0., discounted_cost=0., selection_count=0,
                    selected_raw=None, clipping_count=0, selector_clipping_count=0,
                    selector_lower_clipping_count=0, selector_upper_clipping_count=0,
                    tracking_sum_sq_m2=0., tracking_max_mm=0.,
                    nail_depth_peak_m=0.,
                    joint_target_sum_sq_rad2=0., joint_impulse_peak_nms=[0.] * 6,
                    impulse_exceedance_steps=[0] * 6, delivered_impulse_ns=0.,
                    cat_delta_sum=0., cat_delta_max=0., cat_active_steps=0,
                    kp_sum=[0.] * 6, kd_sum=[0.] * 6)

    def update(self, outcome, physical, raw, pending, evaluation=None,
               contact_safety_records=None, *, physical_done=None):
        completion=outcome["done"] if physical_done is None else physical_done
        # Inputs are immutable evidence for THIS transition, not live reset tensors.
        route_signs = physical.get("route_sign")
        if route_signs is not None:
            if not isinstance(route_signs, torch.Tensor) or route_signs.shape != (len(self.active),):
                raise ValueError("route signs must be one pre-reset sign per world")
            route_signs = route_signs.detach().cpu().tolist()
        if (contact_safety_records is not None
                and len(contact_safety_records) != len(self.active)):
            raise ValueError("contact safety records must contain one entry per world")
        for world, row in enumerate(self.active):
            if physical_done is not None and self.episodes[world]>=1:
                continue
            if route_signs is not None:
                raw_route_sign = route_signs[world]
                if isinstance(raw_route_sign, bool) or raw_route_sign not in (-1, 0, 1):
                    raise ValueError("route sign must be one of -1, 0, or +1")
                route_sign = int(raw_route_sign)
                if row["route_sign"] is None:
                    row["route_sign"] = route_sign
                elif route_sign != row["route_sign"]:
                    raise ValueError("route sign changed before the episode reset")
            if evaluation is not None:
                row["measurement"] = evaluation[world]
            timing = (None if contact_safety_records is None
                      else contact_safety_records[world])
            if row["episode"] == 0 and timing is not None:
                if not isinstance(timing, dict):
                    raise ValueError("contact safety records must be mappings or None")
                timing = dict(timing)
                timing["episode_steps"] = row["steps"] + 1
                if evaluation is not None:
                    reason = evaluation[world].get("termination_reasons")
                    if not reason:
                        raise ValueError("contact safety timing requires a termination reason")
                    timing["termination_reasons"] = reason
                else:
                    raise ValueError("contact safety timing requires pre-reset evaluation evidence")
                row["contact_safety_timing"] = timing
            elif row["episode"] > 0 and timing is not None:
                raise ValueError("contact safety summary may observe only first episodes")
            if (contact_safety_records is not None and row["episode"] == 0
                    and bool(completion[world]) and timing is None):
                raise ValueError("completed first episode lacks contact safety timing evidence")
            step = row["steps"]
            if bool(outcome["charged"][world]):
                if row["charge_count"]:
                    raise ValueError("speed event charged twice in one episode")
                row["charge_count"] = 1
                for key in ("finalized", "productive", "miss", "incomplete"):
                    row[key] = bool(outcome[key][world])
                row["measured_mps"] = (float(outcome["measured_mps"][world])
                                       if bool(outcome["valid_measured"][world]) else None)
                row["cost"] = float(outcome["cost"][world]) * self.coefficient
                row["discounted_cost"] = row["cost"] * self.gamma**step
            row["goal_mps"] = float(outcome["goal_mps"][world])
            nail_depth = float(physical["nail_depth_m"][world])
            if not math.isfinite(nail_depth) or nail_depth < 0.:
                raise ValueError("clamped nail depth must be finite and nonnegative")
            row["nail_depth_peak_m"] = max(row["nail_depth_peak_m"], nail_depth)
            row["hit"] |= bool(physical["started"][world]) or bool(outcome["started"][world])
            if row["hit"] and row["first_contact_time_s"] is None:
                row["first_contact_time_s"] = float(physical["first_contact_time_s"][world])
            if bool(pending[world]):
                row["selection_count"] += 1
                row["selected_raw"] = float(raw[world, 12])
                row["selector_clipping_count"] += int(abs(row["selected_raw"]) > 1.)
                row["selector_lower_clipping_count"] += int(row["selected_raw"] < -1.)
                row["selector_upper_clipping_count"] += int(row["selected_raw"] > 1.)
            row["clipping_count"] += int((raw[world].abs() > 1).sum())
            path_error = float(physical["path_error_m"][world])
            row["tracking_sum_sq_m2"] += path_error**2
            row["tracking_max_mm"] = max(row["tracking_max_mm"], path_error * 1000)
            row["joint_target_sum_sq_rad2"] += float(physical["joint_target_error_rad"][world])**2
            row["delivered_impulse_ns"] = float(physical["delivered_impulse_ns"][world])
            for j in range(6):
                row["joint_impulse_peak_nms"][j] = max(row["joint_impulse_peak_nms"][j], float(physical["joint_impulse_nms"][world, j]))
                row["impulse_exceedance_steps"][j] += int(physical["impulse_exceeded"][world, j])
                for key in ("kp", "kd"):
                    row[key + "_sum"][j] += float(physical[key][world, j])
            delta = float(physical["cat_delta"][world])
            row["cat_delta_sum"] += delta
            row["cat_delta_max"] = max(row["cat_delta_max"], delta)
            row["cat_active_steps"] += int(delta > 0)
            row["steps"] += 1
            if bool(completion[world]):
                row["complete"] = True
                row["timeout"] = bool(physical["timeout"][world])
                self.rows.append(self._finish_row(row))
                self.episodes[world] += 1
                self.active[world] = self._new(world)

    @staticmethod
    def _finish_row(row):
        result = dict(row)
        n = row["steps"]
        result["tracking_rms_mm"] = math.sqrt(result.pop("tracking_sum_sq_m2") / n) * 1000
        result["joint_target_rms_rad"] = math.sqrt(result.pop("joint_target_sum_sq_rad2") / n)
        result["cat_delta_mean"] = result.pop("cat_delta_sum") / n
        for key in ("kp", "kd"):
            result[key + "_mean"] = [x / n for x in result.pop(key + "_sum")]
        return result

    def finish(self):
        return self.rows + [self._finish_row(row) for row in self.active if row["steps"]]


def validate_selector_counts(rows, *, mode: str, expected_episode_count: int | None = None) -> dict:
    """Require one speed selection per policy episode and none for external goals."""
    if mode not in ("external", "policy", "intervention", "paired-fixed"):
        raise ValueError("selector evidence requires a known speed-selection mode")
    first = [row for row in rows if row.get("episode") == 0]
    if not first or (expected_episode_count is not None and len(first) != expected_episode_count):
        raise ValueError("selector evidence requires every first episode")
    expected = 0 if mode in ("external", "paired-fixed") else 1
    matched = sum(row.get("selection_count") == expected for row in first)
    if matched != len(first):
        raise ValueError(f"selector count must be exactly {expected} per first episode")
    return {
        "first_episode_count": len(first),
        "expected_per_episode": expected,
        "episodes_with_expected_count": matched,
        "total_selections": sum(row["selection_count"] for row in first),
    }


_ROUTE_LABELS = {-1: "R-", 0: "R0", 1: "R+"}
SHADOW_POST_ONSET_HORIZON_S = .250
SHADOW_HARD_CAP_FROM_RESET_S = 1.0


def summarize_selector_clipping(rows, *, expected_episode_count: int) -> dict:
    """Count raw coordinate 12 only while the first-attempt selector is active."""
    first = [row for row in rows if row.get("episode") == 0]
    if (len(first) != expected_episode_count
            or len({row["world"] for row in first}) != expected_episode_count):
        raise ValueError("selector clipping requires every unique first attempt")
    for row in first:
        count, raw = row["selection_count"], row["selected_raw"]
        if count not in (0, 1) or (count and (raw is None or not math.isfinite(raw))):
            raise ValueError("selector clipping requires at most one finite first decision")
        expected = (int(count and abs(raw) > 1.), int(count and raw < -1.), int(count and raw > 1.))
        actual = tuple(row[key] for key in ("selector_clipping_count", "selector_lower_clipping_count",
                                           "selector_upper_clipping_count"))
        if actual != expected:
            raise ValueError("selector clipping disagrees with the active first decision")

    def summarize(group):
        active = sum(row["selection_count"] for row in group)
        clipped = sum(row["selector_clipping_count"] for row in group)
        return {"attempts": len(group), "active_decisions": active,
                "clipped_decisions": clipped,
                "lower_clipped_decisions": sum(row["selector_lower_clipping_count"] for row in group),
                "upper_clipped_decisions": sum(row["selector_upper_clipping_count"] for row in group),
                "fraction_of_active_decisions": clipped / active if active else None}

    return {**summarize(first), "by_route": {
        label: summarize([row for row in first if row.get("route_sign") == sign])
        for sign, label in _ROUTE_LABELS.items()}}


def _finite_mean(values) -> dict:
    finite = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return {"mean": sum(finite) / len(finite) if finite else None, "count": len(finite)}


def summarize_route_strata(
    rows, *, expected_route_signs: tuple[int, ...], require_all: bool = True,
    expected_episode_count: int | None = None,
) -> dict:
    """Summarize only first episodes by the route captured before native reset."""
    if (not expected_route_signs
        or len(set(expected_route_signs)) != len(expected_route_signs)
        or any(isinstance(sign, bool) or sign not in _ROUTE_LABELS for sign in expected_route_signs)):
        raise ValueError("expected route signs must be a unique subset of -1, 0, and +1")
    first = sorted((row for row in rows if row.get("episode") == 0), key=lambda row: row["world"])
    if not first or (expected_episode_count is not None and len(first) != expected_episode_count):
        raise ValueError("route evidence requires every first episode")
    groups = {sign: [] for sign in expected_route_signs}
    for row in first:
        sign = row.get("route_sign")
        if isinstance(sign, bool) or sign not in _ROUTE_LABELS:
            raise ValueError("route identity is missing or invalid in a first-episode row")
        if sign not in groups:
            raise ValueError(f"observed route sign {sign} is outside the task route contract")
        groups[sign].append(row)
    missing = [sign for sign, group in groups.items() if not group]
    if require_all and missing:
        raise ValueError(f"evaluation did not observe required route sign(s): {missing}")

    routes = {}
    for sign, group in groups.items():
        measurements = [row.get("measurement", {}) for row in group]
        complete = [measurement for measurement in measurements
                    if measurement.get("common_window", {}).get("window_complete")]
        loads = [measurement.get("loads", {}).get("episode", {}).get("exceeded")
                 for measurement in measurements]
        path_values = [measurement.get("precontact_path", {}).get("observed_bin_rms_m")
                       for measurement in measurements]
        complete_depth_attempts = sum(
            row.get("complete") is True and row.get("nail_depth_peak_m") is not None
            for row in group
        )
        routes[str(sign)] = {
            "route_label": _ROUTE_LABELS[sign],
            "attempts": len(group),
            "hits": sum(bool(row.get("hit")) for row in group),
            "misses": sum(bool(row.get("miss")) for row in group),
            "selector_selections": sum(int(row.get("selection_count", 0)) for row in group),
            "requested_goal_mps": _finite_mean(row.get("goal_mps") for row in group),
            "selected_goal_mps": _finite_mean(row.get("selected_goal_mps") for row in group),
            "achieved_speed_mps": _finite_mean(row.get("measured_mps") for row in group),
            "peak_sampled_nail_depth_mm": _finite_mean(
                None if row.get("complete") is not True or row.get("nail_depth_peak_m") is None
                else float(row["nail_depth_peak_m"]) * 1000 for row in group),
            "complete_depth_attempts": complete_depth_attempts,
            "censored_depth_attempts": len(group) - complete_depth_attempts,
            "censored_observed_peak_depth_mm": _finite_mean(
                None if row.get("complete") is True or row.get("nail_depth_peak_m") is None
                else float(row["nail_depth_peak_m"]) * 1000 for row in group),
            "complete_common_windows": len(complete),
            "positive_axial_impulse_50ms_ns": _finite_mean(
                metric.get("axial_positive_ns") for metric in
                (measurement["common_window"] for measurement in complete)),
            "any_joint_load_cap_exceedance_attempts": sum(
                bool(exceeded) and any(exceeded) for exceeded in loads),
            "precontact_observed_bin_rms_mm": _finite_mean(
                None if value is None else float(value) * 1000 for value in path_values),
        }
    return {
        "first_episode_count": len(first),
        "route_counts": {str(sign): len(group) for sign, group in groups.items()},
        "peak_sampled_nail_depth_mm": _finite_mean(
            None if row.get("complete") is not True or row.get("nail_depth_peak_m") is None
            else float(row["nail_depth_peak_m"]) * 1000 for row in first),
        "complete_depth_attempts": sum(
            row.get("complete") is True and row.get("nail_depth_peak_m") is not None
            for row in first),
        "censored_depth_attempts": sum(
            row.get("complete") is not True or row.get("nail_depth_peak_m") is None
            for row in first),
        "censored_observed_peak_depth_mm": _finite_mean(
            None if row.get("complete") is True or row.get("nail_depth_peak_m") is None
            else float(row["nail_depth_peak_m"]) * 1000 for row in first),
        "nail_depth_measurement_semantics": (
            "peak clamped nail depth sampled once per policy transition (50 Hz in this task); "
            "the complete-case mean is an episode maximum, not retained/final depth or a "
            "physics-rate maximum. Incomplete attempts are censored from that mean and their "
            "observed peak is reported separately. Depth, accepted contact, speed misses, and "
            "joint safety are independent outcomes; misses may have positive observed depth"
        ),
        "routes": routes,
        "required_route_signs": list(expected_route_signs),
    }


def summarize_contact_safety_timing(rows, *, expected_episode_count: int | None = None) -> dict:
    """Summarize first-episode timing records without treating missing coverage as safe."""
    first = sorted((row for row in rows if row.get("episode") == 0), key=lambda row: row["world"])
    if not first or (expected_episode_count is not None and len(first) != expected_episode_count):
        raise ValueError("contact safety timing requires every first episode")
    records = [row.get("contact_safety_timing") for row in first]
    if any(record is None for record in records):
        raise ValueError("contact safety timing is missing from a first episode")
    if any(record.get("schema_version") != "contact_safety_timing_v1"
           or record.get("sample_count", 0) <= 0 for record in records):
        raise ValueError("contact safety timing schema or sample coverage is invalid")

    joint_names = records[0]["joint_velocity"]["joint_names"]
    velocity_limit = records[0]["joint_velocity"]["limit_rad_s"]
    caps = records[0]["joint_load"]["caps_nms"]
    if len(joint_names) != 6 or len(caps) != 6:
        raise ValueError("contact safety timing requires six ordered arm joints")

    def per_joint(channel, field):
        columns = [record[channel][field] for record in records]
        if any(len(values) != 6 for values in columns):
            raise ValueError(f"contact safety timing {channel}.{field} has invalid width")
        values = [[None if value is None else float(value) for value in row] for row in columns]
        if any(value is None or not math.isfinite(value) for row in values for value in row):
            raise ValueError(f"contact safety timing {channel}.{field} is non-finite or missing")
        return [max(joint) for joint in zip(*values)]

    def crossing_counts(channel):
        crossing_field = "first_crossing_sample"
        relative_field = "relative_to_accepted_contact_s"
        crossed, before, at_or_after, unaligned = [0] * 6, [0] * 6, [0] * 6, [0] * 6
        for record in records:
            event = record[channel][crossing_field]
            relative = record[channel][relative_field]
            if len(event) != 6 or len(relative) != 6:
                raise ValueError(f"contact safety timing {channel} event width is invalid")
            for joint, sample in enumerate(event):
                if sample is None:
                    continue
                crossed[joint] += 1
                offset = relative[joint]
                if offset is None:
                    unaligned[joint] += 1
                elif float(offset) < 0:
                    before[joint] += 1
                else:
                    at_or_after[joint] += 1
        return {"any": crossed, "before_accepted_contact": before,
                "at_or_after_accepted_contact": at_or_after,
                "not_aligned_to_accepted_contact": unaligned}

    for record in records[1:]:
        if (record["joint_velocity"]["joint_names"] != joint_names
                or record["joint_velocity"]["limit_rad_s"] != velocity_limit
                or record["joint_load"]["caps_nms"] != caps):
            raise ValueError("contact safety timing limits or joint ordering changed")

    termination_reasons = {}
    for record in records:
        reason = record.get("termination_reasons") or "unknown"
        if reason == "unknown":
            raise ValueError("contact safety timing lacks an evaluation termination reason")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    return {
        "schema_version": "contact_safety_timing_v1",
        "first_episode_count": len(first),
        "timed_episode_count": len(records),
        "completed_episode_count": sum(bool(row.get("complete")) for row in first),
        "missing_episode_count": 0,
        "episodes_with_raw_contact": sum(record.get("first_raw_contact_sample") is not None for record in records),
        "episodes_with_accepted_contact": sum(record.get("accepted_contact_sample") is not None for record in records),
        "termination_reasons": termination_reasons,
        "observation_semantics": "one read-only sample per physics integration from reset through evaluation termination; common-window protocol defers success reset only through its prescribed measurement window",
        "joint_names": joint_names,
        "joint_velocity": {
            "limit_rad_s": velocity_limit,
            "maximum_observed_peak_abs_rad_s": per_joint("joint_velocity", "peak_abs_rad_s"),
            "episodes_with_first_crossing": crossing_counts("joint_velocity"),
        },
        "joint_load": {
            "caps_nms": caps,
            "signal": records[0]["joint_load"]["signal"],
            "maximum_observed_peak_nms": per_joint("joint_load", "peak_nms"),
            "episodes_with_first_crossing": crossing_counts("joint_load"),
        },
        "torque_saturation": {
            "episodes_with_first_saturation": crossing_counts("torque_saturation"),
        },
        "episode_steps": _finite_mean(record["episode_steps"] for record in records),
        "sample_count": _finite_mean(record["sample_count"] for record in records),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--speed-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--speed-cost", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--balanced-routes", action="store_true",
                        help="balance first-reset R-/R0/R+ by world ID, preserving the ordinary route RNG draw")
    parser.add_argument("--mode", choices=("external", "policy", "paired", "intervention", "smoke"), required=True)
    parser.add_argument("--goals", nargs="+", type=float,
                        help="one fixed external goal for shadow replay, or low middle high")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampled-pd500", action="store_true")
    parser.add_argument("--sampled-pd500-gain-mode", choices=("fixed", "variable"), default="fixed")
    parser.add_argument("--measurement-protocol", choices=("native", "common-window"), default="native")
    parser.add_argument("--trace-contact", action="store_true",
                        help="save first-episode physics traces for sampled PD evaluation")
    parser.add_argument("--post-terminal-shadow", action="store_true",
                        help="continue a common-window policy rollout after native success for a bounded diagnostic")
    parser.add_argument("--shadow-post-onset-horizon-s", type=float,
                        help="post-contact shadow observation duration in seconds (default: 0.25)")
    parser.add_argument("--shadow-hard-cap-from-reset-s", type=float,
                        help="post-terminal shadow hard cap from reset in seconds (default: 1.0)")
    parser.add_argument("--contact-safety-summary", action="store_true",
                        help="record compact first per-joint safety crossings for paired common-window evaluation")
    parser.add_argument("--cat-dose-audit", action="store_true",
                        help="record first-episode CaT components for an offline frozen-trace dose counterfactual")
    parser.add_argument("--cat-dose-target-max-p", type=float, default=.75,
                        help="offline velocity CaT max_p used only by --cat-dose-audit (default: 0.75)")
    args = parser.parse_args(argv)
    shadow_timing_overridden = (
        args.shadow_post_onset_horizon_s is not None
        or args.shadow_hard_cap_from_reset_s is not None
    )
    if shadow_timing_overridden and not args.post_terminal_shadow:
        parser.error("shadow timing overrides require --post-terminal-shadow")
    args.shadow_post_onset_horizon_s = (
        SHADOW_POST_ONSET_HORIZON_S if args.shadow_post_onset_horizon_s is None
        else args.shadow_post_onset_horizon_s
    )
    args.shadow_hard_cap_from_reset_s = (
        SHADOW_HARD_CAP_FROM_RESET_S if args.shadow_hard_cap_from_reset_s is None
        else args.shadow_hard_cap_from_reset_s
    )
    if args.post_terminal_shadow and (
            not math.isfinite(args.shadow_post_onset_horizon_s)
            or not math.isfinite(args.shadow_hard_cap_from_reset_s)
            or args.shadow_post_onset_horizon_s <= 0
            or args.shadow_hard_cap_from_reset_s <= args.shadow_post_onset_horizon_s):
        parser.error("shadow timing must be finite and positive, with hard cap greater than horizon")
    lo, hi = args.speed_bounds
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi and
            math.isfinite(args.speed_cost) and args.speed_cost > 0 and args.num_envs > 0):
        parser.error("require positive finite bounds/cost and positive num-envs")
    if args.mode != "smoke" and args.checkpoint is None:
        parser.error("non-smoke evaluation requires --checkpoint")
    if args.mode in ("external", "paired", "intervention"):
        if args.post_terminal_shadow and args.mode == "external":
            if (args.goals is None or len(args.goals) != 1
                    or not lo <= args.goals[0] <= hi):
                parser.error("external post-terminal shadow requires one --goal inside speed bounds")
        elif (args.goals is None or len(args.goals) != 3
              or not lo <= args.goals[0] < args.goals[1] < args.goals[2] <= hi):
            parser.error("require three increasing --goals inside speed bounds")
    elif args.goals is not None:
        parser.error("--goals only applies to external, paired, or intervention")
    if args.output.exists():
        parser.error("output directory already exists; refusing overwrite")
    if args.mode == "smoke" and 64 % args.num_envs:
        parser.error("smoke num-envs must divide 64 (exactly 64 world transitions)")
    if args.balanced_routes and (args.num_envs % 3 or args.mode == "smoke"):
        parser.error("balanced routes require a non-smoke cohort with num-envs divisible by three")
    if args.sampled_pd500_gain_mode == "variable" and not args.sampled_pd500:
        parser.error("variable gain mode requires sampled PD control")
    if args.trace_contact and not args.sampled_pd500:
        parser.error("contact trace requires sampled PD control")
    if args.post_terminal_shadow and (
            args.mode not in ("policy", "external") or not args.sampled_pd500
            or args.sampled_pd500_gain_mode != "variable"
            or args.measurement_protocol != "common-window"
            or not args.trace_contact
            or (args.mode == "policy" and args.goals is not None)
            or (args.mode == "external" and (args.goals is None or len(args.goals) != 1))):
        parser.error(
            "post-terminal shadow requires policy mode without goals or external mode with one goal, "
            "variable sampled-PD, common-window measurement, and the full contact trace"
        )
    if args.contact_safety_summary and not args.sampled_pd500:
        parser.error("contact safety summary requires sampled PD control")
    if args.contact_safety_summary and args.trace_contact:
        parser.error("choose either the full contact trace or compact contact safety summary")
    if args.cat_dose_audit and (not math.isfinite(args.cat_dose_target_max_p)
                                or not .5 < args.cat_dose_target_max_p <= 1.):
        parser.error("CaT dose audit target max_p must be greater than baseline 0.5 and no larger than 1")
    if args.cat_dose_audit and (args.mode != "policy" or args.goals is not None
                                or args.measurement_protocol != "native"):
        parser.error("CaT dose audit requires one autonomous policy replay using the native protocol")
    if args.contact_safety_summary and (
            args.mode not in ("policy", "paired", "external", "smoke") or args.measurement_protocol != "common-window"):
        parser.error("contact safety summary requires policy, paired, external or smoke common-window evaluation")
    return args


class PostTerminalShadowProtocol:
    """Suppress only nail-success reset while retaining its native success/window evidence."""

    def __init__(self, env):
        self.env = env
        self._installed = False

    def __enter__(self):
        manager = self.env.termination_manager
        try:
            index = manager._term_names.index("nail_driven")
        except (AttributeError, ValueError) as exc:
            raise ValueError("post-terminal shadow requires the native nail_driven termination") from exc
        if not hasattr(self.env, "_strike_evaluation"):
            raise ValueError("post-terminal shadow requires strike-evaluation evidence")
        self.cfg = manager._term_cfgs[index]
        self.original = self.cfg.func

        def continue_after_native_success(env, **params):
            native_success = self.original(env, **params)
            env._strike_evaluation.success_termination(native_success)
            return torch.zeros_like(native_success, dtype=torch.bool)

        self.cfg.func = continue_after_native_success
        self._installed = True
        return self

    def __exit__(self, *exc):
        if self._installed:
            self.cfg.func = self.original
            self._installed = False


class BalancedRouteReset:
    """Balance one reset before the native route-pose event and then restore it.

    The native sampler still consumes its normal random draw. All joint/nail
    reset events and observation noise retain their ordinary ordering and RNG.
    """

    def __init__(self, env):
        if env.num_envs % 3:
            raise ValueError("balanced route cohort size must be divisible by three")
        self.env = env

    def __enter__(self):
        self.cfg = self.env.event_manager.get_term_cfg("sample_strike_route_signs")
        if self.cfg.params.get("fixed_route_sign") is not None:
            raise ValueError("balanced routes require the three-route reset treatment")
        if "reset_strike_route_joints" not in self.env.cfg.events:
            raise ValueError("balanced routes require physical route-pose reset")
        self.original = self.cfg.func

        def balanced_sampler(env, env_ids, **params):
            self.original(env, env_ids, **params)
            ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
                   else env_ids)
            ref = env._strike_reference
            signs = ref.route_signs()
            signs[ids] = (ids % 3 - 1).to(dtype=signs.dtype)
            ref.set_route_signs(signs)

        self.cfg.func = balanced_sampler
        return self

    def __exit__(self, *exc):
        self.cfg.func = self.original


def _digest(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reset_evidence(env, obs):
    """Reset physical state and RNG evidence, not a portable replay checkpoint."""
    robot, nail = env.scene["robot"].data, env.scene["nail_block"].data
    return [_digest(robot.joint_pos[i], robot.joint_vel[i], nail.joint_pos[i], nail.joint_vel[i],
                    obs["actor"][i, :40], obs["critic"][i, :40]) for i in range(env.num_envs)]


def reset_identity_evidence(env, obs):
    """Separate physical/reference identity from intentionally varying feedback.

    Complete generalized positions/velocities include the nail slide; entity
    sites capture world-space robot/object geometry. Frozen spline anchors and
    route identity are included independently of actor/critic observations.
    """
    ref = env._strike_reference
    signs = ref.route_signs()
    fields = {
        "qpos": env.sim.data.qpos, "qvel": env.sim.data.qvel,
        "robot_sites_w": env.scene["robot"].data.site_pos_w,
        "nail_sites_w": env.scene["nail_block"].data.site_pos_w,
        "reference_head0_w": ref._head0, "reference_nail_w": ref._nail,
        "reference_target_w": ref._target, "route_sign": signs,
    }
    if any(not bool(torch.isfinite(value).all()) for value in (*fields.values(), *obs.values())):
        raise ValueError("reset identity requires finite physical state and observations")
    return {
        "schema_version": "physical_spline_reset_v1",
        "physical_reset_hashes": [_digest(*(value[i] for value in fields.values()))
                                  for i in range(env.num_envs)],
        "observation_hashes": [{group: _digest(obs[group][i]) for group in ("actor", "critic")}
                               for i in range(env.num_envs)],
        # The shared first 30 coordinates are physical sensors: noisy actor
        # minus clean critic isolates realized sensor noise. Spline feedback
        # starts at coordinate 30 and is deliberately excluded.
        "sensor_noise_hashes": [_digest(obs["actor"][i, :30] - obs["critic"][i, :30])
                                for i in range(env.num_envs)],
        "physical_reset_records": [{name: value[i].detach().cpu().tolist() for name, value in fields.items()}
                                   for i in range(env.num_envs)],
    }


def _parity_models(runner, directory):
    runner.export_policy_to_jit(str(directory), "actor.pt")
    jit = torch.jit.load(str(directory / "actor.pt"))
    backend = None
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError:
        return jit, backend
    runner.export_policy_to_onnx(str(directory), "actor.onnx")
    model = onnx.load(str(directory / "actor.onnx"))
    onnx.checker.check_model(model)
    backend = ReferenceEvaluator(model)
    return jit, backend


def _condition(args, goal, index):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from tensordict import TensorDict
    from src.tasks.hammer.rl.impact_speed_contract import (
        build_impact_speed_metadata, validate_impact_speed_metadata,
    )
    from src.tasks.hammer.rl.impact_speed_wrapper import (configure_impact_speed_experiment,
        configure_impact_speed_controller, install_impact_speed_controller, wrap_impact_speed_env)
    from src.tasks.hammer.mdp.strike_evaluation import (
        configure_strike_evaluation, MeasurementProtocol, summarize_strike_evaluations,
        LOAD_SIGNAL_SEMANTICS)
    from scripts import contact_trace as contact_trace_module
    from scripts.contact_trace import ContactSafetySummary, ContactTrace
    from scripts.cat_dose_audit import (
        CatDoseTrace, summarize_cat_dose_trace, validate_baseline_contract,
    )

    # Play configs remove the finite timeout and alter reset/noise distributions.
    # Evaluation retains the training task's horizon, safety, and reset treatment.
    cfg, agent_cfg = load_env_cfg(args.task, play=False), load_rl_cfg(args.task)
    mode = "external" if args.mode == "external" else "policy"
    # A checkpoint-bearing smoke uses its saved phase, still validated exactly below.
    if args.mode == "smoke" and args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        mode = payload["infos"]["impact_speed_contract"]["speed_mode"]
    configure_impact_speed_experiment(cfg, bounds=tuple(args.speed_bounds), cost=args.speed_cost, mode=mode)
    configure_impact_speed_controller(cfg, sampled_pd500=args.sampled_pd500, gain_mode=args.sampled_pd500_gain_mode)
    configure_strike_evaluation(cfg)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    directory = args.output / f"condition-{index}"
    directory.mkdir()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    adapter = getattr(args, "horizon_adapter", None)
    adapter_entered = False
    try:
        install_impact_speed_controller(env, sampled_pd500=args.sampled_pd500, gain_mode=args.sampled_pd500_gain_mode)
        wrapped = wrap_impact_speed_env(env, agent_cfg.clip_actions)
        runner = load_runner_cls(args.task)(wrapped, asdict(agent_cfg), device=args.device)
        checkpoint_infos = None
        if args.checkpoint is not None:
            checkpoint_infos = runner.load(
                str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
            )
        contract = build_impact_speed_metadata(env, wrapped.clip_actions)
        checkpoint_transfer = None
        if args.checkpoint is not None:
            checkpoint_contract = (checkpoint_infos or {}).get("impact_speed_contract")
            validate_impact_speed_metadata(checkpoint_contract, contract)
            checkpoint_transfer = {
                "loaded_model_components": ["actor"],
                "restored_runner_state": ["common_step_counter"],
                "exact_impact_speed_contract_validated": True,
                "checkpoint_contract_sha256": hashlib.sha256(json.dumps(
                    checkpoint_contract, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()).hexdigest(),
                "cat_ema_state": "fresh evaluator environment; not part of actor inference weights",
            }
        cat_contract = (validate_baseline_contract(
            contract, target_max_p=args.cat_dose_target_max_p,
        ) if args.cat_dose_audit else None)
        # Loading and runner setup may consume RNG. Reset the RNG immediately before
        # each matched reset, then preserve its post-reset state as separate evidence.
        if adapter is not None:
            adapter.bind(env, runner, checkpoint_contract, contract)
            adapter.__enter__()
            adapter_entered = True
        env.seed(args.seed)
        with BalancedRouteReset(env) if getattr(args, "balanced_routes", False) else nullcontext():
            obs, _ = wrapped.reset()
        if adapter is not None:
            adapter.after_reset(obs)
        reset_hashes = _reset_evidence(env, obs)
        reset_identity = reset_identity_evidence(env, obs)
        rng_hash = _digest(torch.get_rng_state())
        cuda_rng_hash = _digest(*torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        if mode == "external" and goal is not None:
            obs = TensorDict(override_latched_goal(env, goal), batch_size=[env.num_envs])
        elif args.mode == "paired" and goal is not None:
            obs = TensorDict(set_goal_before_first_action(env, goal, obs), batch_size=[env.num_envs])
        observation_physical_prefix_hash = _digest(obs["actor"][:, :40], obs["critic"][:, :40])
        policy = runner.get_inference_policy(device=args.device)
        parity = {"pt": "not requested", "onnx": "not requested", "observations": 0}
        jit = backend = None
        if args.mode == "smoke":
            jit, backend = _parity_models(runner, directory)
            parity["onnx"] = "unavailable" if backend is None else "pending"
        gamma = agent_cfg.algorithm.gamma
        acc = EpisodeAccumulator(env.num_envs, gamma=gamma, coefficient=args.speed_cost)
        cat_dose_trace = CatDoseTrace(env.num_envs) if args.cat_dose_audit else None
        steps = (
            math.ceil(args.shadow_hard_cap_from_reset_s / env.step_dt)
            if args.post_terminal_shadow else
            64 // args.num_envs if args.mode == "smoke" else env.max_episode_length + 1
        )
        if adapter is not None:
            steps += math.ceil(.22 / env.step_dt) + 1
        first_action_hash = second_action = second_kp = None
        # Runner/checkpoint validation above sees the original training termination.
        # The context restores it before the final safety/controller contract check.
        trace = (ContactTrace(
                    env, max_policy_steps=steps,
                    post_onset_horizon_s=(args.shadow_post_onset_horizon_s
                                          if args.post_terminal_shadow else None),
                    hard_cap_s=(args.shadow_hard_cap_from_reset_s
                                if args.post_terminal_shadow else None),
                 )
                 if getattr(args, "trace_contact", False) else None)
        safety_observer = (ContactSafetySummary(env)
                           if getattr(args, "contact_safety_summary", False) else None)
        measurement_protocol = (
            PostTerminalShadowProtocol(env) if args.post_terminal_shadow
            else MeasurementProtocol(env, args.measurement_protocol)
        )
        if adapter is not None:
            measurement_protocol = adapter.protocol(measurement_protocol)
        with (torch.inference_mode(), measurement_protocol,
              PreResetSnapshot(
                  env, reader=lambda source: physical_snapshot(
                      source, include_cat_dose_audit=args.cat_dose_audit,
                  ),
              ) as capture,
              trace if trace is not None else safety_observer if safety_observer is not None else nullcontext()):
            for step in range(steps):
                if args.mode == "intervention" and step == 1:
                    if any(acc.episodes):
                        raise ValueError("cannot intervene after first-step reset")
                    obs = TensorDict(override_latched_goal(env, goal), batch_size=[env.num_envs])
                for group in ("actor", "critic"):
                    width = contract["observation_widths"][group]
                    if obs[group].shape != (env.num_envs, width) or not torch.isfinite(obs[group]).all():
                        raise ValueError(f"expected finite {width}D {group} observations from the runtime contract")
                state = env.action_manager.get_term("strike_speed").state
                pending, previous_goal = state.pending.clone(), state.goal_mps.clone()
                raw = policy(obs)
                if raw.shape != (env.num_envs, 13) or not torch.isfinite(raw).all():
                    raise ValueError("expected finite 13D actor output")
                if step == 0:
                    first_action_hash = _digest(raw)
                elif step == 1:
                    second_action = raw[:, :12].cpu().tolist()
                if jit is not None and step < 3:
                    cpu_obs = obs["actor"].cpu()
                    torch.testing.assert_close(jit(cpu_obs), raw.cpu(), rtol=1e-5, atol=1e-6)
                    parity["pt"] = "passed"
                    if backend is not None:
                        predicted = np.concatenate([backend.run(None, {backend.input_names[0]: row[None].numpy()})[0]
                                                    for row in cpu_obs])
                        torch.testing.assert_close(torch.from_numpy(predicted), raw.cpu(), rtol=1e-5, atol=1e-6)
                        parity["onnx"] = "passed: ReferenceEvaluator"
                    parity["observations"] += 1
                if adapter is not None:
                    adapter.before_step(step)
                obs, _, done, extras = wrapped.step(raw)
                if adapter is not None:
                    adapter.after_step(step, done)
                if trace is not None:
                    trace.finish_step(done, capture.latest_evaluation)
                safety_records = (safety_observer.finish_step(done)
                                  if safety_observer is not None else None)
                outcome = extras["impact_speed_outcome"]
                reward_done = adapter.overlay.reward_done(done) if adapter is not None else done.bool()
                if not torch.equal(outcome["done"], reward_done):
                    raise ValueError("terminal speed evidence disagrees with native done")
                lo, hi = args.speed_bounds
                expected = torch.where(pending.bool(), lo + (raw[:, 12].clamp(-1, 1) + 1) * .5 * (hi - lo), previous_goal)
                torch.testing.assert_close(outcome["goal_mps"], expected)
                torch.testing.assert_close(state.pending, torch.zeros_like(pending) if adapter is not None else done.float() if mode == "policy" else torch.zeros_like(pending))
                for key, value in capture.latest.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"nonfinite physical metric {key}")
                if cat_dose_trace is not None:
                    cat_dose_trace.record_step(
                        capture.latest, done=done,
                        first_episode_mask=torch.tensor(
                            [episode == 0 for episode in acc.episodes], dtype=torch.bool,
                        ),
                    )
                acc.update(outcome, capture.latest, raw, pending, capture.latest_evaluation,
                           contact_safety_records=safety_records, physical_done=done.bool() if adapter is not None else None)
                if step == 1:
                    second_kp = capture.latest["kp"].cpu().tolist()
                if (args.post_terminal_shadow and trace is not None
                        and not bool(trace.active.any())):
                    break
                if args.mode != "smoke" and all(episode >= 1 for episode in acc.episodes):
                    break
        if trace is not None:
            trace.save(directory)
        cat_dose_evidence = None
        if cat_dose_trace is not None:
            payload = cat_dose_trace.to_payload()
            torch.save(payload, directory / "cat-dose-trace.pt")
            cat_dose_evidence = summarize_cat_dose_trace(
                payload, baseline_max_p=cat_contract["baseline_velocity_max_p"],
                target_max_p=cat_contract["offline_counterfactual_velocity_max_p"],
            )
        if adapter is not None:
            adapter.__exit__(None, None, None)
            adapter_entered = False
        if build_impact_speed_metadata(env, wrapped.clip_actions) != contract:
            raise ValueError("live safety/controller contract changed during evaluation")
        rows = acc.finish()
        if args.post_terminal_shadow:
            # Keep only the original first-attempt records; continued post-success control is
            # diagnostic behavior, not a new native episode/task outcome.
            rows = [row for row in rows if row["episode"] == 0]
        lo, hi = args.speed_bounds
        for row in rows:
            row["selected_goal_mps"] = (
                lo + (max(-1., min(1., row["selected_raw"])) + 1) * .5 * (hi - lo)
                if row["selection_count"] and row["selected_raw"] is not None else None
            )
            row.update(condition=index, condition_goal_mps=goal,
                       intervention_goal_mps=goal if args.mode == "intervention" else None,
                       episode_id=f"{row['world']}:{row['episode']}",
                       reset_hash=reset_hashes[row["world"]] if row["episode"] == 0 else None,
                       physical_reset_hash=(reset_identity["physical_reset_hashes"][row["world"]]
                                            if row["episode"] == 0 else None),
                       reset_observation_hashes=(reset_identity["observation_hashes"][row["world"]]
                                                 if row["episode"] == 0 else None),
                       reset_sensor_noise_hash=(reset_identity["sensor_noise_hashes"][row["world"]]
                                                if row["episode"] == 0 else None))
        selector_mode = (
            "paired-fixed" if args.mode == "paired" and goal is not None else mode
        )
        selector_evidence = validate_selector_counts(
            rows, mode=selector_mode, expected_episode_count=env.num_envs
        )
        selector_clipping = summarize_selector_clipping(rows, expected_episode_count=env.num_envs)
        route_event = cfg.events.get("sample_strike_route_signs")
        fixed_sign = (None if route_event is None
                      else route_event.params.get("fixed_route_sign"))
        expected_signs = (
            (adapter.case[1],) if adapter is not None else
            (-1, 0, 1) if fixed_sign is None else (fixed_sign,)
        )
        route_evidence = summarize_route_strata(
            rows, expected_route_signs=expected_signs,
            require_all=args.mode != "smoke", expected_episode_count=env.num_envs,
        )
        timing_evidence = (summarize_contact_safety_timing(
            rows, expected_episode_count=env.num_envs,
        ) if safety_observer is not None else None)
        shadow_evidence = None
        if args.post_terminal_shadow:
            window = trace.shadow_window
            shadow_evidence = {
                "post_onset_horizon_s": args.shadow_post_onset_horizon_s,
                "hard_cap_from_reset_s": args.shadow_hard_cap_from_reset_s,
                "continued_policy": True,
                "suppressed_termination": "nail_driven only",
                "native_success_and_50ms_window_latched": True,
                "worlds_reaching_horizon": int(window.complete.sum()),
                "worlds_censored": int(window.censored.sum()),
                "semantics": (
                    "post-terminal actor continuation; any later depth, contact, or joint-limit result "
                    "is separate from the native first-attempt measurement"
                ),
            }
        if timing_evidence is not None:
            timing_evidence["observer_source_sha256"] = _file_sha256(
                Path(contact_trace_module.__file__),
            )
        first_episode_route_signs = [
            row["route_sign"] for row in sorted(rows, key=lambda row: row["world"])
            if row["episode"] == 0
        ]
        condition_role = (
            "autonomous_policy_selector" if args.mode == "paired" and goal is None else
            "preaction_fixed_target" if args.mode == "paired" else
            "external_target" if mode == "external" else
            "post_first_action_intervention" if args.mode == "intervention" else
            "single_condition"
        )
        return rows, {"goal_mps": goal, "condition_role": condition_role,
                      "contract": contract, "reset_hashes": reset_hashes,
                      "reset_identity": reset_identity,
                      "balanced_routes": bool(getattr(args, "balanced_routes", False)),
                      "observation_physical_prefix_hash": observation_physical_prefix_hash,
                      "first_episode_route_signs": first_episode_route_signs,
                      "route_stratified": route_evidence,
                      "selector_validation": selector_evidence,
                      "selector_clipping": selector_clipping,
                      "contact_trace": ("contact-trace.pt" if trace is not None else None),
                      "cat_dose_trace": ("cat-dose-trace.pt" if cat_dose_evidence is not None else None),
                      "cat_dose_audit": cat_dose_evidence,
                      "cat_dose_contract": cat_contract,
                      "post_terminal_shadow": shadow_evidence,
                      "checkpoint_transfer": checkpoint_transfer,
                      "contact_safety_timing": timing_evidence,
                      "measurement": {"protocol": args.measurement_protocol,
                          "post_terminal_shadow": args.post_terminal_shadow,
                          "all_observed_attempts": summarize_strike_evaluations([r["measurement"] for r in rows]),
                          "success_reset_override": args.measurement_protocol == "common-window",
                          "site_state_offset_physics_samples": -1,
                          "termination_check_cadence_s": env.step_dt,
                          "physics_dt_s": env.physics_dt, "required_window_samples": env._strike_evaluation.window,
                          "window_duration_s": .05, "onset_sample_included": True,
                          "path_bins": 16, "impulse_dose": cfg.metrics["cat_soft"].params["imp_max_p"],
                          "load_signal": LOAD_SIGNAL_SEMANTICS},
                      "torch_post_reset_rng_hash": rng_hash, "cuda_post_reset_rng_hash": cuda_rng_hash,
                      "first_action_hash": first_action_hash, "second_motor_action": second_action,
                      "second_kp": second_kp, "transitions": (step + 1) * env.num_envs,
                      "selection_count": sum(row["selection_count"] for row in rows), "parity": parity}
    finally:
        try:
            if adapter_entered:
                adapter.__exit__(*sys.exc_info())
        finally:
            env.close()


def run(args):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    args.output.mkdir(parents=True, exist_ok=False)
    rows, conditions = [], []
    condition_goals = [None, *args.goals] if args.mode == "paired" else (args.goals or [None])
    for index, goal in enumerate(condition_goals):
        condition_rows, report = _condition(args, goal, index)
        rows.extend(condition_rows)
        conditions.append(report)
    comparisons = []
    if args.goals and not args.post_terminal_shadow:
        baseline = conditions[0]
        for condition in conditions[1:]:
            for key in ("reset_hashes", "first_episode_route_signs",
                        "torch_post_reset_rng_hash", "cuda_post_reset_rng_hash",
                        "observation_physical_prefix_hash", "contract"):
                if condition[key] != baseline[key]:
                    raise ValueError(f"paired state/RNG/contract mismatch: {key}")
            if args.mode == "intervention" and condition["first_action_hash"] != baseline["first_action_hash"]:
                raise ValueError("paired intervention first actions differ")
        def response(index):
            initial = sorted((r for r in rows if r["condition"] == index and r["episode"] == 0), key=lambda r: r["world"])
            return {"episode_id": [r["episode_id"] for r in initial], "reset_hash": [r["reset_hash"] for r in initial],
                    "hit": [r["hit"] for r in initial], "speed": [r["measured_mps"] for r in initial]}
        if args.mode == "paired":
            pair_specs = [(0, target, None, args.goals[target - 1]) for target in (1, 2, 3)]
            pair_specs.extend(
                (low, high, args.goals[low - 1], args.goals[high - 1])
                for low, high in ((1, 2), (1, 3), (2, 3))
            )
        else:
            pair_specs = [(low, high, args.goals[low], args.goals[high])
                          for low, high in ((0, 1), (1, 2), (0, 2))]
        for low, high, low_goal, high_goal in pair_specs:
            contrast = compare_goal_responses(response(low), response(high), args.seed)
            contrast.update(low_condition=low, high_condition=high,
                            low_condition_role=conditions[low]["condition_role"],
                            high_condition_role=conditions[high]["condition_role"],
                            low_goal_mps=low_goal, high_goal_mps=high_goal)
            low_rows = [row for row in rows if row["condition"] == low]
            high_rows = [row for row in rows if row["condition"] == high]
            contrast["outcome_contrast"] = compare_paired_outcomes(
                low_rows, high_rows, args.seed
            )
            if args.mode == "intervention":
                contrast["second_motor_action_rms_difference"] = float(np.sqrt(np.mean((np.array(conditions[high]["second_motor_action"]) - np.array(conditions[low]["second_motor_action"]))**2)))
                contrast["second_kp_rms_difference"] = float(np.sqrt(np.mean((np.array(conditions[high]["second_kp"]) - np.array(conditions[low]["second_kp"]))**2)))
            comparisons.append(contrast)
    checkpoint_hash = None
    if args.checkpoint is not None:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint_hash = digest.hexdigest()
    summary = {"task": args.task, "mode": args.mode, "seed": args.seed,
               "checkpoint": str(args.checkpoint) if args.checkpoint else None, "checkpoint_sha256": checkpoint_hash,
               "actor_status": "checkpoint" if args.checkpoint else "untrained software smoke only",
               "qualification": "not assessed; bounds/cost are supplied experiment inputs, not physical certification",
               "transitions": sum(c["transitions"] for c in conditions), "learning_updates": 0,
               "contact_safety_summary_enabled": bool(args.contact_safety_summary),
               "cat_dose_audit_enabled": bool(args.cat_dose_audit),
               "cat_dose_target_max_p": (args.cat_dose_target_max_p if args.cat_dose_audit else None),
               "post_terminal_shadow_enabled": bool(args.post_terminal_shadow),
               "conditions": conditions, "comparisons": comparisons,
               "initial_state_pairing": (
                   "verified_policy_vs_preaction_fixed_targets" if args.mode == "paired" else
                   "paired_within_external_goal_conditions" if (
                       args.mode == "external" and args.goals and not args.post_terminal_shadow
                   ) else
                   "paired_within_intervention_conditions" if args.mode == "intervention" and args.goals else
                   "not_compared_across_speed_modes; use campaign reset-hash and route-sign audit"
               ),
               "timeout_count": sum(r["timeout"] for r in rows), "completed_episodes": sum(r["complete"] for r in rows),
               "cost_semantics": "positive cost magnitude; gamma-discounted from first transition; timeout is reset-boundary shaping, not a hard deadline",
               "tracking_semantics": "all sampled steps, geometric reference preview, reward-time one-substep-stale positions",
               "reset_evidence_semantics": "first episode joint/nail positions and velocities plus physical observations; fresh same-seed native construction; not a portable full-state replay"}
    with (args.output / "episodes.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    args = parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "transitions": result["transitions"],
                      "completed_episodes": result["completed_episodes"], "learning_updates": 0}))


if __name__ == "__main__":
    main()

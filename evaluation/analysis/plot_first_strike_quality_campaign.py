"""Exactly two static figures for the lean first-contact quality campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from evaluation.analysis.first_strike_quality_campaign import (
    LABELS,
    analyze_quality_campaign,
)


COLORS = {
    "F8": "#4C78A8",
    "F0": "#F58518",
    "D0": "#54A24B",
    "FQ-min": "#E45756",
}


def build_aggregate_nail_plane_contact_map(
    contacts_by_arm: Mapping[str, np.ndarray], *, nail_radius_m: float
):
    """Build four comparable nail-plane contact panels with common limits."""

    if nail_radius_m <= 0.0 or not np.isfinite(nail_radius_m):
        raise ValueError("nail radius must be finite and positive")
    arrays = {}
    extent = nail_radius_m
    for label in LABELS:
        values = np.asarray(contacts_by_arm.get(label, []), dtype=float)
        if values.size == 0:
            values = np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError(f"{label}: contacts must be a finite Nx2 matrix")
        arrays[label] = values
        if values.size:
            extent = max(extent, float(np.max(np.abs(values))))
    limit = max(1.25 * nail_radius_m, 1.1 * extent)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
    for axis, label in zip(axes.flat, LABELS, strict=True):
        values = arrays[label]
        if values.size:
            axis.scatter(
                values[:, 0] * 1e3,
                values[:, 1] * 1e3,
                s=8,
                alpha=0.22,
                color=COLORS[label],
                linewidths=0,
            )
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                nail_radius_m * 1e3,
                fill=False,
                linewidth=1.5,
                color="#333333",
            )
        )
        axis.axhline(0.0, color="#cccccc", linewidth=0.6)
        axis.axvline(0.0, color="#cccccc", linewidth=0.6)
        axis.set(
            title=label,
            xlim=(-limit * 1e3, limit * 1e3),
            ylim=(-limit * 1e3, limit * 1e3),
            xlabel="nail-plane x (mm)",
            ylabel="nail-plane y (mm)",
        )
        axis.set_aspect(1.0)
    figure.suptitle("First-contact locations in the nail plane")
    return figure


def _build_paired_seed_effects(analysis: Mapping):
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
    primary = analysis["primary_contrasts"]
    names = list(primary)
    differences = [
        np.asarray(primary[name]["paired_differences"], dtype=float)
        for name in names
    ]
    labels = ["F0 − F8", "D0 − F8", "FQ-min − D0"]
    axis = axes[0, 0]
    for index, (label, values) in enumerate(zip(labels, differences, strict=True)):
        axis.scatter(
            np.full(values.size, index),
            values,
            color=COLORS[("F0", "D0", "FQ-min")[index]],
            alpha=0.75,
        )
        axis.plot(index, values.mean(), marker="_", markersize=18, color="black")
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set(
        title="Registered quality contrasts",
        ylabel="paired quality difference",
        xticks=range(3),
        xticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=15)

    practical = analysis["fq_min_practical_acceptance"]
    axis = axes[0, 1]
    quality = np.asarray(
        practical["quality_gain"]["paired_differences"], dtype=float
    )
    speed = np.asarray(
        practical["useful_speed_ratio"]["paired_differences"], dtype=float
    )
    axis.scatter(np.zeros(8), quality, color=COLORS["FQ-min"], alpha=0.75)
    axis.scatter(np.ones(8), speed, color=COLORS["F8"], alpha=0.75)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set(
        title="FQ-min practical effects vs F8",
        xticks=(0, 1),
        xticklabels=("quality gain", "speed difference"),
        ylabel="paired difference",
    )

    axis = axes[1, 0]
    first = np.asarray(
        practical["first_window_success"]["paired_differences"], dtype=float
    )
    overall = np.asarray(
        practical["overall_success"]["paired_differences"], dtype=float
    )
    for index, values in enumerate((first, overall)):
        axis.scatter(
            np.full(8, index), values, color=COLORS["FQ-min"], alpha=0.75
        )
    axis.axhline(-0.05, color="#B279A2", linestyle="--", linewidth=1.0)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set(
        title="FQ-min success preservation vs F8",
        xticks=(0, 1),
        xticklabels=("first-window", "overall"),
        ylabel="paired success-rate difference",
    )

    axis = axes[1, 1]
    depth = practical["depth_gain_ratio"]
    speed_ratio = practical["useful_speed_ratio"]
    ratio_records = (speed_ratio, depth)
    lower_plotted = False
    for index, (record, color) in enumerate(
        zip(ratio_records, (COLORS["F8"], COLORS["D0"]), strict=True)
    ):
        estimate = record.get("estimate")
        lower = record.get("one_sided_95_lower")
        available = (
            bool(record.get("valid", True))
            and estimate is not None
            and np.isfinite(float(estimate))
            and lower is not None
            and np.isfinite(float(lower))
        )
        if available:
            axis.bar(index, float(estimate), color=color, alpha=0.8)
            axis.scatter(
                index,
                float(lower),
                marker="v",
                color="black",
                label="95% lower" if not lower_plotted else None,
            )
            lower_plotted = True
        else:
            axis.text(
                index,
                0.5,
                "unavailable",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="center",
                color="#666666",
                fontstyle="italic",
            )
    axis.axhline(0.95, color=COLORS["F8"], linestyle="--", linewidth=1.0)
    axis.axhline(0.90, color=COLORS["D0"], linestyle=":", linewidth=1.0)
    axis.set(
        title="FQ-min preservation ratios vs F8",
        xticks=(0, 1),
        xticklabels=("useful speed", "depth gain"),
        xlim=(-0.5, 1.5),
        ylabel="ratio",
    )
    if lower_plotted:
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Seed-paired first-contact quality campaign effects")
    return figure


def render_quality_figures(
    rows: Sequence[Mapping],
    accepted_evaluation_manifest: Sequence[Mapping],
    *,
    output_dir: str | Path,
) -> dict:
    """Render the two and only two registered static campaign figures."""

    analysis = analyze_quality_campaign(rows, accepted_evaluation_manifest)
    if not analysis["valid"]:
        raise ValueError(
            "cannot render invalid quality campaign: "
            + "; ".join(analysis["invalidation_reasons"])
        )
    contacts = {
        label: np.asarray(
            [
                [coordinate["x_m"], coordinate["y_m"]]
                for coordinate in analysis["valid_contact_coordinates"]
                if coordinate["treatment"] == label
            ],
            dtype=float,
        ).reshape(-1, 2)
        for label in LABELS
    }
    nail_radius = float(analysis["nail_geometry"]["nail_radius_m"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired_path = output / "paired_seed_effects.png"
    contact_path = output / "aggregate_nail_plane_contact_map.png"
    paired = _build_paired_seed_effects(analysis)
    contact = build_aggregate_nail_plane_contact_map(
        contacts,
        nail_radius_m=nail_radius,
    )
    paired.savefig(paired_path, dpi=180, bbox_inches="tight")
    contact.savefig(contact_path, dpi=180, bbox_inches="tight")
    plt.close(paired)
    plt.close(contact)
    return {
        "analysis": analysis,
        "artifacts": [str(paired_path), str(contact_path)],
    }

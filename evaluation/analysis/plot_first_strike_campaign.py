"""Plotly-only figures for the fixed-impedance first-strike campaign."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.analysis.first_strike_campaign import (
    ARM_ORDER,
    ARM_TASKS,
    PRIMARY_METRIC,
    _episode_trace_digest,
    analyze_campaign,
)


ARM_COLORS = {
    "C": "#64748b",
    "D-prime": "#7c3aed",
    "F": "#0f766e",
    "E": "#dc2626",
}


def _trace_arrays(trace: Mapping) -> dict[str, np.ndarray]:
    physical = trace["physical"]
    position = np.asarray(physical["head_position_m"], dtype=float)
    contact = np.asarray(physical["contact"], dtype=bool)
    force = np.asarray(physical["net_axial_force_n"], dtype=float)
    depth = np.asarray(physical["clamped_depth_m"], dtype=float)
    tracker_depth_post = np.asarray(
        physical["tracker_depth_post_integration_m"], dtype=float
    )
    event = trace["event_trace"]
    cumulative_event_impulse = np.asarray(
        event["event_cumulative_impulse_n_s"], dtype=float
    )
    tracker_finalized = np.asarray(event["tracker_finalized"], dtype=bool)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or contact.shape != (position.shape[0],)
        or force.shape != contact.shape
        or depth.shape != contact.shape
        or tracker_depth_post.shape != contact.shape
        or cumulative_event_impulse.shape != contact.shape
        or tracker_finalized.shape != contact.shape
    ):
        raise ValueError("representative trace arrays are shape-inconsistent")
    if (
        not np.isfinite(cumulative_event_impulse).all()
        or np.any(np.diff(cumulative_event_impulse) < -1e-12)
    ):
        raise ValueError("stored tracker event impulse must be finite/nondecreasing")
    finalized = np.flatnonzero(tracker_finalized)
    if finalized.size and not np.allclose(
        cumulative_event_impulse[finalized[0] :],
        cumulative_event_impulse[finalized[0]],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("stored tracker event impulse must freeze at finalization")
    return {
        "position": position,
        "contact": contact,
        "force": force,
        "depth": depth,
        "tracker_depth_post": tracker_depth_post,
        "cumulative_event_impulse": cumulative_event_impulse,
        "tracker_finalized": tracker_finalized,
    }


def _label(trace: Mapping) -> str:
    arm = str(trace["arm"])
    task = str(trace["task"])
    if arm not in ARM_TASKS or task != ARM_TASKS[arm]:
        raise ValueError(f"representative trace has invalid arm/task pair: {arm}/{task}")
    return f"{arm} — {task}"


def _marker_indices(
    trace: Mapping, arrays: Mapping[str, np.ndarray]
) -> tuple[int | None, int | None]:
    onset = trace.get("first_strike", {}).get("accepted_onset_index")
    contact_index = None if onset is None else int(onset)
    if (
        contact_index is not None
        and (
            contact_index < 0
            or contact_index >= arrays["contact"].size
            or not bool(arrays["contact"][contact_index])
        )
    ):
        raise ValueError("accepted onset must index tracker-accepted contact")
    finalized = np.flatnonzero(arrays["tracker_finalized"])
    success_index = (
        int(finalized[0])
        if trace.get("first_strike", {}).get("reason") == "success"
        and finalized.size
        else None
    )
    return contact_index, success_index


def _validate_representative_traces(traces: Sequence[Mapping]) -> None:
    observed = {
        (str(trace.get("arm")), bool(trace.get("overall_success", False)))
        for trace in traces
    }
    expected = {
        (arm, success)
        for arm in ARM_ORDER
        for success in (False, True)
    }
    if not expected <= observed:
        raise ValueError(
            "representative set needs a success and failure representative "
            "for every arm"
        )
    episode_ids = [str(trace.get("episode_id", "")) for trace in traces]
    if not all(episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("representative episode IDs must be non-empty and unique")


def _validated_campaign_representatives(
    rows: Sequence[Mapping], traces: Sequence[Mapping]
) -> dict[str, str]:
    """Recompute trace digests and prove exact membership in validated NPZ rows."""

    requested: dict[tuple[str, str], tuple[str, str]] = {}
    for trace in traces:
        episode_id = str(trace.get("episode_id", ""))
        arm = str(trace.get("arm", ""))
        try:
            recomputed_digest = _episode_trace_digest(trace)
            canonical_trace = json.dumps(
                trace,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except Exception as error:
            raise ValueError(
                f"representative {episode_id!r} cannot be content-verified: {error}"
            ) from error
        if str(trace.get("trace_digest", "")) != recomputed_digest:
            raise ValueError(
                f"representative {episode_id!r} trace digest mismatch"
            )
        requested[(arm, episode_id)] = (recomputed_digest, canonical_trace)

    matched: dict[str, str] = {}
    for row in rows:
        path = Path(str(row.get("sampled_trace_path", "")))
        with np.load(path, allow_pickle=False) as saved:
            payload = json.loads(str(saved["payload_json"]))
        for candidate in payload.get("episodes", []):
            key = (
                str(candidate.get("arm", "")),
                str(candidate.get("episode_id", "")),
            )
            if key not in requested:
                continue
            expected_digest, expected_trace = requested[key]
            candidate_digest = _episode_trace_digest(candidate)
            if (
                str(candidate.get("trace_digest", "")) == candidate_digest
                and candidate_digest == expected_digest
                and json.dumps(
                    candidate,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                == expected_trace
            ):
                matched[key[1]] = expected_digest

    missing = sorted(
        episode_id
        for (_, episode_id), _ in requested.items()
        if episode_id not in matched
    )
    if missing:
        raise ValueError(
            "representative traces are not exact members of validated campaign "
            f"NPZ artifacts: {', '.join(missing)}"
        )
    return matched


def _video_timing_mapping(
    trace: Mapping,
    video_artifact: Mapping,
    *,
    sample_count: int,
) -> dict:
    """Return an explicit video-frame to 500 Hz trace-index mapping."""

    timing = video_artifact.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("video artifact requires explicit render timing metadata")
    dt_s = float(trace.get("physics_dt_s", np.nan))
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("trace physics_dt_s must be positive and finite")
    max_trace_time_s = (sample_count - 1) * dt_s

    if "frame_timestamps_s" in timing:
        source_times = np.asarray(timing["frame_timestamps_s"], dtype=float)
        if (
            source_times.ndim != 1
            or source_times.size == 0
            or not np.isfinite(source_times).all()
            or np.any(np.diff(source_times) <= 0)
        ):
            raise ValueError(
                "timing.frame_timestamps_s must be a non-empty, finite, "
                "strictly increasing vector"
            )
        video_times = source_times - source_times[0]
        timing_mode = "frame_timestamps_s"
    else:
        try:
            start_time_s = float(timing["start_time_s"])
            fps = float(timing["fps"])
            frame_count = int(timing["frame_count"])
            trim = timing["trim"]
            trim_start = int(trim["start_frame"])
            trim_end = int(trim["end_frame_exclusive"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "timing requires frame_timestamps_s or "
                "start_time_s/fps/frame_count/trim"
            ) from error
        if (
            not np.isfinite(start_time_s)
            or start_time_s < 0
            or not np.isfinite(fps)
            or fps <= 0
            or frame_count <= 0
            or trim_start < 0
            or trim_end <= trim_start
            or trim_end > frame_count
        ):
            raise ValueError("video timing frame-count/trim mapping is invalid")
        source_frame_indices = np.arange(trim_start, trim_end, dtype=float)
        source_times = start_time_s + source_frame_indices / fps
        video_times = np.arange(source_frame_indices.size, dtype=float) / fps
        timing_mode = "fps_trim"

    tolerance = max(1e-12, dt_s * 1e-6)
    if (
        np.any(source_times < -tolerance)
        or np.any(source_times > max_trace_time_s + tolerance)
    ):
        raise ValueError("video timing maps outside the source trace")
    trace_indices = np.rint(source_times / dt_s).astype(int)
    trace_indices = np.clip(trace_indices, 0, sample_count - 1)
    if np.any(np.diff(trace_indices) < 0):
        raise ValueError("video timing must map monotonically through the trace")
    return {
        "mode": timing_mode,
        "video_frame_timestamps_s": video_times.tolist(),
        "source_trace_timestamps_s": source_times.tolist(),
        "frame_trace_indices": trace_indices.tolist(),
    }


def _add_2d_state_path(
    figure: go.Figure,
    *,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    marker: Mapping,
    label: str,
    arm: str,
    hover: str,
    row: int,
    col: int,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": color},
            marker=marker,
            name=label,
            legendgroup=arm,
            showlegend=False,
            hovertemplate=hover,
        ),
        row=row,
        col=col,
    )


def build_trajectory_figure(traces: Sequence[Mapping]) -> go.Figure:
    """Build state-ground-truth path and contact-mechanism panels."""

    traces = list(traces)
    if not traces:
        raise ValueError("at least one representative trace is required")
    _validate_representative_traces(traces)
    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scene"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D state path",
            "x-z side path",
            "x-y top path",
            "Contact-aligned axial force",
            "Depth and cumulative event impulse",
            "Actual reward-manager payout",
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
    )
    geometry = traces[0].get("nail_geometry")
    if geometry is None:
        raise ValueError("representative traces must carry frozen nail geometry")
    geometry_key = (
        tuple(float(value) for value in geometry["nail_xy_m"]),
        float(geometry["nail_radius_m"]),
        str(geometry["source_sha256"]),
    )
    for trace in traces[1:]:
        candidate = trace.get("nail_geometry")
        if candidate is None or (
            tuple(float(value) for value in candidate["nail_xy_m"]),
            float(candidate["nail_radius_m"]),
            str(candidate["source_sha256"]),
        ) != geometry_key:
            raise ValueError("representative traces do not share frozen nail geometry")
    nail_x, nail_y = geometry_key[0]
    nail_radius = geometry_key[1]
    all_z = np.concatenate([_trace_arrays(trace)["position"][:, 2] for trace in traces])
    z_min, z_max = float(np.min(all_z)), float(np.max(all_z))
    theta = np.linspace(0.0, 2.0 * np.pi, 101)
    geometry_label = f"Frozen nail axis/radius — {geometry_key[2]}"
    figure.add_trace(
        go.Scatter3d(
            x=[nail_x, nail_x],
            y=[nail_y, nail_y],
            z=[z_min, z_max],
            mode="lines",
            line={"color": "#111827", "width": 5, "dash": "dash"},
            name=geometry_label,
            hovertemplate=geometry_label + "<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[nail_x, nail_x],
            y=[z_min, z_max],
            mode="lines",
            line={"color": "#111827", "dash": "dash"},
            name=geometry_label,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=nail_x + nail_radius * np.cos(theta),
            y=nail_y + nail_radius * np.sin(theta),
            mode="lines",
            line={"color": "#111827", "dash": "dash"},
            name=f"Frozen nail-head radius ({nail_radius * 1000:.0f} mm)",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    for trace in traces:
        arrays = _trace_arrays(trace)
        label = _label(trace)
        arm = str(trace["arm"])
        color = ARM_COLORS[arm]
        dt_s = float(trace.get("physics_dt_s", 0.002))
        n = arrays["position"].shape[0]
        normalized_time = np.linspace(0.0, 1.0, n)
        contact_index, success_index = _marker_indices(trace, arrays)
        alignment_index = contact_index if contact_index is not None else n - 1
        aligned_ms = (np.arange(n) - alignment_index) * dt_s * 1000.0
        cumulative_impulse = arrays["cumulative_event_impulse"]
        hover = (
            f"{label}<br>trace {trace['trace_digest']}"
            "<br>normalized time %{marker.color:.3f}<extra></extra>"
        )
        common_marker = {
            "size": 4,
            "color": normalized_time,
            "colorscale": "Viridis",
            "showscale": False,
        }
        figure.add_trace(
            go.Scatter3d(
                x=arrays["position"][:, 0],
                y=arrays["position"][:, 1],
                z=arrays["position"][:, 2],
                mode="lines+markers",
                line={"color": color, "width": 4},
                marker=common_marker,
                name=label,
                legendgroup=arm,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
        _add_2d_state_path(
            figure,
            x=arrays["position"][:, 0],
            y=arrays["position"][:, 2],
            color=color,
            marker=common_marker,
            label=label,
            arm=arm,
            hover=hover,
            row=1,
            col=2,
        )
        _add_2d_state_path(
            figure,
            x=arrays["position"][:, 0],
            y=arrays["position"][:, 1],
            color=color,
            marker=common_marker,
            label=label,
            arm=arm,
            hover=hover,
            row=1,
            col=3,
        )
        for index, marker_name, symbol in (
            (contact_index, "first contact", "x"),
            (
                success_index,
                "tracker success (post-depth transition)",
                "star",
            ),
        ):
            if index is None:
                continue
            point = arrays["position"][index]
            figure.add_trace(
                go.Scatter(
                    x=[point[0]],
                    y=[point[1]],
                    mode="markers",
                    marker={"symbol": symbol, "size": 11, "color": color},
                    name=f"{label} {marker_name}",
                    legendgroup=arm,
                    showlegend=False,
                    hovertemplate=(
                        f"{label}<br>{marker_name}<br>trace "
                        f"{trace['trace_digest']}<extra></extra>"
                    ),
                ),
                row=1,
                col=3,
            )
        figure.add_trace(
            go.Scatter(
                x=aligned_ms,
                y=arrays["force"],
                mode="lines",
                line={"color": color},
                name=label,
                legendgroup=arm,
                showlegend=False,
                hovertemplate=(
                    f"{label}<br>%{{x:.1f}} ms<br>%{{y:.3g}} N"
                    f"<br>trace {trace['trace_digest']}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=aligned_ms,
                y=arrays["depth"] * 1000.0,
                mode="lines",
                line={"color": color},
                name=f"{label} pre-integration depth",
                legendgroup=arm,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=aligned_ms,
                y=arrays["tracker_depth_post"] * 1000.0,
                mode="lines",
                line={"color": color, "dash": "dash"},
                name=f"{label} tracker post-integration depth",
                legendgroup=arm,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=aligned_ms,
                y=cumulative_impulse,
                mode="lines",
                line={"color": color, "dash": "dot"},
                name=f"{label} cumulative event impulse",
                legendgroup=arm,
                showlegend=False,
                yaxis="y6",
            ),
            row=2,
            col=2,
        )
        reward = trace.get("reward", {})
        impact = np.asarray(reward.get("impact_payout", ()), dtype=float)
        delivered = np.asarray(reward.get("delivered_payout", ()), dtype=float)
        control_n = max(impact.size, delivered.size)
        if control_n:
            control_ms = np.arange(control_n) * 20.0
            if impact.size < control_n:
                impact = np.pad(impact, (0, control_n - impact.size))
            if delivered.size < control_n:
                delivered = np.pad(delivered, (0, control_n - delivered.size))
            figure.add_trace(
                go.Scatter(
                    x=control_ms,
                    y=impact,
                    mode="lines+markers",
                    line={"color": color},
                    name=f"{label} impact payout",
                    legendgroup=arm,
                    showlegend=False,
                ),
                row=2,
                col=3,
            )
            figure.add_trace(
                go.Scatter(
                    x=control_ms,
                    y=delivered,
                    mode="lines+markers",
                    line={"color": color, "dash": "dot"},
                    name=f"{label} delivered payout",
                    legendgroup=arm,
                    showlegend=False,
                ),
                row=2,
                col=3,
            )

    figure.update_xaxes(title_text="x (m)", row=1, col=2)
    figure.update_yaxes(title_text="z (m)", row=1, col=2)
    figure.update_xaxes(title_text="x (m)", row=1, col=3)
    figure.update_yaxes(title_text="y (m)", scaleanchor="x3", row=1, col=3)
    figure.update_xaxes(title_text="time from contact (ms)", row=2, col=1)
    figure.update_yaxes(title_text="axial force (N)", row=2, col=1)
    figure.update_xaxes(title_text="time from contact (ms)", row=2, col=2)
    figure.update_yaxes(title_text="depth (mm) / impulse (N·s)", row=2, col=2)
    figure.update_xaxes(title_text="control time (ms)", row=2, col=3)
    figure.update_yaxes(title_text="weighted payout", row=2, col=3)
    figure.update_layout(
        title={
            "text": (
                "First-strike representative traces<br>"
                "<sup>Simulator state is ground truth; colors encode normalized time. "
                "Reward streams are actual task payouts, not duplicated physical traces.</sup>"
            )
        },
        height=900,
        width=1500,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.11},
    )
    return figure


def build_campaign_figure(rows: Sequence[Mapping]) -> go.Figure:
    """Plot the seed-level primary and key sampled mechanism diagnostics."""

    rows = list(rows)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Seed-level useful first-strike speed",
            "Sampled contact structure",
            "Discounted maximize components",
        ),
    )
    for arm_index, arm in enumerate(ARM_ORDER):
        arm_rows = sorted(
            (row for row in rows if row["treatment"] == arm),
            key=lambda row: int(row["training_seed"]),
        )
        x = [arm_index + (int(row["training_seed"]) - 3.5) * 0.025 for row in arm_rows]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[float(row[PRIMARY_METRIC]) for row in arm_rows],
                mode="markers",
                marker={"size": 10, "color": ARM_COLORS[arm]},
                name=f"{arm} — {ARM_TASKS[arm]}",
                legendgroup=arm,
                customdata=[int(row["training_seed"]) for row in arm_rows],
                hovertemplate=(
                    f"{arm}<br>training seed %{{customdata}}"
                    "<br>useful speed %{y:.3f} m/s<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=[arm],
                y=[
                    float(
                        np.mean(
                            [
                                row["first_strike_success_rate_sampled"]
                                for row in arm_rows
                            ]
                        )
                    )
                ],
                marker_color=ARM_COLORS[arm],
                name=f"{arm} first-window success",
                legendgroup=arm,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        impact = float(
            np.mean(
                [row["impact_return_discounted_mean_sampled"] for row in arm_rows]
            )
        )
        delivered = float(
            np.mean(
                [row["delivered_return_discounted_mean_sampled"] for row in arm_rows]
            )
        )
        figure.add_trace(
            go.Bar(
                x=[arm],
                y=[impact],
                marker_color=ARM_COLORS[arm],
                name=f"{arm} impact",
                legendgroup=arm,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        figure.add_trace(
            go.Bar(
                x=[arm],
                y=[delivered],
                marker_color=ARM_COLORS[arm],
                marker_pattern_shape="/",
                name=f"{arm} delivered",
                legendgroup=arm,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    figure.update_xaxes(
        tickmode="array", tickvals=list(range(4)), ticktext=list(ARM_ORDER), row=1, col=1
    )
    figure.update_yaxes(title_text="Y = first-window success × v_precontact", row=1, col=1)
    figure.update_yaxes(title_text="rate", range=[0, 1.05], row=1, col=2)
    figure.update_yaxes(title_text="discounted weighted return", row=1, col=3)
    figure.update_layout(
        barmode="stack",
        title=(
            "Fixed-impedance first-strike campaign<br>"
            "<sup>Diamonds/points are independent training seeds; reset episodes remain nested.</sup>"
        ),
        template="plotly_white",
        height=600,
        width=1450,
    )
    return figure


def write_video_overlay_html(
    trace: Mapping,
    *,
    video_artifact: Mapping,
    output_path: str | Path,
) -> Path:
    """Overlay synchronized state path and event markers directly on a video."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    label = _label(trace)
    episode_id = str(trace.get("episode_id", ""))
    digest = _episode_trace_digest(trace)
    if str(trace.get("trace_digest", "")) != digest:
        raise ValueError("source state trace digest does not match recomputed trace")
    if str(video_artifact.get("episode_id", "")) != episode_id:
        raise ValueError("video episode ID does not match source state trace")
    if str(video_artifact.get("trace_digest", "")) != digest:
        raise ValueError("video trace digest does not match source state trace")
    video_path = Path(str(video_artifact.get("path", "")))
    if not video_path.is_file():
        raise ValueError("video artifact does not exist")
    video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()
    if str(video_artifact.get("video_sha256", "")) != video_sha256:
        raise ValueError("video artifact SHA-256 does not match its bytes")
    arrays = _trace_arrays(trace)
    contact_index, success_index = _marker_indices(trace, arrays)
    nail_xy = trace["nail_geometry"]["nail_xy_m"]
    timing_mapping = _video_timing_mapping(
        trace,
        video_artifact,
        sample_count=arrays["position"].shape[0],
    )
    overlay_json = json.dumps(
        {
            "episode_id": episode_id,
            "trace_digest": digest,
            "label": label,
            "x": arrays["position"][:, 0].tolist(),
            "z": arrays["position"][:, 2].tolist(),
            "nail_x": float(nail_xy[0]),
            "contact_index": contact_index,
            "success_index": success_index,
            **timing_mapping,
        },
        sort_keys=True,
        allow_nan=False,
    ).replace("</", "<\\/")
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(label)} state overlay</title>
  <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
</head>
<body data-trace-digest="{html.escape(digest)}">
  <h1>{html.escape(label)}</h1>
  <p><strong>State ground truth</strong> is synchronized by explicit render timing as a
     path/event overlay directly on the rendered video. Source trace digest:
     <code>{html.escape(digest)}</code>; video SHA-256:
     <code>{video_sha256}</code>. No image tracker is used.</p>
  <div style="position: relative; display: inline-block; max-width: 100%;">
    <video id="rendered-video" controls preload="metadata"
           style="display: block; max-width: 100%;"
           src="{html.escape(str(video_path))}"></video>
    <div id="state-overlay"
         style="position: absolute; top: 6%; right: 4%; width: 34%; height: 40%;
                background: transparent; pointer-events: none;"></div>
  </div>
  <script>
  const overlay = {overlay_json};
  const video = document.getElementById("rendered-video");
  const stateOverlay = document.getElementById("state-overlay");
  const minX = Math.min(...overlay.x, overlay.nail_x);
  const maxX = Math.max(...overlay.x, overlay.nail_x);
  const minZ = Math.min(...overlay.z);
  const maxZ = Math.max(...overlay.z);
  const xPad = Math.max(1e-6, (maxX - minX) * 0.08);
  const zPad = Math.max(1e-6, (maxZ - minZ) * 0.08);
  const layout = {{
    title: {{
      text: `${{overlay.label}}<br><sup>state x-z path • explicit render timing</sup>`,
      font: {{color: "#f8fafc", size: 12}},
    }},
    margin: {{l: 36, r: 8, t: 42, b: 30}},
    paper_bgcolor: "rgba(0, 0, 0, 0)",
    plot_bgcolor: "rgba(0, 0, 0, 0)",
    font: {{color: "#f8fafc", size: 9}},
    showlegend: true,
    legend: {{
      orientation: "h",
      x: 0,
      y: 1.02,
      font: {{color: "#f8fafc", size: 8}},
      bgcolor: "rgba(0, 0, 0, 0)",
    }},
    xaxis: {{
      title: "x (m)",
      range: [minX - xPad, maxX + xPad],
      gridcolor: "rgba(148, 163, 184, 0.25)",
      zeroline: false,
    }},
    yaxis: {{
      title: "z (m)",
      range: [minZ - zPad, maxZ + zPad],
      gridcolor: "rgba(148, 163, 184, 0.25)",
      zeroline: false,
    }},
  }};
  const config = {{displayModeBar: false, responsive: true}};
  function plotlyTraces(current) {{
    const contactVisible = overlay.contact_index !== null
      && overlay.contact_index <= current;
    const successVisible = overlay.success_index !== null
      && overlay.success_index <= current;
    return [
      {{
        type: "scatter",
        mode: "lines",
        x: [overlay.nail_x, overlay.nail_x],
        y: [minZ - zPad, maxZ + zPad],
        line: {{color: "#94a3b8", width: 2, dash: "dash"}},
        name: "nail axis",
        hoverinfo: "skip",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: overlay.x.slice(0, current + 1),
        y: overlay.z.slice(0, current + 1),
        line: {{color: "#22d3ee", width: 3}},
        marker: {{color: "#22d3ee", size: 4}},
        name: "state path",
        customdata: Array(current + 1).fill(overlay.trace_digest),
        hovertemplate: (
          `${{overlay.label}}<br>x=%{{x:.4f}} m<br>z=%{{y:.4f}} m`
          + `<br>trace %{{customdata}}<extra></extra>`
        ),
      }},
      {{
        type: "scatter",
        mode: "markers",
        x: contactVisible ? [overlay.x[overlay.contact_index]] : [],
        y: contactVisible ? [overlay.z[overlay.contact_index]] : [],
        marker: {{color: "#f59e0b", size: 10}},
        name: "tracker-accepted contact",
        hovertemplate: "tracker-accepted contact<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "markers",
        x: successVisible ? [overlay.x[overlay.success_index]] : [],
        y: successVisible ? [overlay.z[overlay.success_index]] : [],
        marker: {{color: "#22c55e", size: 12, symbol: "diamond"}},
        name: "tracker success (post-depth transition)",
        hovertemplate: "tracker success (post-depth transition)<extra></extra>",
      }},
    ];
  }}
  function currentOverlayIndex() {{
    const frameTimes = overlay.video_frame_timestamps_s;
    if (!frameTimes.length) return 0;
    const currentTime = Number.isFinite(video.currentTime)
      ? Math.max(0, video.currentTime) : 0;
    let low = 0;
    let high = frameTimes.length;
    while (low < high) {{
      const middle = Math.floor((low + high) / 2);
      if (frameTimes[middle] <= currentTime) low = middle + 1;
      else high = middle;
    }}
    const frameIndex = Math.max(0, Math.min(frameTimes.length - 1, low - 1));
    return overlay.frame_trace_indices[frameIndex];
  }}
  function syncPlotlyOverlay() {{
    Plotly.react(
      stateOverlay,
      plotlyTraces(currentOverlayIndex()),
      layout,
      config,
    );
  }}
  Plotly.newPlot(stateOverlay, plotlyTraces(0), layout, config);
  video.addEventListener("timeupdate", syncPlotlyOverlay);
  video.addEventListener("loadedmetadata", syncPlotlyOverlay);
  window.addEventListener("resize", () => Plotly.Plots.resize(stateOverlay));
  </script>
</body>
</html>
"""
    output.write_text(document)
    return output


def generate_report(
    rows: Sequence[Mapping],
    representative_traces: Sequence[Mapping],
    *,
    output_dir: str | Path,
    video_artifacts: Mapping[str, Mapping] | None = None,
    bootstrap_samples: int = 100_000,
) -> dict:
    """Persist Plotly HTML figures, digest-bound overlays, and analysis JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze_campaign(rows, bootstrap_samples=bootstrap_samples)
    if not bool(analysis.get("valid", False)):
        raise ValueError(
            "cannot generate representative artifacts from an invalid campaign"
        )
    validated_digests = _validated_campaign_representatives(
        rows, representative_traces
    )
    campaign_path = output / "first_strike_campaign.html"
    trajectory_path = output / "first_strike_trajectories.html"
    build_campaign_figure(rows).write_html(campaign_path, include_plotlyjs="cdn")
    build_trajectory_figure(representative_traces).write_html(
        trajectory_path, include_plotlyjs="cdn"
    )
    overlays = []
    if video_artifacts is not None:
        expected_episode_ids = {
            str(trace["episode_id"]) for trace in representative_traces
        }
        if set(video_artifacts) != expected_episode_ids:
            raise ValueError(
                "video artifacts must cover every representative episode exactly"
            )
        for trace in representative_traces:
            episode_id = str(trace["episode_id"])
            path = output / f"{episode_id}_state_overlay.html"
            write_video_overlay_html(
                trace,
                video_artifact=video_artifacts[episode_id],
                output_path=path,
            )
            video_path = Path(str(video_artifacts[episode_id]["path"]))
            overlays.append(
                {
                    "episode_id": episode_id,
                    "trace_digest": validated_digests[episode_id],
                    "html": str(path),
                    "video": str(video_path),
                    "video_sha256": hashlib.sha256(
                        video_path.read_bytes()
                    ).hexdigest(),
                    "timing": dict(video_artifacts[episode_id]["timing"]),
                }
            )
    result = {
        "analysis": analysis,
        "artifacts": {
            "campaign_html": str(campaign_path),
            "trajectory_html": str(trajectory_path),
            "video_overlays": overlays,
        },
        "state_is_ground_truth": True,
        "computer_vision_tracking": False,
    }
    (output / "first_strike_campaign_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result

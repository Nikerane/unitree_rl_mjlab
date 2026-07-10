"""Cross-run training-curve comparison for the Z1 hammer arms (Task 6 observability).

Reads TB event files with ``tensorboard.backend.event_processing.event_accumulator`` and plots
per-arm mean +/- std bands (across seeds) for six panels: mean reward; the success signal
(``Episode_Termination/nail_driven``); ``Episode_Metrics/substep_delivered`` (object-side delivered
impulse, episode-cumulative); the worst-joint peak Λ_j = max over ``Episode_Metrics/imp_peak_joint1..6``
(with J_limit cap lines if ``--j-limit`` is given); ``Episode_Metrics/cat_delta_peak`` (peak binding
pressure -- note ``Episode_Metrics/cat_soft`` is the diluted EPISODE-MEAN δ, not comparable); and
episode length.

Not every arm carries every key -- pre-impulse arms (``c3_track``, ``c3_catsoft``) predate
``cat_impulse`` and have no ``imp_peak_joint*`` / ``cat_delta_peak`` scalars at all. EVERY key read
is guarded with ``key in ea.Tags()["scalars"]``: a missing key means that arm is simply omitted from
the affected panel (and blank in summary.csv), never a crash.

Grouping: multiple seeds of the SAME arm share a run-name after stripping the leading timestamp
(the standard rsl_rl log-dir prefix ``<timestamp>_<run_name>``) and the trailing ``_seedN`` suffix
(the ``train_array.sbatch --agent.run-name ${RUN}_seed${SEED}`` contract). Runs with no run-name
(bare-timestamp dirs, e.g. pre-array manual runs) keep their full basename as a distinct label so
they do not spuriously merge into one arm.

Usage:
  python scripts/compare_runs.py --runs 'logs/rsl_rl/z1_hammer/*c3_*' --out /tmp/compare_runs \
      --j-limit 1.640,3.280,1.640,1.640,1.640,1.640
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")
_SEED_RE = re.compile(r"_seed\d+$")

_IMP_JOINT_KEYS = tuple(f"Episode_Metrics/imp_peak_joint{k}" for k in range(1, 7))

Series = tuple[np.ndarray, np.ndarray]  # (steps, values)


def arm_label(run_dir: Path) -> str:
  """Basename minus the leading timestamp and trailing _seedN (train_array.sbatch's contract)."""
  name = run_dir.name
  stripped = _SEED_RE.sub("", _TS_RE.sub("", name))
  return stripped if stripped else name  # bare-timestamp legacy dirs: keep the full name


def _resolve_runs(patterns: list[str]) -> list[Path]:
  dirs: list[Path] = []
  for pat in patterns:
    matches = sorted(glob.glob(pat))
    if not matches and Path(pat).is_dir():
      matches = [pat]
    dirs.extend(Path(m) for m in matches if Path(m).is_dir())
  seen: set[Path] = set()
  out: list[Path] = []
  for d in dirs:
    if d not in seen:
      seen.add(d)
      out.append(d)
  return out


def _load(run_dir: Path) -> EventAccumulator:
  ea = EventAccumulator(str(run_dir))
  ea.Reload()
  return ea


def _series(ea: EventAccumulator, key: str) -> Series | None:
  """(steps, values) for `key`, or None if this run never logged it. NEVER assume presence."""
  if key not in ea.Tags()["scalars"]:
    return None
  events = ea.Scalars(key)
  if not events:
    return None
  steps = np.array([e.step for e in events], dtype=np.float64)
  values = np.array([e.value for e in events], dtype=np.float64)
  return steps, values


def _worst_joint_series(ea: EventAccumulator) -> Series | None:
  """max over Episode_Metrics/imp_peak_joint1..6 -- None if this run has none of them (pre-impulse arms)."""
  present = [s for s in (_series(ea, k) for k in _IMP_JOINT_KEYS) if s is not None]
  if not present:
    return None
  n = min(s[0].shape[0] for s in present)
  steps = present[0][0][:n]
  values = np.stack([s[1][:n] for s in present], axis=0).max(axis=0)
  return steps, values


_PANELS: list[tuple[str, str, str, object]] = [
  ("Mean reward", "reward", "reward", lambda ea: _series(ea, "Train/mean_reward")),
  (
    "Success signal (Episode_Termination/nail_driven)", "count/iter", "success",
    lambda ea: _series(ea, "Episode_Termination/nail_driven"),
  ),
  (
    "Delivered impulse (episode-cumulative)", "N·s", "substep_delivered",
    lambda ea: _series(ea, "Episode_Metrics/substep_delivered"),
  ),
  ("Worst-joint peak Λ_j = max(imp_peak_joint1..6)", "N·m·s", "worst_joint_peak", _worst_joint_series),
  (
    "Peak δ (cat_delta_peak) -- cat_soft is the diluted episode-MEAN", "δ", "cat_delta_peak",
    lambda ea: _series(ea, "Episode_Metrics/cat_delta_peak"),
  ),
  ("Episode length", "steps", "episode_length", lambda ea: _series(ea, "Train/mean_episode_length")),
]


def _interp_onto_grid(series_list: list[Series], grid: np.ndarray) -> np.ndarray:
  # No left/right override: numpy's default clamps to the endpoint value outside a run's own
  # step range, so a run whose logging starts a few iterations late doesn't punch a NaN hole.
  return np.stack([np.interp(grid, steps, values) for steps, values in series_list], axis=0)


def _plot_panel(
  ax, arms: dict[str, list[Path]], eas: dict[Path, EventAccumulator], key_fn, title: str, ylabel: str,
  j_limit: list[float] | None = None,
) -> bool:
  any_plotted = False
  for label, run_dirs in arms.items():
    series_list = [s for s in (key_fn(eas[d]) for d in run_dirs) if s is not None]
    if not series_list:
      continue  # this arm never logged this key -- omit, do not crash (e.g. pre-impulse arms)
    max_common = min(float(s[0].max()) for s in series_list)
    if not np.isfinite(max_common) or max_common <= 0:
      continue
    grid = np.linspace(0.0, max_common, 100)
    stacked = _interp_onto_grid(series_list, grid)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    ax.plot(grid, mean, label=f"{label} (n={len(series_list)})")
    ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    any_plotted = True
  if j_limit is not None:
    for k, lim in enumerate(j_limit):
      ax.axhline(lim, linestyle="--", color=f"C{k}", alpha=0.5, linewidth=1)
  ax.set_title(title, fontsize=9)
  ax.set_ylabel(ylabel, fontsize=8)
  ax.set_xlabel("iteration", fontsize=8)
  if any_plotted:
    ax.legend(fontsize=6)
  else:
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
  return any_plotted


def _last_frac_mean(series: Series | None, frac: float) -> float | None:
  if series is None:
    return None
  _, values = series
  if values.size == 0:
    return None
  n = max(1, int(np.ceil(values.size * frac)))
  return float(np.mean(values[-n:]))


def _write_summary_csv(
  run_dirs: list[Path], eas: dict[Path, EventAccumulator], last_frac: float, out_path: Path
) -> None:
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_dir", "arm"] + [p[2] for p in _PANELS])
    for d in run_dirs:
      ea = eas[d]
      row = [str(d), arm_label(d)]
      for _, _, _, key_fn in _PANELS:
        m = _last_frac_mean(key_fn(ea), last_frac)
        row.append("" if m is None else f"{m:.6g}")
      w.writerow(row)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", nargs="+", required=True, help="glob(s) and/or explicit run dirs")
  ap.add_argument("--out", default="/tmp/compare_runs")
  ap.add_argument("--last-frac", type=float, default=0.2, help="fraction of each run's tail to average for summary.csv")
  ap.add_argument("--j-limit", default=None, help="CSV of 6 floats (N·m·s); draws cap lines on the worst-joint panel")
  args = ap.parse_args()

  j_limit: list[float] | None = None
  if args.j_limit:
    j_limit = [float(x) for x in args.j_limit.split(",")]
    if len(j_limit) != 6:
      raise SystemExit(f"--j-limit must be exactly 6 comma-separated floats, got {len(j_limit)}: {args.j_limit}")

  run_dirs = _resolve_runs(args.runs)
  if not run_dirs:
    raise SystemExit(f"No run directories matched {args.runs!r}")

  arms: dict[str, list[Path]] = {}
  for d in run_dirs:
    arms.setdefault(arm_label(d), []).append(d)

  eas = {d: _load(d) for d in run_dirs}

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(2, 3, figsize=(16, 9))
  for ax, (title, ylabel, _, key_fn) in zip(axes.flat, _PANELS):
    kwargs = {"j_limit": j_limit} if key_fn is _worst_joint_series else {}
    _plot_panel(ax, arms, eas, key_fn, title, ylabel, **kwargs)
  fig.suptitle(f"{len(run_dirs)} runs, {len(arms)} arms: " + ", ".join(sorted(arms)), fontsize=8)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.savefig(out_dir / "compare.png", dpi=150)
  plt.close(fig)

  _write_summary_csv(run_dirs, eas, args.last_frac, out_dir / "summary.csv")
  print(f"[compare_runs] {len(run_dirs)} runs, {len(arms)} arms -> "
        f"{out_dir / 'compare.png'}, {out_dir / 'summary.csv'}")
  for label, dirs in sorted(arms.items()):
    print(f"  arm={label!r} n_seeds={len(dirs)}")


if __name__ == "__main__":
  main()

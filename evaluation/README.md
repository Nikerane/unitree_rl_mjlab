# `evaluation/` — policy evaluation & analysis scripts

Reusable scripts that roll out trained policies and analyse/visualise their behaviour. Run everything
from the **repo root** with `PYTHONPATH=.` and the `unitree_mjlab` conda env, e.g.
`PYTHONPATH=. python evaluation/trajectory/plot_grid.py dc`.

- **Scripts** live here (`evaluation/trajectory/`).
- **Data** (the JSON the trace scripts produce) lives in `evaluation/data/` — durable + committed, so
  figures regenerate without re-running the (slow) rollouts.
- **Figures** are written to `docs/results/assets/2026-07-20_policy_videos/` (where the presentation &
  the maximization-ablation doc reference them).

## `evaluation/trajectory/` — the trajectory analysis pipeline

Two stages: **trace** (checkpoint → JSON of per-substep head path + metrics), then **plot/eval**
(JSON → figure). All trace from an identical fixed reset (seed 12345) unless noted.

| script | what it does | reads | writes |
|---|---|---|---|
| `trace_head_trajectories.py` | trace the July policies (af1, nf1, …) | checkpoints | `data/traj_all.json` |
| `trace_mx.py` | trace the mx ablation (maxoff/maxon/maxmax/maxofftrk) | checkpoints | `data/traj_mx.json` |
| `trace_dc.py` | trace the decomposition (maxoff/imponly/delonly/maxmax) | checkpoints | `data/traj_dc.json` |
| `trace_lg.py` | trace 500-vs-1500-iter arms | checkpoints | `data/traj_lg.json` |
| `eval_mx_multireset.py` / `eval_dc_multireset.py` / `eval_lg_multireset.py` | **robustness**: N randomized resets (`play=False`), per-rollout swing/v_touch/delivered/dwell/peak-force/events/success | checkpoints | `data/*_multireset.json` |
| `check_deep.py` | per-joint Λ load + swing-causality (action replay) | checkpoints | stdout |
| `plot_head_trajectories.py` | the July x-z small-multiples grid (`trajectories_grid.png`) | `data/traj_all.json` | figure |
| `plot_grid.py dc` / `plot_grid.py lg` | clean x-z grids for the dc & lg campaigns | `data/traj_{dc,lg}.json` | figures |
| `plot_mx.py` | mx dose-response + trajectory grid | `data/traj_mx.json` | figures |
| `plot_before_after.py` | July-vs-mx before/after comparison | both JSONs | figure |
| `plot_dc.py` / `plot_mx_multireset.py` | decomposition + multireset box plots | `data/*_multireset.json` | figures |
| `plot_trajectory_3d.py` | full 3D path + orthographic projections | `data/traj_dc.json` | figure |
| `swing_vs_speed.py` | does the swing buy contact speed? scatter | `data/traj_all.json`* | figure |

Typical loop: `trace_*` (needs the checkpoints in `logs/rsl_rl/z1_hammer/`) → `plot_*`. To just
regenerate a figure from committed data, run the `plot_*`/`swing_vs_speed` script alone.

## Other evaluation scripts (elsewhere — indexed here, not moved to avoid breaking references)

**Pre-training gates** (`docs/research/reward-design/`, referenced by `CLAUDE.md` — run before any GPU train):
- `validate_rewards.py` — reward-term end-to-end checks (phases A–M).
- `verify_contact_sensor.py` — contact sensor sanity.
- `verify_reward_setup.py` — random-policy reward sweep.
- `playback_reference.py` — scripted single-strike feasibility gate.

**Impulse / policy eval** (`scripts/`):
- `eval_impulse.py`, `eval_impulse.sh`, `slurm/vega_eval.sbatch` — the delivered-impulse / Λ eval harness.
- `diag_policy_trace.py`, `diag_impulse_trace.py`, `diag_strike_probe.py`, `diag_cuda_substep_probe.py` — diagnostics.

**Rendering / visual inspection** (`scripts/`, some referenced by `CLAUDE.md`):
- `render_policy.py` — headless offscreen render of a trained policy → PNG/mp4.
- `render_reference.py` / `render_reference_path.py` / `play_reference.py` — the scripted reference strike.
- `play.py` — interactive/viser playback.

**Fixed-impedance / whip probes** (`docs/results/assets/2026-07-17_fixed_impedance_diag/probes/`):
- `effort_sweep.py`, `dps_sweep.py`, `ceiling.py`, `armature_check.py`, `sensitivity_run.py`,
  `probe_battery.py`, `impedance_one.py`, `coord_whip2.py` — the Phase-0 / B2 CPU physics probes
  (press-vs-ballistic, the ~1.4 m/s effort ceiling, the ~4.2 m/s kinematic whip bound).

See `docs/results/2026-07-20_maximization_ablation_plan.md` for what these produced.

## Policy index

`python evaluation/list_policies.py` scans `logs/rsl_rl/z1_hammer/`, reads each run 's `params/` (reward weights, iters), and (re)generates `evaluation/POLICIES.md` — always current, no hand-maintenance. One-line campaign notes live in the script's `NOTES` dict.

# `evaluation/results/` — generated graphs & figures

Home for the figures the `evaluation/` scripts produce. **Convention: one folder per analysis campaign,
named `YYYY-MM-DD_<topic>/`** — so results are tracked by date and topic, and old graphs never get
silently overwritten by a new experiment.

- Figures are written here by `evaluation/trajectory/plot_*.py` (their `OUTDIR` points at the campaign
  folder). Regenerate any figure from the committed data in `evaluation/data/` — no re-running rollouts.
- Large media (`*.mp4`) is git-ignored per folder; PNGs are committed.
- When you start a new analysis, make a new `YYYY-MM-DD_<topic>/` folder and point the plot script's
  `OUTDIR` at it.

## Campaigns

### `2026-07-20_fic_maximization/` — the fixed-impedance impact-maximization investigation
(companion write-up: `docs/results/2026-07-20_maximization_ablation_plan.md`)

| figure | what it shows |
|---|---|
| `mx_dose_response.png` | the causal dose-response: swing rises with impact-max reward; speed flat; impulse rises via dwell |
| `mx_trajectory_grid.png` | mx arms (maxoff/maxon/maxmax/maxofftrk) head paths, x-z small-multiples |
| `dc_trajectory_grid.png` | decomposition paths — speed→swing vs impulse→straight press |
| `dc_decomposition_box.png` | the two rewards drive dissociable strategies (swing/force/dwell distributions) |
| `lg_trajectory_grid.png` | 500 vs 1500 iters — the overtraining drift/collapse, visible |
| `mx_multireset_box.png` | robustness across 50 randomized resets |
| `before_after_trajectories.png` | July (correlational) vs mx (controlled) — the method story |
| `trajectory_3d.png` | full 3D path + orthographic projections (reveals the lateral y-sweep) |
| `trajectories_grid.png` | the original July-policies x-z grid (the reference style) |
| `swing_vs_speed.png` / `swing_excursion.png` | does the swing buy speed? / per-policy swing |
| `af1_clean_strike*.{png,mp4}` / `nf1_park_and_farm*.{png,mp4}` | the two policy videos + montages |

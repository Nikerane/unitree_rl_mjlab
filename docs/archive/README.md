# Archived docs

Superseded / historical documents. Nothing here is deleted — it is the provenance / research
trail. Every file carries an in-file banner; a `PreToolUse` hook also warns on any read under
`docs/archive/`. **Do not act on anything here without checking `docs/README.md` (current truth).**

Three waves: an initial declutter (2026-06-17), the consolidation (2026-07-05), and the
tooling garbage-collection (2026-07-14, `tooling/`).

## Merged into a living doc (content absorbed verbatim)

| Doc | Absorbed into |
|---|---|
| `REWARD_LITERATURE.md` | `../research/reward-design/LITERATURE.md` |
| `hammering_literature_notes.md` | `../research/reward-design/LITERATURE.md` |
| `impact_tracking_rl_litreview.md` | `../research/reward-design/LITERATURE.md` (§2 bibliography + §4 corrections; generate-then-track framing dropped) |
| `CAT_DEEP_DIVE.md` | `../research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md` (CaT deep-dive appendix) |
| `JOINT_VELOCITY_BOUND_RESEARCH.md` | `../research/reward-design/CONSTRAINED_RL_LANDSCAPE.md` (joint-velocity-bound appendix) |

## Superseded by code or a current doc

| Doc | Why archived | Superseded by |
|---|---|---|
| `IMPACT_PROGRESS_IMPL_SPEC.md` | The `impact_progress` term shipped | `src/tasks/hammer/hammer_env_cfg.py` + `../research/hammering_reward_design_deep_dive_v2.md` |
| `REWARD_VALIDATION_METHODOLOGY.md` | Methodology now lives in the gate script | `../research/reward-design/validate_rewards.py` (docstring + phase A–M comments) |
| `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` | Its impulse machinery (windowed-reward + Lagrangian ladder) was superseded by shipped per-event-pulse soft-CaT | `../research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` + `../research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md` |
| `FUTURE_UPDATES.md` | Items applied or superseded | `../results/2026-06-17_b_strike.md` (applied) + `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (deferred reward work) |
| `HANDOVER.md` | 2026-06-09 cold-start handover; the "Z1 = throwaway diagnostic / start fresh on G1" framing was walked back | `../thesis/README.md` + `../../CLAUDE.md` (Z1-primary direction) |
| `DEEP_RESEARCH_REPORT.md` | Pre-2026-06-02 synthesis; predates the 7-term config + `impact_progress` | `../research/hammering_reward_design_deep_dive_v2.md` |
| `RECOMMENDED_REWARD_SPEC.md` | Aspirational 9-term spec; the implemented 7-term reward is the source of truth | `src/tasks/hammer/hammer_env_cfg.py` |
| `REWARD_DESIGN_MATRIX.md` | Lightweight term table; duplicates `RECOMMENDED_REWARD_SPEC.md` §3–4 | `src/tasks/hammer/hammer_env_cfg.py` (live terms + weights) |
| `IMPACT_TRACKING_REWARD_SPEC.md` | The generate-then-track / DeepMimic architecture the supervisor walked back | the single-policy direction — `../thesis/README.md` |
| `BASELINE_AUDIT.md` | 2026-05-22 point-in-time validation snapshot | `../VEGA_TRAINING_PLAN.md` + `../results/` |

## Archived tooling (`tooling/`, 2026-07-14 — the Vega layer; the cluster is retired)

| Script | Why archived | Superseded by |
|---|---|---|
| `tooling/setup_vega.sh` | Vega (EuroHPC) is no longer the GPU route | `scripts/lightning_pair.sh` (Lightning.ai) |
| `tooling/sanity.sbatch` | Vega Slurm smoke job | Lightning smoke: `ITERS=50 NENVS=2048 bash scripts/lightning_pair.sh` |
| `tooling/train_array.sbatch` | Vega Slurm seed-array launcher. Its `<timestamp>_<arm>_seed<N>` run-name contract is still live — canonical statement now in `scripts/compare_runs.py` | `scripts/lightning_pair.sh` |
| `tooling/eval_peak_qv.sh` | Frozen June joint-velocity eval protocol (hardcoded June Vega checkpoints); kept as the protocol lineage `scripts/eval_impulse.sh` cites | `scripts/eval_impulse.sh` (same-env cross-arm protocol, impulse metrics) |

Deleted outright the same day (zero consumers; recoverable from git history): `record_reference_trajectory.py` + its orphan `reference_strike_qtraj.npz` (recorded with the pre-`117e070` degenerate reference, loaded nowhere), `diag_grip_choice.py` (gripper-era EE), `diag_strike_trace.py` (June site-bug one-off, two-generations-stale overshoot), `_play_sanity.py`, `_play_checkpoint_sanity.py`, `retest_video.sh` (2026-05-22 audit one-offs), `visualize_terrain.py` (vendored upstream, non-thesis robots).

## Dated records (kept for provenance; no single successor)

| Doc | What it is |
|---|---|
| `OPUS_AUDIT.md` | 2026-05-22 audit record (the augment-not-replace strategy it recommended is now in `../../CLAUDE.md`) |
| `PEER_REVIEW_v2.md` | 2026-06-02 peer-review record (the press-exploit finding it flagged is captured in `../research/reward-design/OPEN_QUESTIONS.md` Q1) |
| `REAL_HAMMER_PLAN.md` | Executed 2026-06-17 real-hammer integration/recalibration record (gripper-era EE; the L6 fixture postdates it) |
| `hammering_lit_sweep_RUNBOOK.md` | Executed literature-sweep playbook (tiering ladder, integrity rules, corpus boundaries); its OUTPUT lives in `../research/reward-design/LITERATURE.md` |
| `2026-06-17-z1-single-strike-fix.md` | Executed plan; outcome in `../results/2026-06-17_b_strike.md` |
| `2026-06-17-r_imit-tracking-reward.md` | Executed plan; the `r_imit` tracking arm shipped in `src/tasks/hammer/` |
| `2026-06-17-z1-strike-not-press-redesign-design.md` | Executed "make it strike, kill pressing" design record |

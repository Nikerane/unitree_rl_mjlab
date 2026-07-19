# Vega experiments — local backup manifest (2026-07-20)

All GPU-trained policies + training curves were Vega-only (scratch FS, went read-only once already).
**Backed up locally 2026-07-20** to `logs/rsl_rl/z1_hammer/` (gitignored — tooling reads it, git stays clean).

**What was pulled (lean set, ~242 MB):** each run's `model_499.pt` (final policy), `events.out.tfevents*`
(full training curves), `params/*.yaml` (env+agent config — reproducibility), and `git/` (rsl_rl's captured
training-time diff — provenance). **Intermediate checkpoints (`model_0/50/250`, ~694 MB) were NOT pulled**
(local disk was near-full); they remain on Vega `~/repos/unitree_rl_mjlab/logs/rsl_rl/z1_hammer/` until it
purges. Eval summary CSVs are in `eval/` + banked under `docs/results/assets/`.

To re-analyze a policy: `PYTHONPATH=. python scripts/eval_impulse.py --ckpt logs/rsl_rl/z1_hammer/<run>/model_499.pt ...`
or curves via TensorBoard on `logs/rsl_rl/z1_hammer/`. Per-run provenance: `<run>/git/unitree_rl_mjlab.diff`.

## Campaign index (newest first)

| Campaign | Date | n | What it was | Description doc |
|---|---|---|---|---|
| **af1_fixed** | 07-19 | 3 | Audit-fix regression validation (F1/F5/F9/F6/F4/F11) — behavior-neutral, git 6b447bf | `assets/2026-07-17_depth_gate_sweep/RESULT.md` (af1 section) |
| **dg1_gate / dg0_nogate** | 07-17 | 3+3 | Depth-gate sweep (F1 Phase-2 Change-1): gate-on vs gate-off → NO-OP | `assets/2026-07-17_depth_gate_sweep/RESULT.md` |
| **ip24_ip / ip48_ip** | 07-17 | 3+3 | impact_progress weight 24 / 48 (ante-impact velocity) → ip24 FLAT, ip48 DIVERGED 3/3 | `RESULT.md` (Phase-2b) + `2026-07-20_B2_effort_ceiling.md` |
| **g1_none / g1_track / g2_delivoff** | 07-17 | 8×3 | F0/F1/F2 8-seed maximization campaign (A-none / A-track / delivered-off). The canonical fixed-impedance result | `assets/2026-07-16_parking_fix/g_ff0f1f2_eval_summary.csv`, `2026-07-15_vega_campaign_plan.md` |
| **f1_none / f1_track / f2_delivoff** | 07-16 | 8×3 | Earlier F campaign — TIMED OUT (warp contention), re-run as g. Superseded | `2026-07-15_vega_campaign_plan.md` |
| **a05_none / a05_track** | 07-16 | 3+2 | Parking-fixed A arm (nail_driven 2.0→0.5): success 0→1.00 | `assets/2026-07-16_parking_fix/ab1_eval_summary.csv`, `nail-driven-farm-bug` memory |
| **b1_noterm** | 07-16 | 3 | Option B (DAPG NoTerm, completion 100→1.0) — DISQUALIFIED (multi-strike 83.8× i_ref) | `2026-07-15_vega_campaign_plan.md` |
| **nf1_none / nf1_track** | 07-15 | 3+3 | nail_driven=2.0 validity (parking farm) — 5/6 parked <30mm | `nail-driven-farm-bug` memory |
| **a_base / a_track / b_strike** | 06-17 | 3×3 | Earliest baseline / prior / velocity-bound strike campaigns (V1) | (older records) |
| **c_a1_delta010 … c_a4_dcmotor, c_hardterm, c_softcat** | 06-17/18 | 3–6 | CaT ablation lineage: velocity-bound arms A1(delta), A2(vpenalty), A3(cat), A3s(substep), A4(dcmotor), hardterm, soft-CaT | `soft-cat`/`cat-constraints-as-terminations` memories, `docs/research/reward-design/` |

## Key results these back (see the per-campaign docs for numbers)
- **Fixed-impedance impulse maximization is closed** (g/dg/ip): reward reshaping can't exceed ~0.87× i_ref.
- **The ceiling is a hardware joint-velocity limit** (B.2 CPU probes, not these GPU runs): `2026-07-20_B2_effort_ceiling.md`.
- **Constraint is vacuous / press-not-ballistic** (Phase-0 CPU): `2026-07-17_phase0_diagnostics.md`.
- **All fixed-impedance runs are log-only** (`imp_max_p=0`, δ≡0) — enforcement never shaped a GPU run.

## TODO / gaps
- Intermediate checkpoints (model_50/250) not backed up — pull from Vega if a mid-training analysis is needed
  (e.g. ip48 divergence used model_50), before Vega purges.
- Local disk was 99% full at backup time — consider external/cloud storage for the full 900 MB set + future runs.
- The full Codex handover is `2026-07-20_CODEX_HANDOVER.md`; the experiment menu is `2026-07-19_next_experiments_codex.md`.

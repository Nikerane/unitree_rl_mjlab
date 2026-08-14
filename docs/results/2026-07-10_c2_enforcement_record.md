# C2 enforcement gate — durable provenance record (2026-07-10)

> **Impulse-threshold provenance correction (2026-08-14):** This record's “real caps,” `kappa=2` basis, and `[1.64, 3.28, ...]` vector are historical experiment inputs, not manufacturer-certified Z1 damage limits. Its body and numbers remain frozen for reproducibility; see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md` for the current banked/provisional/diagnostic identities.

> **Dated result record** — facts frozen as of the C2 gate run (2026-07-10). Written retroactively
> on 2026-07-14 to discharge adversarial-review finding **I9** (the C2 rationale previously lived
> only in the gitignored SDD ledger and now-deleted `/tmp` transcripts). Where a number has since
> gone stale, the **STALENESS** notes below say so explicitly — check them before reusing any
> figure from this record.

## What C2 was

C2 = the local (CPU) enforcement gate for the impulse-CaT arm: turn `imp_max_p` on against a
scaled-down probe limit, certify the δ machinery is *graded* (not saturated, not inert), pick the
production `imp_max_p`, and surface the `delivered_impulse` reward-share decision. Machinery:
`docs/research/reward-design/c2_enforcement_gate.py` + `reward_design_util.py`
(commits `e4201cb`, fix `381575d`, plan amendment `74672f6`).

## Protocol (as amended 2026-07-10)

- **Probe limit** `0.05 × IMP_J_LIMIT` → **amended to `PROBE_SCALE = 0.02`**: reference strikes
  measure only ~2.9–3.8% of the real caps (binding_ratio 0.029 real-caps report; derive-script [4]
  worst-joint p95 3.8%), so 0.05 sat *above* every strike and produced zero over-limit samples.
- **Heights `(0.06, 0.10, 0.15)` ascending = most-violent-FIRST** (amended from descending): under
  the post-2026-07-06 windup NEAR_NAIL geometry, excess *grows* as approach height shrinks. The
  CaT hook seeds its normalizer `c_max` from the FIRST over-limit sample and slow-EMA-updates
  after (`tau=0.95`), so an *increasing* excess sequence keeps every sample above the lagging
  normalizer and δ saturates at `max_p` on every strike — the graded-δ check could never pass in
  the original order. Most-violent-first seeds `c_max` at the top, letting later smaller excesses
  grade strictly inside `(0, max_p)`.

## Gate results (both `imp_max_p ∈ {0.25, 0.5}` PASS, exit 0)

| Check | Result |
|---|---|
| Graded δ (probe caps) | 3/3 over-limit strikes; first (most violent) saturates at `max_p` **as expected** (it seeds `c_max`); later strikes grade strictly inside: δ = **[0.5, 0.369, 0.112]** at `max_p=0.5`, **[0.25, 0.184, 0.056]** at 0.25 |
| `c_max` self-seed | binding joint (joint3) seeded to **0.01346** (> the 1e-3 `imp_seed` floor, finite, ≤ 100× worst excess 0.0142); compliant joints stay at the floor |
| Regression tripwire (4a) | probe-pass nail depths byte-identical (atol 1e-6) to the log-only baseline for both `max_p` — δ has no physics pathway in playback |
| δ delivery (4b) | probe passes deliver δ>0 on contact; log-only baseline δ ≡ 0 |
| Real-caps report | band_fraction [0.8·J, J] = 0.000; binding_ratio max Λ/J = **0.029** — expected for gentle reference strikes |
| CPU smoke train under enforcement | `imp_max_p=0.5`, 5/5 iterations, no NaN (tyro flag `--env.metrics.cat-soft.params.imp-max-p` verified) |

## Decisions recorded

1. **`imp_max_p = 0.5` chosen** — did not degenerate in the probe (graded δ 0.369/0.112 strictly
   inside (0, 0.5); `c_max` bounded; depths unperturbed) and keeps cross-arm comparability with
   the velocity soft-CaT arm's `max_p=0.5`. The repo default stays **`imp_max_p=0` (log-only)**;
   0.5 is the per-run value for enforcement campaigns.
2. **`delivered_impulse` weight 2.0 KEPT (user decision)** — the reward-share table on the
   reference-strike battery measured `delivered_impulse` at **26.5% of the positive reward total**
   (nail_depth_delta 32.1%, impact_progress 31.7%, approach 8.1%, nail_driven 1.6%); surfaced as
   the explicit weight decision point, user kept 2.0.
3. `IMP_J_LIMIT = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` N·m·s — untouched throughout
   (measured 2026-07-06 fixture-era derivation, τ_rated × 2 HD-Repeated-Peak × Δt≈27.3 ms).

## STALENESS notes (read before reusing these numbers)

- **The 26.5% delivered share is STALE.** The 2026-07-13 reference follow-through fix (`117e070`)
  raised `i_ref` 0.0811 → **0.6094** (the reward's normalizer), so the share must be re-measured
  on the first post-fix run before trusting the weight-2.0 decision (flagged in `env_cfgs.py`'s
  i_ref comment).
- **The `imp_max_p=0.5` calibration predates the sliding-window semantics** (`581578d`,
  2026-07-13). Window (25 substeps) > decimation (10) means one violation now spans ~3
  consecutive 50 Hz reads → per-event survival ≈ (1−δ)³, ~3× the termination pressure per
  violation vs the single-read pulse semantics C2 calibrated under. **Recalibrate before any
  enforcement run** (pre-enforcement gate (d) in the 2026-07-12 Khadiv VIC addendum).
- **Enforcement itself is gated on Khadiv decision (e)** (what quantity Λ should bound —
  ballistic / windowed reaction / split) plus the four pre-enforcement gates; see
  `docs/results/2026-07-12_khadiv_vic_addendum.md`.
- The probe geometry (windup NEAR_NAIL, approach-height sweep) predates the derive-script's
  2026-07-14 switch to a `WINDUP_CLEARANCES` sweep (the apex floor collapsed the height sweep);
  if C2 is ever re-run, mirror that amendment.

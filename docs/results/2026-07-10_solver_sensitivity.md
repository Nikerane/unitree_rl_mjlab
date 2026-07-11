# 2026-07-10 — Solver-parameter sensitivity of the per-joint impact impulse Λ_j

> **Provenance:** one-off local defense experiment, run 2026-07-10 on branch `soft-cat`,
> HEAD `704f30b` (repo untouched during the run — all perturbations applied to scratchpad copies /
> in-memory spec wrappers). Env: conda `unitree_mjlab`, mjlab 1.4.0, mujoco_warp 3.8.1,
> Warp 1.13.0 (CPU backend). Recommended by `../research/reward-design/QFRC_MEASUREMENT_AUDIT.md`
> §2 item 1 ("Recommended addition").

## 1. Context

`QFRC_MEASUREMENT_AUDIT.md` §2.1 identifies solver-parameter dependence as the biggest scientific
critique of `Λ_j = Σ|qfrc_constraint_j|·dt`, and offers the momentum-pinning defense: the contact
**force profile** is a modeling choice (solref/solimp/timestep), but the **impulse integral** should
be pinned by momentum transfer (∫F·dt ≈ m_eff·Δv) *provided the accumulation window covers the whole
event*. Prediction under test: **"impulse moves by percent while peak force moves by tens of
percent."** This run executes the audit's recommended one-off sensitivity experiment.

## 2. What changed (configurations, exact before → after)

| | BASELINE | HALF-TIMESTEP | SOLREF-2X |
|---|---|---|---|
| physics timestep | 0.002 s | **0.001 s** | 0.002 s |
| decimation | 10 | **20** | 10 |
| control dt (trajectory) | 0.02 s | 0.02 s (unchanged) | 0.02 s |
| `hammer_head_0/1` geom solref | (0.02, 1) | (0.02, 1) | **(0.04, 1)** |
| `nail_head` geom solref | (0.008, 1) | (0.008, 1) | **(0.016, 1)** |
| combined contact timeconst* | 0.014 s | 0.014 s | **0.028 s** |
| `timeconst ≥ 2×timestep` valid? | yes (7×) | yes (14×) | yes (14×) |

\* MuJoCo mixes geom solref by solmix-weighted average; both sides have default solmix=1 and
priority 0 (no overrides in either XML), so combined = mean. SOLREF-2X doubles the timeconst on
**both** contacting geom sets so the mixed contact timeconst exactly doubles (contact stiffness ÷4).
Override applied by wrapping the entity `spec_fn`s **before compile** (no asset/repo file edited)
and **verified in the compiled model** (`geom_solref` read back: 0.04/0.04/0.016). Half-timestep:
`env.physics_dt` asserted = 0.001 and all accumulators (script-side and shipped
`SubstepImpulseAccumulator`/`ContactRowImpulseAccumulator`) read `env.physics_dt`, so the
accumulation dt follows automatically; doubling decimation keeps the control-rate reference
trajectory unchanged (verified: approach speed at first touch −0.44 m/s in all three configs).

## 3. Method

- Copy of `docs/research/reward-design/derive_impulse_thresholds.py` (gate-identical 15-strike
  protocol: 3 approach heights × 5 repeats, per-substep hook, CPU/warp), parametrized as
  `sensitivity_run.py` (in [assets](assets/2026-07-10_solver_sensitivity/)).
- Whole-event adjudication via `diag_event.py` (1 strike/height, full-episode time series) — checks
  whether the contact window actually contains a completed event.
- **Baseline reproduction (patch-integrity gate):** joint2 raw Λ = 0.0689 (committed ≈ 0.069 ✓),
  joint3 rows Λ = 0.0420 (≈ 0.042 ✓), object-side ∫F·dt = 0.0811 N·s (≈ 0.0811 ✓).
  Shipped-accumulator leak = 0, shipped-vs-script cross-checks exact (excess 0.000000) **in all
  three configs** — the measurement machinery is intact under both perturbations.

## 4. Results

### Absolute values (means over 15 strikes)

| quantity | BASELINE | HALF-DT | SOLREF-2X |
|---|---:|---:|---:|
| raw Λ j1/j2/j3/j4/j5/j6 [N·m·s] | 0.0044 / 0.0689 / 0.0146 / 0.0572 / 0.0017 / 0.0005 | 0.0042 / 0.0654 / 0.0106 / 0.0534 / 0.0018 / 0.0003 | 0.0038 / 0.0531 / 0.0044 / 0.0462 / 0.0013 / 0.0004 |
| sub Λ (shipped, enforced) [N·m·s] | 0.0011 / 0.0416 / 0.0420 / 0.0299 / 0.0007 / 0.0000 | 0.0011 / 0.0394 / 0.0366 / 0.0274 / 0.0008 / 0.0001 | 0.0005 / 0.0257 / 0.0263 / 0.0189 / 0.0003 / 0.0000 |
| rows Λ (efc-row GT) [N·m·s] | 0.0030 / 0.0416 / 0.0420 / 0.0303 / 0.0017 / 0.0001 | 0.0029 / 0.0394 / 0.0366 / 0.0275 / 0.0017 / 0.0001 | 0.0017 / 0.0257 / 0.0263 / 0.0189 / 0.0009 / 0.0001 |
| object-side ∫F_axial·dt [N·s] | 0.0811 | 0.0770 | 0.0502 |
| PEAK F_axial (mean over strikes) [N] | 3.39 | 3.38 | 2.02 |
| peak \|qfrc\| j2/j3/j4 [N·m] | 2.73 / 0.75 / 2.20 | 2.73 / 0.52 / 2.13 | 2.04 / 0.47 / 1.81 |
| contact window [substeps] | 13.7 | 26.0 | 13.7 |
| contact window [ms] | 27.33 | 26.00 | 27.33 |

### % deltas vs baseline

| quantity | HALF-DT | SOLREF-2X |
|---|---:|---:|
| **(a) per-joint Λ** — sub j2 / j3 / j4 | −5.2% / −12.8% / −8.3% | −38.1% / −37.3% / −36.7% |
| rows j2 / j3 / j4 | −5.2% / −12.8% / −9.4% | −38.1% / −37.3% / −37.7% |
| raw j2 / j3 / j4 | −5.1% / −27.5% / −6.6% | −23.0% / −69.8% / −19.2% |
| (small joints j1/j5/j6, tiny absolute values) | −4% … +71% | −10% … −57% |
| **(b) object-side ∫F·dt** | **−5.0%** | **−38.1%** |
| **(c) peak force** — object-side peak F_axial | **−0.2%** | **−40.4%** |
| peak \|qfrc\| j2 / j3 / j4 | −0.1% / −30.1% / −2.7% | −25.2% / −36.9% / −17.7% |
| **(d) window length [ms]** | −4.9% (13.7→26.0 substeps) | +0.0% (identical substep counts) |

Internal consistency across configs: worst-joint rows Λ ÷ object-side ∫F·dt =
0.518 / 0.512 / 0.524 m (baseline/halfdt/solref2x) — the robot-side and object-side measurements
move **together** (stable effective moment arm, spread ~2%), so the solref2x shift is physical,
not an instrumentation artifact.

### Whole-event adjudication (`diag_event.py` — full-episode time series, 1 strike/height)

The audit's momentum-pinning defense is conditioned on the window covering the whole event. It
does **not** on this scene:

| config | h | windows | window end | ∫F·dt first-window vs whole-episode | nail depth at end [mm] | v_head at touch [m/s] |
|---|---|---|---|---|---|---|
| baseline | 0.06/0.10/0.15 | 1/1/1 | **recording end (no release)** | identical (0.0881/0.0818/0.0733) | 9.1 / 8.2 / 6.9 | −0.44 |
| halfdt | 0.06/0.10/0.15 | 1/1/1 | recording end (no release) | identical (0.0847/0.0772/0.0692) | 8.5 / 7.4 / 6.2 | −0.44 |
| solref2x | 0.06/0.10/0.15 | 1/1/1 | recording end (no release) | identical (0.0558/0.0511/0.0437) | 5.4 / 4.8 / 3.9 | −0.44 |

- **Contact never releases within the protocol.** Every strike has exactly one contact window that
  begins 12–15 substeps (baseline count) before the end of the `playback + 6 hold steps` recording
  and is still closed when recording stops. The "contact window length" is therefore
  **protocol-clocked, not physics-clocked** — which is why it is byte-identical (12/14/15 substeps)
  between baseline and solref2x.
- **No missing impulse in re-contacts:** first-window ∫F·dt == whole-episode ∫F·dt exactly, in all
  configs. The solref2x impulse deficit is not measurement leakage.
- **The physical outcome changes with solref:** nail penetration drops −41…−43% under solref2x
  (−7…−10% under halfdt) — the softer contact genuinely transfers less momentum in the available
  time.
- **The event is a driven press, not a ballistic impact:** approach speed is only 0.44 m/s, the
  position-controlled arm drives the head *through* the nail-top target, and the force profile is a
  flat plateau (~3.4 N ≈ mean force 3.0 N; peak ≈ mean) set by the nail's resistance — there is no
  impact spike whose shape the solver could reshape at constant area.

## 5. Conclusion (split verdict)

**The data do NOT support the blanket claim "impulse moves by percent while peak force moves by
tens of percent" on this scene. The result is split by parameter:**

1. **Timestep (discretization): SUPPORTED.** Halving the timestep (with the control trajectory held
   fixed) moves the enforced/GT impulse by ~5% on the dominant joints (j2 −5.2%, j4 −8/−9%) and the
   object-side ∫F·dt by −5.0%, while the number of integration substeps doubles. (Joint3, the known
   dof-friction sign-cancellation joint, moves −12.8% sub/rows and −27.5% raw — the raw metric's
   fragility there was already documented.) Λ is robust to the integration grid.
2. **Contact solref (constitutive softness): NOT SUPPORTED — negative result, reported as-is.**
   Doubling the contact timeconst moves the impulse by **−37…−38%** (sub/rows j2–j4, and
   object-side −38.1%) — the *same* tens-of-percent scale as the peak force (−40.4%). Impulse
   inherited the force's parameter fragility ~1:1.
3. **Why the defense's premise fails here (and where it would hold):** the momentum-pinning
   argument requires a completed impact contained in the window (∫F·dt ≈ m_eff·Δv with the
   impactor's Δv fixed by approach kinematics). The Z1 reference strike is not that event: it is a
   slow (0.44 m/s), position-controlled, resistance-limited **press whose contact never releases
   within the measurement protocol** — the window is truncated by the end of playback at a fixed
   27.33 ms, so Λ ≈ F̄ × T_window with T fixed, and F̄ scales with contact stiffness. Under solref2x
   the combined contact timeconst (28 ms) even exceeds the whole observable window (27 ms). A
   genuinely impulsive strike (higher approach speed, contact that closes with an observed release
   inside the window) is the regime where the claim could still be demonstrated; this experiment
   does not test that regime and therefore cannot confirm it.

**Peak force note:** peak force was *predicted* to be fragile in both arms; it was fragile under
solref (−40%) but *stable* under timestep (−0.2%) — because the profile is a resistance-limited
plateau, not a stiffness-limited spike. So even the "peak is fragile" half of the prediction holds
only for the solref axis on this scene.

## 6. Decision / implications

- **The "the integral is pinned by momentum transfer" defense cannot be quoted unconditionally**
  for the current Z1 hammer scene: state it for the *ballistic/impulsive* event class, demonstrate
  timestep-robustness with this experiment's half-dt arm, and treat solref as recorded modeling
  provenance (as the audit's mitigation already plans) rather than claiming solref-invariance.
  `QFRC_MEASUREMENT_AUDIT.md` was updated in place (2026-07-10 annotations) to state the defense
  conditionally.
- **Side-finding for the C0 gate's documentation:** the gate's "contact window" (and hence the
  committed Λ figures 0.069/0.042 N·m·s, object-side 0.0811 N·s, and the derived J_limit
  Δt_impact = 27.3 ms) is **protocol-truncated** — contact is still closed when the reference
  playback + 6-step hold ends. The numbers are self-consistent as defined, but they are "first
  27 ms of a continuing press", not a completed impact event.
- **Design consequence (not a bug) for training:** under the shipped per-event PULSE semantics, an
  unreleased sustained press is ONE growing event — the constraint therefore penalizes sustained
  pressing *pressure* via a growing δ, not only sharp impacts. That is coherent (the harmonic drive
  does not care whether the load is a spike or a press) but must be stated explicitly in the
  writeup rather than implying the constraint only sees impacts.
- The three-way machinery itself (raw/sub/rows + object-side) is solid: shipped-vs-script
  cross-checks exact and the robot-side/object-side ratio stable within ~2% across all three
  configs.

## 7. Caveats

- Play-mode resets are deterministic (`position_range=(0,0)`), so the 15-strike protocol is really
  **3 distinct strikes × 5 identical repeats** (per-strike values within a height are
  bit-identical). Means are over 3 effective samples; no strike-to-strike noise floor exists to
  compare deltas against — but the deltas (5% vs 38%) are far above float/warp noise anyway.
- CPU float32 warp backend; mujoco_warp is non-deterministic on GPU but the CPU runs here
  reproduced the committed numbers exactly.
- `raw Λ` on joint3 and small-magnitude joints (j1/j5/j6, Λ < 0.005 N·m·s) show large *relative*
  deltas on tiny absolute values; the enforced (sub) and GT (rows) metrics on the load-bearing
  joints j2–j4 are the meaningful rows.
- The shipped delivered-impulse per-event cap (25 substeps) halves its wall-clock coverage at
  half-dt (25 ms vs 50 ms); the reported ∫F·dt here is the script-side uncapped window integral,
  unaffected. (If half-dt were ever adopted for training, `event_window_substeps` would need
  rescaling.)
- solref2x doubles the hammer-geom solref globally (hammer↔anything), not only vs the nail; in
  these episodes the hammer head contacts only the nail, so the difference is immaterial here.

## 8. Reproduce / artefacts

All artefacts preserved under [`assets/2026-07-10_solver_sensitivity/`](assets/2026-07-10_solver_sensitivity/):

- `sensitivity_run.py` — the main run script (gate-identical 15-strike protocol, parametrized over
  timestep/decimation/solref; run per config from repo root with the `unitree_mjlab` env).
- `diag_event.py` — the whole-event adjudication script (full-episode time series, window census).
- `baseline.json` / `halfdt.json` / `solref2x.json` — raw per-strike results (per-strike arrays
  included).
- `tables.txt` — the absolute-values / %-delta / headline tables as generated.

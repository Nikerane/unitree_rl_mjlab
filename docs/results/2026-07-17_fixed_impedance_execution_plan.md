# Fixed-impedance experiment campaign — execution plan (2026-07-17)

**For approval before execution.** Companion to `2026-07-17_fixed_impedance_deep_dive.md` (the *why* — system
model + findings). This is the *how* — sequenced, with exact scripts/diffs/commands from a 5-agent dig
(Codex probe-runbook + 4 Claude digs + external research). VIC stays parked (user, 2026-07-17). Nothing here
is executed until signed off.

## The through-line (what every experiment is really testing)
On fixed impedance the ballistic impact impulse is set **pre-contact** by `m_eff·v`, and `v` is effort-clamped
at ~1.3 m/s while the 7 g nail yields (e≈0). So **both** spine symptoms — "delivered ≈ reference, never above"
and "Λ never binds" — are the *same* physics: the enforced Λ and the deliverable impulse are **press
integrals**, and the press regime can't be pushed past reference nor made to bind meaningfully (lit-confirmed:
Varin IROS'19, van Steen 2024 impact-aware control; CaT 2024 + Paternain NeurIPS'19 for the slack-constraint
side). The campaign's honest goal is therefore: **characterize the fixed-impedance impact envelope precisely,
push the policy to that physical ceiling with better reward shaping, and quantify exactly how far under the cap
it sits** — so the case for variable impedance (later) rests on measured ceilings, not assertion.

---

## Phase 0 — Local diagnostic battery (no GPU, no Vega, ~CPU-minutes each)
**Goal:** map the press↔ballistic knee and the constraint's bindability across the reachable envelope, cheaply.
Every item **extends an existing probe** (Codex-verified); all CPU-only, no MuJoCo GL, no `mjpython`. Run under
the conda env `unitree_mjlab`.

| ID | Extend | What to add | Settles |
|---|---|---|---|
| **A1** velocity sweep 0.5×–8× | `docs/results/assets/2026-07-12_impulse_vacuity/probes/probe_battery.py` (speed grid already at :179; per-substep force/kinematics :45; contact-window census :104) | emit CSV of Λ/cap, delivered ∫F·dt, peak & mean force, contact-window duration, release census — **x-axis = measured `approach_speed_at_contact` (:122), not command factor** (action clamps at :84) | Λ linear-in-v + duration-independent ⇒ ballistic; flat-once-clamped + duration-driven ⇒ **press** |
| **A2** solref ×0.5/×2 | `docs/results/assets/2026-07-10_solver_sensitivity/sensitivity_run.py` (wraps both spec_fns, verifies compiled `geom_solref`, records peak force :205) | targets `nail_head`, `hammer_head_0/1` (:53); report ΔΛ vs Δpeak-force + release status | press ⇒ Λ tracks force (~−38%); ballistic ⇒ solref-robust. **THE discriminator** |
| **A3** window/cap 2-point | `docs/results/assets/2026-07-12_impulse_vacuity/probes/impedance_one.py` (reliable rigid target via nail-slide armature lock + displacement<1mm assert :100; captures shipped accumulator :76) | assert runtime `acc._window==25`; ratio Λ under shipped 27.3 ms-caps `[1.64,3.28,…]` **vs** formula-consistent 50 ms-caps `[3,6,3,3,3,3]`, on yielding **and** rigid target | the **decision-(e) support table** (binds 1.12–1.17× vs 0.61–0.64× — pairing, not training, decides) |
| **A5** rigid + velocity-injection | `frontier_one.py` (point; zeroes/scales impact-time PD, injects joint velocity :112) + `frontier_sweep.py` (one-env-per-process driver) — **fix:** use A3's armature-lock rigid target, NOT `frictionloss=1e6` (:46) | report **impact-only** Λ/cap (impact window only, vs the 50 ms enforced) + reachability (`max|q̇|` vs `Z1_JOINT_VEL_LIMIT`, achieved head speed) | does *any* reachable regime cross cap by **impact** not press? = can the constraint ever bind on a genuine strike |
| **A6** armature sensitivity | `docs/results/assets/2026-07-12_impulse_vacuity/probes/armature_check.py` (reads `dof_armature`, builds full M + Jacobian, scales ×1/×10/×30 :46) — reuse `ceiling.py::over_cap()/crossing_v()` | recompute `m_eff` + ballistic Λ/cap margin at armature ×1/×10/×30 | Khadiv (f): bounds the model-sensitivity of the vacuity number (m_eff 0.4→~1.4 kg roughly doubles the ballistic margin) |

**Deliverable:** one short results note (`docs/results/2026-07-18_*`) with the A1–A6 tables + the decision-(e)
support table. This is the "understand the system" payload and needs **zero** Vega.

---

## Phase 1 — Instrumentation (cheap code, before any Vega sweep)
So every training run *shows* the constraint behaving live. **Correction from the dig:** true per-episode
Λ/cap *quantiles* cannot be logged during training (the MetricsManager reduces every metric to an env-mean,
twice). What *is* loggable and meaningful: **mean worst-ratio + over-cap fraction** (the tail proxy) +
contact-window length + press-domination fraction. Genuine p95/max stays in `eval_impulse.py` (already computes
`worst_ratio_max_sampled`).

**C1 wiring (~15 lines, 3 files, no runner change):** add readers `worst_ratio`, `over_cap_frac`,
`contact_window_len`, `press_dom_frac` to `src/tasks/hammer/mdp/impulse_bound.py` (read
`_episode_peak_perjoint` / the `hammer_nail_contact` `last_contact_time`), export in `mdp/__init__.py`, register
4 `MetricsTermCfg`s (`reduce="last"`) in the `if cat_impulse:` block of `env_cfgs.py` (~:305, after `cat_soft`
so the ordering guard holds). TB keys `Episode_Metrics/{worst_ratio,over_cap_frac,contact_window_len,
press_dom_frac}`, live from iter 1 on all `-CaT-Impulse*` arms. Guard with a small unit test + `validate_rewards.py`.

---

## Phase 2 — Impulse-maximization experiments (Vega; small seed counts; after Phase 0/1)
The productive fork the research surfaced — **test both, let data decide** (both are cheap once instrumented):

**Track 2a — ante-impact velocity (the physically-honest "strike harder").** Reward pre-impact momentum, not
the press integral. **`impact_progress` already is an ante-impact axial-velocity term** — so reshape it:
sweep `impact_progress.weight` (8 → 16/32), raise `v_expected`, and test removing its one-shot / depth gates so
it keeps paying for faster contact. *This is the lever the literature endorses (reward nail/outcome velocity).*
Existing weights are directly CLI-overridable; gate-removal needs the 2-step config edit.

**Track 2b — delivered press integral (the direct-but-limited lever).** Reward-impl spec, `rewards.py`:
- **Change 1 — remove the `delivered_impulse` depth-gate** (make ∫F·dt into a seated nail pay). Safe: the
  per-event 50 ms cap + re-arm debounce bound press-farming *at the accumulator* (gate is now redundant).
  Breaks `test_depth_gate_blocks_payout_without_progress` → rewrite as `test_pays_regardless_of_progress`.
- **delivered weight sweep** 2 / 20 / 60 (combined with gate-off — weight alone saturates).

**Track 2c — completion-cliff soften (Change 3), coupled.** `completion` 100 → 20 **requires** `nail_driven`
0.5 → ~0.05 (or `nail_driven`→0.1 + `approach`→0.05) to keep `test_nail_driven_not_farmable` passing
(`w_nd·0.9766 + w_ap < 0.20`). NOT `no_terminate` (proven worse). `nail_depth_delta` (w=600) carries success,
so softening completion doesn't kill the finish incentive.

**Metrics compared across tracks:** delivered/i_ref (does it exceed 1.0?), ante-impact head/nail velocity,
Λ/cap (Phase-1 live metrics), success, contact-window length (press-farm watchdog).

**Launch (dig-verified):** task `Unitree-Z1-Hammer-CaT-Impulse`, 4096 envs, `imp_max_p 0`, save-interval 50,
500 iters. Weight sweeps reuse `vega_train.sbatch` `SINGLE_TASK` + `DELIVERED_W`; `completion.weight` needs one
new `EXTRA_FLAGS` passthrough line; a *new* toggle (depth-gate) needs the config edit first. Always
`--exclude=gn03`; budget ~10 min warp/cephfs startup into `--time`; eval self-certifies via
`impossible_success_n`/`lambda_dead_n`.

### ⛔ Safety gate (do NOT skip)
**Change 2 (superlinear excess-over-i_ref bonus) stays code-only / disabled until `imp_max_p > 0`.** Under the
C0 log-only default (no CaT counter-pressure) it is an *unbounded "hit as hard as possible" farm*. Land its code
+ tests behind weight 0; enable only after enforcement is live (post-Khadiv). This is the one lever that
*needs* the constraint switched on to be safe.

---

## Phase 3 — Constraint-binding characterization → Khadiv decision package (analysis)
Fold Phase 0's decision-(e) table + Phase 2's live Λ/cap distributions into a decision memo for Khadiv (e):
bound the **ballistic** impulse (clean but provably vacuous), the **windowed-press** reaction (binds only under
the 27.3 ms-cap/50 ms-window pairing; reduces to avg-torque×window — invites "why not just bound torque?"), or
**split**. Plus (f) armature verification (A6). **Enforcement (`imp_max_p>0`) stays gated** — it is a guaranteed
no-op at the shipped pairing on a healthy policy, and enabling it needs the event-level `1−∏(1−δ_t)`
recalibration (window>decimation ⇒ ~3 reads/violation). Do not enable unilaterally.

---

## Guardrails (standing, from CLAUDE.md + user)
- `imp_max_p` stays **0** until Khadiv (e)/(f). `IMP_J_LIMIT` values are hardware measurements — **fixed** (do
  not tighten to manufacture a bind). **No VIC.**
- Full pytest green + `validate_rewards.py` (A–M) + `verify_contact_sensor.py` before any Vega training.
- Commit only named files; no `git add -A`; no co-author trailer; never commit `.env/.codex/AGENTS.md/CLAUDE.md`.
- One GPU per run; re-read files before editing (tree edited concurrently); never edit installed packages.

## Housekeeping (opportunistic, cheap)
Update campaign-plan F2 5/6→7/8 + annotate the retracted line-76 vacuity claim; fix stale line cites
(`nail_driven` :172→:177, i_ref :347→:348) by symbol; note Option-B's disqualifier is training-env-only.

---

## Proposed order of execution
1. **Phase 0** (A1→A2→A3→A5→A6, local, no approval-risk) + write the results note. ← *start here*
2. **Phase 1** C1 instrumentation (small PR + tests).
3. **Commit** the already-baked `nail_driven=0.5` fix (still uncommitted) alongside Phase-1.
4. **Phase 2** Vega maximization sweeps (2a + 2b + 2c), instrumented.
5. **Phase 3** Khadiv decision memo.

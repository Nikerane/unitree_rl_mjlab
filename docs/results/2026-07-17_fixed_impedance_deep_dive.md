# Fixed-impedance deep dive — system model + experimental agenda (2026-07-17)

> **Impulse-threshold provenance correction (2026-08-14):** The `27.3 ms` basis and
> `[1.64, 3.28, ...]` vector below are historical project inputs, not a validated Z1
> reaction-impulse or damage limit; the unsupported `kappa=2` interpretation is retired.
> The numeric body remains frozen; see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Status:** living plan of record for the *fixed-impedance* phase. VIC is deferred (user, 2026-07-17:
"keep VIC for later, not now at all"). Built from a 5-map salvage of the `fixed-impedance-deep-dive`
workflow (`wf_33142d62-c2d`; web-research + parameter-inventory legs lost to a session limit) plus the
completed g-campaign eval (`docs/results/assets/2026-07-16_parking_fix/g_ff0f1f2_eval_summary.csv`,
24/24, all invariants clean). Code re-verified against the tree, not memory.

---

## Part 1 — System model (the two spine questions)

### The one finding both questions converge on
On the fixed-impedance Z1 **the enforced Λ and the deliverable impulse are BOTH press integrals, not
ballistic momentum** — because the ballistic regime is physically unreachable. A 7 g nail that *yields*
(drives ~33 mm home, e≈0) plus an effort-clamped ~1.3 m/s head speed means every reachable strike is a
drive-through press (peak/mean force 1.1–3.25× vs the 5–20× a real collision shows). So the thesis's
"maximize impact impulse subject to a per-joint impulse bound" is, as currently built, **"maximize/bound a
press integral (torque×dwell)."** That is the load-bearing scoping fact for the whole phase.

### Q1 — How does the constraint work / get trained / why is it vacuous?
**The chain (when enabled):** `SubstepImpulseAccumulator` integrates `Λ_j = Σ|qfrc_constraint_j − baseline|·dt`
over a **25-substep (50 ms) time-based sliding window**, contact-masked to the hammer↔nail sensor
(`impulse_bound.py:166–194`) → `joint_impulse_excess` returns raw margin `c = Λ_j − IMP_J_LIMIT`
(`constraints.py:52`; caps `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]` N·m·s, joint2 doubled) → `CaT.delta_map`
maps to `δ = clamp(c/c_max)·imp_max_p`, soft-OR max over joints → `CatPPO` discounts **positive reward only**:
`r_total − δ·r_pos`, plus a `(1−δ)` value bootstrap (`cat_ppo.py:35–104`). Larger violation → more of the
strike's positive return discounted → policy pushed toward compliant strikes.

**Current state: the constraint shapes NOTHING.** `imp_max_p = 0` (`env_cfgs.py:295`) hits a hard
short-circuit (`hook.py:194–196`) → `δ_imp ≡ 0` → byte-identical to stock PPO. It is **log-only**.

**Why it's vacuous even if enabled** (two convergent lines):
- *Physics:* worst *reachable* Λ/cap ≈ 0.36 (CPU forensics). The enforced Λ is a press integral — proven by
  solref-fragility (Λ moved −38% under solref×2; a momentum-pinned ballistic integral would be solref-robust),
  a flat force plateau (peak≈mean), and contact that never releases inside the window. `kp` lives in `τ`, **not
  in `M(q)`**, so it has ~zero lever on ballistic impulse (Λ flat across 40× kp) — the physics reason VIC is deferred.
- *Empirics (g-campaign, this run's banked CSV):* worst Λ/cap is **vacuous at deployment, near-binding in
  exploration**. Deterministic (deployed) worst Λ/cap ≈ 0.16–0.20 (max 0.39); **sampled** (the surface CaT
  acts on) 0.50–0.80, one F1 seed **1.004** (just over cap). Over-cap fraction 0 across ~13k eps.
  `lambda_max_joint1` (shoulder) is the sole loaded joint everywhere; joint6 ≈ 0.01. **Fixing the parking farm
  moved the policy AWAY from binding** — a healthier, more efficient strike produces *less* joint reaction
  (nf1 sampled 0.91–1.12 → a05/g sampled 0.50–0.78). "More efficient = more vacuous, not less."

**The lever that decides bindability is the window/cap pairing, NOT training.** Caps were derived at
Δt≈27.3 ms but the window integrates 50 ms. Under the shipped pairing a *rigid-target press* reads 1.12–1.17×
cap (binds); rescaled to a formula-consistent 50 ms cap it reads 0.61–0.64× (does not bind). So whether the
metric ever binds is a **policy choice = Khadiv decision (e)**, not more seeds.

### Q2 — Is impulse maximization happening? Can we improve it?
**No, it is not happening — and the reward doesn't actually ask for it.** Every positive strike-reward channel
is *depth-gated*, and depth is hard-capped at 0.032 m with episode termination at 0.030 m. `completion`
(w=100, binary, one-shot, impulse-blind) and `nail_depth_delta` (w=600, depth-progress, impulse-blind)
dominate. `delivered_impulse` (w=2.0, the only explicit maximize term) contributes only ~1.75 reward/episode
≈ **1.7% of completion** — doubling delivered impulse adds ~1.75 more, ~50:1 against striking harder. And it's
**depth-gated** (`advanced = Δdepth > eps`, `rewards.py:259,266–269`): impulse into an already-bottomed nail
pays nothing, so it saturates the moment the nail seats. The nail ratchets → the *minimum* impulse to reach
0.030 m suffices. Two negative terms (`action_rate`, `joint_pos_limits`) actively bias toward the *gentlest*
completing strike.

**Evidence (honest spread — corrects the earlier "0.35–0.47× i_ref" shorthand):** the *terminating*
parking-fixed A arm clusters **around** i_ref (delivered_mean ≈ 0.877× i_ref; 0.46–1.23× across seeds, several
≥ 1.0×); the *non-terminating* arm taps-and-parks at 0.31–0.38× i_ref. Never *consistently* ≫ i_ref, which a
genuine maximizer would show. Qualitative conclusion (not maximized) holds; the exact "always sub-reference"
claim does not — report the spread.

**Ceiling caveat:** even with a perfect reward, on fixed impedance you can drive *more delivered ∫F·dt* only by
pressing harder/longer — that's press, not ballistic momentum. So "maximize impulse" here largely means
"maximize the press/delivered integral," which loops straight back into the Q1 press-vs-ballistic decision.

**Highest-leverage levers (map ranking):** (1) **loosen/remove the `delivered_impulse` depth-gate** so
seated-nail impulse pays — the single biggest change; (2) raise `delivered_impulse` weight and/or add a
**super-linear excess-over-i_ref bonus**; (3) **soften the completion cliff** (100→~20) *combined with* a
stronger delivered term — **NOT `no_terminate` alone** (proven worse: tap-and-park, 0.34× i_ref); (4) physics:
make the nail *resist* (frictionloss / return spring / min-impulse-per-advance) so a soft tap can't advance it —
most faithful to real impact-safety but re-derives i_ref + the whole cap calibration (entangled with Khadiv e).

---

## Part 2 — Ranked experimental agenda

Cheap-first. **P1 = local diagnostics, no GPU** (perfect for "tweak params, watch behavior, understand the
system"). **P2 = instrumentation.** **P3 = Vega reward-maximization sweeps.** **P4 = gated on Khadiv (e)/(f).**

### P1 — Local diagnostics (scriptable on the mac, no GPU, no Vega)
These directly characterize the constraint + the press-vs-ballistic knee. Most extend probes we've already run.

| # | Experiment | Knobs | Metric / what it settles |
|---|---|---|---|
| A1 | **Approach-velocity sweep** (0.5×–8× the scripted strike; extend the 1×–32× battery) | reference speed factor; effort_limit | Λ/cap, delivered ∫F·dt, peak/mean force, contact-window duration, release census vs v. **Linear-in-v + duration-independent ⇒ ballistic; flat-once-clamped + duration-driven ⇒ press.** Quantifies the press character across the *reachable* envelope. |
| A2 | **solref sensitivity re-run** (×0.5/×2 on nail_head + hammer_head contact) | contact solref | ΔΛ vs Δpeak-force. Press: tracks force (~−38%). Ballistic: solref-robust. THE discriminator. |
| A3 | **Window/cap pairing 2-point** (50 ms-window×27.3 ms-cap vs 50 ms/50 ms) on yielding **and** rigid target | event_window_substeps, IMP_J_LIMIT (compare, don't sweep) | Λ/cap under each pairing. Produces the **decision-(e) support table** (1.12–1.17× vs 0.61–0.64×). Shows bindability is a pairing choice. |
| A4 | **Force-gated / shortened window ablation** | event_window_substeps; a force-gate variant | Fraction of Λ that is ballistic vs press across reachable strikes → informs the (e) ballistic/windowed/split choice. |
| A5 | **Rigid / heavier / stiffer target probe** + whip-style direct-velocity injection | nail mass/stiffness/spring (asset); scripted velocity | Does *any* reachable regime show solref-robust (ballistic) Λ that crosses cap by *impact* not *press*? Establishes whether the constraint can EVER bind on a genuine strike. |
| A6 | **Armature sensitivity** (placeholder 0.01/0.02 vs gear²-plausible 10–30×) — Khadiv (f) | armature | Recompute m_eff (0.4→~1.4 kg) and the ballistic Λ/cap margin. Bounds the model-sensitivity of the vacuity number. Verify real Z1 rotor inertia. |

### P2 — Instrumentation (cheap code; do early so P3 is legible)
- **C1 — Training-time constraint distribution logging.** Add TB metrics for **sampled** per-episode worst
  Λ/cap quantiles, over-cap fraction, contact-window length, and a press-domination flag. Without this we
  can't *see* whether exploration approaches the cap (recall F1 sampled hit 1.004). Prereq for any
  "does it bind in exploration" claim.
- **C2 — Contact-position logging.** Rule out edge/shoulder contact within `hammer_head_0` (exploit hygiene;
  currently invisible, only proxied by invariants + renders).
- **A7 (code, no enable) — event-level `1−∏(1−δ_t)` CaT-pressure recalibration.** Make `imp_max_p`
  read-count-aware (completing ~1 read vs press ~3 reads) so enforcement is *correct when enabled*.
  `imp_max_p` **stays 0** — this is plumbing, not enabling.

### P3 — Vega reward-maximization sweeps (small seed counts; gate on P1/B0 first)
- **B0 (design, cheap) — pin the maximize objective.** Delivered object-side ∫F·dt vs pre-impact momentum
  `m_eff·v`? They diverge and wire to different sensors. Also resolve from `env.yaml` whether the g/ab1/nf1
  checkpoints were *trained* with `delivered_impulse` or only *measured* in the cat_impulse eval env — changes
  whether "w=2.0 is too weak" or "it was never in the reward."
- **B1 — `delivered_impulse` depth-gate ablation** (gate-on vs gate-off). #1-leverage change. Watch
  delivered/i_ref, does it exceed 1.0, does press-farming appear (per-event 50 ms cap should bound it).
- **B2 — `delivered_impulse` weight sweep** (2 / 20 / 60), combined with gate-off (weight alone won't help while
  the gate saturates).
- **B3 — excess-over-reference bonus** (`max(0, delivered−i_ref)²` or an uncapped depth-ungated channel) —
  encodes "beat the reference," not "match it." Needs the per-event cap watchdog (C1).
- **B4 — completion-cliff softening** (100→~20) *with* a stronger delivered term (NOT `no_terminate` alone).
  Watch success stays high **and** delivered climbs.

### P4 — Gated / do-not-do-unilaterally
- **B5 — nail-resistance physics lever** (frictionloss↑ / return spring / min-impulse-per-advance): most
  faithful to real impact-safety, but re-derives i_ref and re-pins IMP_J_LIMIT → entangled with Khadiv (e). Do
  after P3.
- **Enforcement (`imp_max_p > 0`) E0-vs-E1:** a *guaranteed no-op* at the shipped window/cap on a healthy
  policy (over-cap 0). Blocked on **Khadiv decision (e)** (ballistic vs windowed-press vs split) + (f) (armature)
  + the A7 recalibration. **Do not raise `imp_max_p`.**
- **IMP_J_LIMIT tightening to force a bind:** forbidden (user decision) — destroys the hardware-provenance
  defense.

---

## Part 3 — Housekeeping the maps surfaced (cheap, do opportunistically)
- Campaign plan `docs/results/2026-07-15_vega_campaign_plan.md`: update F2 **5/6 → 7/8** (line ~182, pre-backfill
  snapshot); annotate the line-76 "constraint binds on real strikes" claim as **superseded by line 156** (it was
  true only of the parked nf1 *sampled* rollout, not the healthy policy).
- Stale line cites: `nail_driven` weight is now `hammer_env_cfg.py:177` (not :172, comment block inserted);
  i_ref is `env_cfgs.py:348` (not :347). Fix in CLAUDE.md + docs; verify by symbol, not line.
- Note that Option B's disqualifier (83.8×/51.6× i_ref multi-strike) is a **training-env-only** signal — the ab1
  CSV row for B looks clean because play_cfg terminates B at its first strike.
- Provenance: a05/g rows record `git_hash=b2ed6ee` but the 0.5 weight came via CLI override living only in
  `env.yaml` (tree now has 0.5 baked). Reproduction must read `env.yaml`, not just the hash.

---

## Part 4 — Doc/code contradictions to keep straight
- `2026-07-12_impulse_vacuity.md` §4–5 ("robot-side accumulator has no press cap, add one") is **STALE** since
  581578d (2026-07-13): the enforced accumulator is now a 25-substep sliding window that saturates at
  F̄·window·dt. Only the *log-only* `ContactRowImpulseAccumulator` remains uncapped (feeds nothing).
- `IMPULSE_CAT_IMPL_PLAN.md` §1 idealizes the bounded quantity as `Σ|qfrc_contact|·dt` (clean efc-row
  projection); the **shipped** enforced quantity is sensor-gated + baseline-subtracted `qfrc_constraint`
  (all rows), an approximation. The clean efc-row version exists only as diagnostic ground truth.
- qfrc contamination is **dof-friction (~45%), not a weld** — training scene has neq=0. Any doc still naming a
  weld share is wrong for the training scene.

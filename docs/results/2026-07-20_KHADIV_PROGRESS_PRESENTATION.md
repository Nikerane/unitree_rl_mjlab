# Progress presentation content — Prof. Khadiv, 2026-07-20

**Purpose of this file:** slide-by-slide CONTENT for the student to build into actual slides (not a
finished deck). Every number is quoted from banked eval CSVs / diagnostic docs; sources are cited per
slide. Claims are tagged **PROVEN** / **OPEN** / **CORRECTED THIS WEEK** — keep those tags visible in
the actual talk, they are the point of this update.

**Sources:** `2026-07-20_B2_effort_ceiling.md`, `2026-07-17_phase0_diagnostics.md`,
`assets/2026-07-17_depth_gate_sweep/RESULT.md`, `2026-07-19_reward_code_fault_audit.md`,
`2026-07-20_vega_experiments_manifest.md`, `2026-07-20_CODEX_HANDOVER.md`, `docs/thesis/README.md`.

---

## Slide 1 — Title / context

- **Impact-safe contact-rich manipulation on the Unitree Z1** — hammer-a-nail task.
- Thesis end goal: variable-impedance action space + explicit per-joint impulse constraint.
- **This month's scope, by direction (2026-06-17):** build and validate the full machinery
  **fixed-impedance first** on the Z1 — soft CaT, the substep impulse constraint, then variable
  impedance last. Not the G1 (optional future replication, not the focus).
- Where we are: the fixed-impedance phase is essentially done — one clean result, one important
  correction to last week's conclusion, and a set of decisions that need your judgment.

*Speaker note:* frame this as "fixed-impedance chapter close-out," not "final results" — VIC hasn't
started yet.

---

## Slide 2 — The goal + the three machinery pieces

- **Piece 1 — Soft CaT (constraints-as-terminations):** probabilistic termination ∝ violation,
  `reward = r_total − δ·r_pos`. Scale-free, manager-based, ported faithfully (γ(1−δ) bootstrap).
- **Piece 2 — Per-joint impulse constraint (Λ_j):** substep-accumulated, contact-masked, the
  thesis's headline safety mechanism — bounds impact reaction at each joint, not just velocity.
- **Piece 3 — Variable impedance (deferred):** policy commands per-joint stiffness via `set_gains`.
  **Not started.** Everything in this talk is fixed-gain (kp=1000/1500, kd=100/150, fixed).
- Sequencing rationale: get fixed-impedance right (soft CaT works? impulse constraint measures the
  right thing?) before adding a second controllable axis on top of it.

---

## Slide 3 — Task + action space, plainly

- **Task:** Z1 arm (6 DoF) drives a 7 g nail into a wood block. Success = depth ≥ 30 mm (physical
  stop at 32 mm). Episode **terminates on success** — this is a single-strike task, not repetitive
  hammering.
- **Action space:** position-only Cartesian DiffIK on the hammer head, 3-D delta-position,
  `delta_pos_scale = 0.15`, PD gains **fixed** (kp/kd constants, effort limits 30 N·m / 60 N·m on
  joint 2). The policy never commands stiffness — that is "fixed impedance."
- **Reward:** 8 terms — approach, nail_driven, nail_depth_delta, impact_progress, completion,
  action_rate, joint_pos_limits, delivered_impulse. Live weights read from code, not any spec doc.
- **Constraint:** soft-CaT on per-joint Λ_j, currently **log-only** (measures, does not yet act).

---

## Slide 3a — Reward structure: what each term is and WHY (arguably the crux)

Eight terms, in three roles. Weights are the live values from code (not any spec).

| term | weight | role | why it exists |
|---|---|---|---|
| `completion` | **100** | **task signal** | the actual objective: one-shot bonus when the nail crosses 30 mm. Not shaping. |
| `nail_depth_delta` | **600** | progress shaping | `max(0, depth − max_so_far)` — a dense gradient from 0 mm, where the Gaussian below is ~flat. The main "drive the nail" signal (cumulative ~17 over a full drive). |
| `nail_driven` | **0.5** | progress shaping | Gaussian on depth — extra pull once the nail moves. *(Was 2.0 → a reward-hacking farm; see next slide.)* |
| `approach` | 0.1 | progress shaping | weak pull of the head toward the nail (std 8 cm); low so hovering is never optimal. |
| `impact_progress` | **8** | impact objective | ante-impact *axial velocity* on a fresh, nail-advancing contact. On a position-only action space the controllable impact lever is **momentum**, not force — so we reward pre-impact speed. Literature-motivated. |
| `delivered_impulse` | **2.0** | **impact objective** | the thesis's headline: object-side delivered impulse ∫F·dt normalized by a reference strike. The explicit "maximize impact" term. *(Only on the constraint arm.)* |
| `action_rate` | −0.01 | regularizer | smoothness. |
| `joint_pos_limits` | −10 | regularizer/safety | soft joint-limit penalty. |

- **Design philosophy — "augment, not replace":** we add a term only when a specific failure mode is
  observed, rather than deriving the stack from first principles. Pragmatic, but see the honest caveats next.

*Speaker note:* frame this as "here is the reward, and here is exactly why each piece is present — I
expect you'll want to push on this."

---

## Slide 3b — Reward design: the honest tensions (please critique)

- **The maximize-objective is out-competed by its own task signal.** `completion` (100, one-shot) dwarfs
  `delivered_impulse` (2.0) by roughly **50:1** in per-episode return. So the policy is rewarded far more
  for *finishing* than for *hitting hard* — it does the minimum impulse that completes. **This is a prime
  suspect for why impulse maximization stalled** (Slide 10), and it is a reward-structure choice, not a law.
- **Shaping terms can become exploits.** `nail_driven` at 2.0 created the parking farm (Slide 5) — a dense
  per-step reward on a ratcheting quantity that paid more to *not* finish. Lesson baked in: dense shaping on
  an irreversible state is dangerous; we now audit for it.
- **The stack is progress-heavy.** Three terms (`nail_depth_delta`, `nail_driven`, `approach`) all just
  "drive the nail." Is that over-shaped? Could the task be carried by `completion` + `impact_progress` +
  `delivered_impulse` alone? Open to simplification.
- **It is not derived from the formal objective.** The thesis objective is *maximize impact impulse subject
  to a per-joint impulse bound*; the shipped reward is a hand-tuned shaping stack that approximates
  "complete the task, gently." Whether to re-derive it around the true objective (e.g. reward delivered
  impulse beyond reference, soften the completion cliff) is an open design decision.
- **PROVEN:** the current stack trains a clean, successful single strike. **OPEN:** whether its *balance*
  (completion ≫ impulse) is the right one for the thesis's actual goal.

*Speaker note:* this is the slide most likely to generate the feedback you want — lead the discussion here.

---

## Slide 4 — Headline result: it works, cleanly (PROVEN)

**Best policy (`af1`, audit-fixed code, log-only constraint):**

| metric | value |
|---|---|
| success rate | **1.00** (3/3 seeds) |
| nail depth | **32.0 mm** (physical stop, all seeds) |
| episode length | **7.3 control steps** — one clean strike, not a multi-tap grind |
| delivered impulse | **0.90× i_ref** (sampled mean; seed range 0.64–1.21×) |
| worst-joint Λ/cap | **0.53** (sampled mean; seed range 0.31–0.64) |
| safety invariants | clean — `impossible_success=0`, `lambda_dead=0` |

- This is on **validated, audit-clean code** with correct git provenance (`git_hash=6b447bf`, no
  dirty tree).
- Comparable healthy arms (`ab1`/`dg`) land in the same neighborhood: success 1.00, depth 32 mm,
  delivered 0.76–0.79×, Λ/cap 0.60–0.67 — af1 is not a cherry-picked outlier.
- **Caveat (n=3 seeds):** our own eval protocol says n=3 is under-powered (seed spread on af1 alone
  is 0.56–1.25× on delivered). Treat the *shape* of the result as solid, the exact mean as noisy.

---

## Slide 5 — How we got here: the nf1 → af1 arc

- **Before (`nf1`, pre-fix):** success **0.54**, depth 28.8 mm, delivered 0.39× i_ref, Λ/cap **0.91–1.12
  (some seeds OVER cap)**, with **5 of 6 seeds running the full 200-step episode cap** (parked). The policy
  **parked itself just below the success threshold** and farmed a dense reward loophole instead of finishing.
- **Root cause:** `nail_driven` (a per-step Gaussian reward on depth, weight 2.0) paid *more* to
  hold the nail at ~28 mm forever than to cross 30 mm and terminate the episode — a classic
  non-terminal-state-pays-more-than-terminal bug.
- **Fix:** cut `nail_driven` weight 2.0 → 0.5 (2026-07-16).
- **After (`af1`):** success **1.00**, clean 7-step single strike, delivered 0.90×, Λ/cap 0.53.
- This is the concrete "before/after" evidence that the machinery — reward design, CaT hook,
  impulse instrumentation — is wired correctly end to end.

*Speaker note:* this is the "look, it works" slide — lead with it, then immediately pivot to the
honest caveats in the following slides.

---

## Slide 6 — Q1: Are the constraints being respected? (honest, with a caveat)

**Short answer: yes, but "respected" currently means "never challenged," not "enforcement kept it
safe."**

- The per-joint impulse constraint is **log-only** — `imp_max_p = 0`, termination probability δ ≡ 0.
  It measures Λ_j every substep but applies zero pressure. It has never once affected training.
- It is also **vacuous** on fixed impedance at the shipped policy: worst-case Λ/cap sits at 0.53,
  never approaching 1.0. Over-cap fraction is **0 across ~13,000 evaluated episodes**.
- **Important nuance (honest):** the *pre-fix parked* policy (`nf1`) actually reached Λ/cap **0.91–1.12
  (over cap on some seeds)** — because parking *presses* the nail, generating sustained joint reaction.
  Fixing the reward moved the policy to an efficient single strike that *doesn't* press — so it dropped to
  0.53. **Fixing the parking farm made the constraint MORE vacuous, not less.** The cap is real; the
  healthy behaviour simply doesn't approach it (the pressing pathology did).
- So: nothing has ever tested whether the constraint mechanism *would* keep the arm safe under
  pressure, on this quantity — on a *healthy* policy.
- **Contrast — the velocity soft-CaT DID get tested and DID work** (earlier campaign): mean peak
  |q̇| dropped 3.54 → 2.5 rad/s under enforcement, with the strike preserved. The mechanism enforces
  when it bites. The impulse constraint just hasn't bitten yet.

**PROVEN:** CaT enforcement works (velocity case). **OPEN:** whether/how it would behave under
impulse enforcement — untested because the quantity never approaches its cap.

---

## Slide 7 — Physics finding: what Λ actually measures (PROVEN)

- Question: is the enforced Λ a **ballistic momentum transfer** (the impact-safety quantity we
  want) or a **sustained press-reaction integral** (a different, less interesting quantity)?
- **Test:** double the contact solver stiffness (`solref×2`) and rerun the identical reference
  strike. A true ballistic momentum quantity is solver-*insensitive*; a press integral is not.
- **Result:** Λ drops **−76.3%**, delivered impulse drops **−76.6%**, in near-perfect lockstep.
  Peak force drops only −42.6%, contact window shrinks −13.6%.
- **Conclusion: the enforced Λ is a press integral, not a ballistic impact quantity.** It measures
  how hard/long the arm pushes through contact, not the momentum delivered at the instant of
  impact.
- Corroborating evidence: commanding the arm 8× harder never raises contact speed past ~1.4 m/s
  (effort-clamped) and Λ stays non-monotonic, never exceeding 0.36 of cap — the signature of a
  drive-through press, not a collision (a real collision shows 5–20× peak/mean force ratio; this
  shows 1.4–1.7×, with zero rebound).

**PROVEN** (`2026-07-17_phase0_diagnostics.md`, diagnostics A1/A2).

---

## Slide 8 — Physics finding: when does it even bind? (decision-support table)

On a **rigid** target (nail locked, isolating the window/cap question from nail softness):

| quantity | shipped caps (27.3 ms basis) | formula-consistent caps (50 ms basis) | binds? |
|---|---|---|---|
| impact-only (9 substeps, genuine ballistic) | 0.61× | 0.33× | **no** |
| accumulated (shipped 50 ms window = what's enforced) | **1.18×** | 0.65× | shipped: yes / formula: no |

- The constraint only "binds" at the intersection of three things: a **rigid** target (not our soft
  7 g nail), **press-through** contact (not a clean impact), and the **shipped cap/window
  time-basis mismatch** (caps derived at 27.3 ms, window integrates 50 ms). Change any one of the
  three and it stops binding.
- On the real (soft, yielding) nail: Λ/cap sits at **0.19–0.36×**, nowhere close.
- Model-uncertainty check: even inflating simulated arm inertia 30× (implausibly high), the
  realistic ballistic impulse is still only **0.50× cap**.

**PROVEN.** This table is the evidence base for decision (e) on Slide 12.

---

## Slide 9 — Q2: Was using CaT sensible, or should we pivot? (honest, balanced)

- **CaT itself is sound and it demonstrably works** — it's scale-free, manager-based, ported
  faithfully, and it enforced the velocity constraint successfully (Slide 6).
- **For the impulse constraint specifically, CaT's value is currently unproven** — not because the
  mechanism is broken, but because the *quantity it's asked to bound is vacuous* (Slides 7–8). A
  mechanism can't demonstrate anything on a constraint that's never approached.
- **This is a task/quantity problem, not a CaT defect.** The open question is *what Λ should
  bound*: a ballistic momentum jump (clean semantics, but provably vacuous on this soft nail), a
  windowed press-reaction (binds, but only under a specific, somewhat arbitrary cap/window
  pairing), or a split of the two.
- A CMDP/Lagrangian formulation is a documented alternative, but it has a category mismatch for a
  per-step worst-case bound (it constrains the expected discounted *sum*, not the *max*) — so it's
  not obviously better-suited here.
- **My recommendation frame:** keep CaT as the mechanism; the real decision is (a) what Λ should
  measure, and (b) whether to make the *task* demand more (so a healthy strike approaches hardware
  caps) rather than accept the vacuity.

**PROVEN:** CaT mechanism works when tested. **OPEN:** whether it's the right mechanism *for this
quantity*, pending decision (e).

---

## Slide 10 — Q3: Was impulse maximization sensible? (this week's correction — read carefully)

- **Goal was sound:** push delivered impulse toward/past the reference under reward shaping alone,
  before concluding we need variable impedance.
- **What we tried:** removing the delivered-impulse depth-gate, and tripling then 6×-ing the
  ante-impact-velocity reward weight (`impact_progress`).
- **Result: delivered impulse plateaued at ~0.87–0.90× i_ref regardless** — depth-gate removal was
  a statistical no-op; 3× the velocity weight was flat (0.86× vs 0.87×); 6× **destabilized training**
  (3/3 seeds diverged).
- **My conclusion last week: this is a hardware joint-velocity wall** (real Z1 limit 3.1415 rad/s;
  the shipped policy already runs the dominant joint to 2.56 rad/s, ~81% of that limit).
- **⚠ CORRECTED THIS WEEK, by an independent re-evaluation (Codex):**
  1. One of my diagnostic probes (`effort_sweep.py`) had a **shared-state bug** — it mutated a
     module-level object across sweep steps, which faked a perfect "more torque changes nothing"
     result. Rerun in isolated processes: torque *does* matter a little (+7.5% contact speed at
     2× effort), and the baseline strike is genuinely torque-saturated at 3 of 6 joints.
  2. **The ~1.4 m/s ceiling is trajectory-specific, not a hardware wall.** Head speed is a function
     of *how* the joints move together, not just how fast each one spins. At the *same* per-joint
     velocity budget (3.1415 rad/s), a coordinated multi-joint **whip** trajectory reaches
     **≈4.2 m/s** head speed — our straight-down strike uses only **~45%** of that envelope.
- **So: impulse maximization is reopened.** A whip-capable fixed-impedance policy might strike
  materially harder within the *existing* hardware limits, no variable impedance required — that
  needs to be tested before VIC is declared necessary.
- **What still survives the correction:** the soft 7 g nail still caps how much of any given
  momentum becomes *bounded, delivered* impulse (Slides 7–8) — that finding is robust to the
  velocity correction.

**CORRECTED THIS WEEK.** This is the most important thing to flag honestly in this talk.

---

## Slide 11 — What we solved this month (timeline)

1. **Root-caused and fixed a reward "parking farm"** — policy was parking below success to farm a
   dense-reward loophole; success 54% → 100% (Slide 5).
2. **Characterized the impulse constraint's physics** — proved the enforced Λ is a press integral,
   not ballistic momentum (−76% solref sensitivity test, Slide 7).
3. **Mapped the constraint's vacuity** and produced the decision-support table for what Λ should
   bound (Slide 8).
4. **Ran a full reward-code fault audit** (two independent reviewers) — 11 latent-bug findings, 6
   fixed, including closing a reward-farm exploit worth **12.9× reference impulse** on a
   locked-nail press probe. All fixes GPU-validated as behavior-neutral on the healthy policy.
5. **Hardened experiment provenance** — git-based deploy only (no more `scp`), dirty-tree detection
   on every eval row, after a stale-hash incident this exact discipline was built to prevent.
6. **Diagnosed, then corrected, the impact-ceiling conclusion** — see Slide 10. Caught our own
   mistake before it went into a decision.

---

## Slide 12 — Rigor and provenance (this is deliberately un-flashy)

- **Reward audit:** two independent reviewers (myself + an AI coding agent used as an adversarial
  second opinion), cross-verified against code. 11 findings, ranked by severity; 6 fixed and GPU
  re-validated as behavior-neutral (`af1`), 5 deferred with explicit "must fix before enforcement"
  flags (time-basis mismatch on the caps; multi-read enforcement asymmetry; etc.).
- **The most serious finding (F1):** the delivered-impulse reward could be farmed by pressing on a
  locked/ratcheted nail — proved with a discriminating unit test, not just "the training curve
  looks fine" (the healthy policy never presses, so the curve alone can't distinguish "fixed" from
  "never triggered").
- **Provenance:** every eval CSV row now stamps a git hash, tagged `-dirty` if the tree wasn't
  clean at eval time — a stale-hash incident earlier this month is exactly why this exists now.
- **Guardrails I'm holding myself to:** impulse caps stay at hardware/manufacturer-derived values,
  never tuned to manufacture a binding result; `imp_max_p` stays 0 until you weigh in on decision
  (e); no VIC/`set_gains` code without your explicit sign-off.

*Speaker note:* this slide is evidence of process discipline, not a result — include it because it's
honest about how much of the month went into finding and fixing our own bugs, not just training runs.

---

## Slide 13 — Open decisions for you (Khadiv decision points)

- **(e) — What should Λ bound?** Three options, with the Slide 8 table as evidence:
  1. **Ballistic impulse** — clean "impact-safe" semantics, but *provably* vacuous (≤0.61× cap even
     on a rigid target at reachable speed; 0.19–0.39× on the real nail).
  2. **Windowed-press reaction** (what's currently shipped) — binds, but only under a specific
     cap/window pairing, and reduces to "average torque × window," which invites "why not just
     bound torque directly?"
  3. **Split** — separate ballistic and press/torque constraints.
- **CaT vs. CMDP/Lagrangian** — my recommendation is keep CaT (it demonstrably works when tested,
  and a Lagrangian has a category mismatch for a per-step worst-case bound) — but this is worth
  your explicit sign-off, not just my inference.
- **Make the task harder to legitimately bind the constraint, vs. accept vacuity honestly** —
  e.g., a stiffer/heavier/spring-loaded nail could make a *healthy* strike approach the cap without
  touching the cap itself. Untested; a real option.
- **Whip-maximization-first vs. go straight to VIC** — given this week's correction (Slide 10), do
  we spend time testing whether a whip-capable fixed-impedance policy closes the impulse gap before
  starting variable impedance, or is VIC's motivation (decoupling contact speed from steady-state
  joint velocity) strong enough to start now regardless?
- **Reward re-balance (Slides 3a–3b)** — should we re-derive the reward around the formal objective
  (maximize impulse s.t. the bound) rather than the current "complete-the-task-gently" shaping stack?
  Concretely: soften the `completion` cliff and/or reward delivered impulse *beyond* reference so the
  50:1 imbalance stops suppressing the maximize term — coupled carefully to avoid re-opening a farm.
  This may matter more than any single experiment, since the reward defines what "success" even means.
- **Hard rule I'm holding regardless of your answer:** the impulse caps stay at manufacturer/
  hardware-derived values. Whatever we decide, we don't tune the cap to make a story work.

---

## Slide 14 — Proposed next steps

- **Immediate (before touching VIC):** a CPU-only trajectory optimization test — can a
  whip-shaped strike, under the *same* hardware constraints (30/60 N·m, 3.1415 rad/s per joint),
  actually reach the ~4.2 m/s kinematic opportunity in a real dynamic rollout, or does it collapse
  back down once controller dynamics are accounted for? This directly resolves the Slide 10
  reopening, cheaply, before committing to VIC.
- **In parallel:** the harder-target frontier sweep (Slide 13, option 3) — check whether a modified
  nail (mass/friction/spring) creates a *legitimately* binding constraint without touching the caps.
- **Enforcement-readiness housekeeping** (not turning enforcement on yet): fix the two deferred
  audit findings that block it — the cap time-basis mismatch and the multi-read enforcement
  asymmetry — so that whenever decision (e) lands, we can flip `imp_max_p` on correctly the same
  day, rather than discovering a miscalibration after the fact.
- **Then:** whichever of whip-maximization or VIC your decision on Slide 13 points to.

---

## Slide 15 — What I'm asking you

- Which of the three options on decision (e) — ballistic / windowed-press / split — matches what
  you want the thesis to claim "impact-safe" means?
- Are you comfortable with CaT as the enforcement mechanism, given it's proven on velocity but
  unproven on impulse (because the impulse quantity has never bound)?
- Should I spend the next 1–2 weeks on the whip-maximization test before starting VIC, or is the
  VIC motivation strong enough already to start now in parallel?
- Is a harder/modified nail target (to make the constraint bind legitimately) in scope, or does
  changing the task risk diluting the thesis's grounding in the real hardware target?

---

## Appendix — exact numbers + caveats

**Best policy (af1), 3 seeds, `git_hash=6b447bf`, `imp_max_p=0`:**
- success: 1.00 / 1.00 / 1.00 · nail depth: 32.0 / 32.0 / 32.0 mm · ep_len: 8 / 7 / 7 steps
- delivered/i_ref (sampled): 1.21× / 0.87× / 0.64× → mean **0.90×**
- worst Λ/cap (sampled): 0.64 / 0.64 / 0.31 → mean **0.53**
- invariants: `impossible_success=0`, `lambda_dead=0` all seeds

**Progress arc:**
- nf1 (pre-fix): success 0.54, depth 28.8 mm, delivered 0.39×, Λ/cap 0.98, ep_len 168
- af1 (post-fix + audit): success 1.00, depth 32.0 mm, delivered 0.90×, Λ/cap 0.53, ep_len 7.3

**Physics (Phase 0, CPU, no GPU):**
- solref×2 sensitivity: Λ −76.3%, delivered −76.6%, peak force −42.6%, window −13.6%
- rigid-target window/cap table: impact-only 0.61× (shipped) / 0.33× (formula); accumulated 1.18×
  (shipped) / 0.65× (formula)
- armature ×30 (implausibly high): realistic ballistic still only 0.50× cap

**Ceiling correction (B.2, CPU):**
- superseded claim: "hardware joint-velocity wall, ~1.4 m/s ceiling, not reward-limited, not
  torque-limited" — **overturned this week**
- corrected: effort ablation had a shared-state probe bug (torque is a small, real factor, not an
  exact no-op); whip trajectory reaches ≈4.2 m/s at the same joint-velocity budget the straight-down
  strike only uses ~45% of

**Known caveats to say out loud:**
- n=3 seeds per arm throughout this update; our own protocol wants ≥5 for a real distributional
  claim — treat exact means as indicative, not final.
- The delivered-impulse and Λ/cap numbers have wide seed spread (0.64–1.21× and 0.31–0.64×
  respectively on af1 alone) — the mean is a summary, not a tight estimate.
- The impulse constraint has been log-only for the entire month; every number above describes
  *measurement*, not *enforcement* behavior.
- The whip-maximization reopening (Slide 10) is a CPU physics argument for *reachability*, not yet
  a trained policy result — the decisive experiment hasn't been run yet.

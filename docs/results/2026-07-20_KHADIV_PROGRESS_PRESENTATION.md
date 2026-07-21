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

Eight terms. Weights are the live values from code (not any spec). The **functional form** of each term
is a deliberate choice — the middle column is the "why this shape," which is where most of the intuition lives.

| term (w) | functional form | intuition — why this shape |
|---|---|---|
| `completion` (**100**) | **sparse one-shot** `1[depth ≥ 30 mm]` | the TRUE objective. Sparse rewards the *outcome*, not a prescribed *way* to strike — no shaping bias on the motion. |
| `nail_depth_delta` (**600**) | **increment** `max(0, depth − max_so_far)` | reward only *new* progress → monotone, **cannot be farmed by holding**, and dense from 0 mm (where a distance-Gaussian is ~flat). The workhorse drive signal. |
| `nail_driven` (**0.5**) | **Gaussian on depth**, centered at goal | a smooth, bounded [0,1] attractor pulling depth toward the goal. Intuitive — but a Gaussian pays near-max *while parked near the goal*, which is exactly what **farmed** at weight 2.0. |
| `approach` (0.1) | **Gaussian on head→nail distance** (std 8 cm) | a soft, saturating "get close" guide; bounded so it can't dominate, weak so hovering-near is never a stable optimum. |
| `impact_progress` (**8**) | **gated linear velocity** `(v_axial/v̄)·1[first-contact]·1[advanced]` | on a position-only action the impact lever is **momentum (m_eff·v)**, so reward pre-impact *speed* — but only on a real, nail-advancing strike (double-gated → unfarmable by scraping/tapping). |
| `delivered_impulse` (**2.0**) | **normalized increment** `Δ(∫F·dt)/i_ref` | the thesis objective, measured object-side. Increment form (like nail_depth_delta) → monotone + farm-resistant; normalized by a reference strike so the weight is interpretable. *(constraint arm only)* |
| `action_rate` (−0.01) | **quadratic penalty** `−‖Δa‖²` | standard smoothness regularizer — discourage jerky commands. |
| `joint_pos_limits` (−10) | **barrier penalty** near travel limits | soft safety wall keeping joints off their mechanical limits. |

- **The core design intuition — attractors vs. increments.** Distance/goal **Gaussians** (approach,
  nail_driven) are intuitive smooth attractors, but they pay at *non-terminal* states, so on a ratcheting /
  irreversible quantity they can be **farmed by parking near the peak** (precisely the nail_driven bug).
  **Increment / delta** terms (nail_depth_delta, delivered_impulse) reward only *new* progress → monotone,
  farm-resistant, and dense from zero. **Sparse** (completion) is reserved for the true objective so we don't
  bias *how* the strike is done. Lesson baked in: prefer increments for the drive + objective, use Gaussians
  only as weak bounded guides.
- **Philosophy — "augment, not replace":** add a term only when a specific failure mode appears, rather than
  deriving the stack from first principles. Pragmatic, but see the honest caveats next.

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
  is 0.56–1.25× deterministic / 0.64–1.21× sampled on delivered). Treat the *shape* of the result as solid, the exact mean as noisy.

---

## Slide 4a — Scorecard: the three thesis quantities in one view (honest)

The thesis has three goals: **(1)** do the task, **(2)** maximize impact impulse, **(3)** respect the
per-joint impulse constraint. For the best policy `af1`:

| thesis quantity | target | achieved (af1) | honest status |
|---|---|---|---|
| **1. Task** — drive the nail | depth ≥ 30 mm | 32 mm, **100% success**, clean single strike (7 steps) | ✅ **SOLVED** |
| **2. Impact maximization** — maximize delivered impulse | as high as possible | **0.90× i_ref** (sampled) — *matches* a scripted reference strike, does **not** exceed it | ⚠️ **MATCHED, not maximized** |
| **3. Constraint** — Λ_j ≤ L_j ∀ joints | worst Λ/cap < 1 | worst **0.53** (sampled) / **0.17** (deployed); all 6 joints under; over-cap fraction **0** / ~13k episodes | ✅ **SATISFIED — but unforced + vacuous** |

**Per-joint Λ/cap (deployed policy):** j1 (shoulder) **0.17**, j2 0.08, j3 0.11, j4 0.08, j5 0.10, j6 0.01
— the shoulder carries the most reaction, every joint sits far under its cap.

- **The honest one-liner:** *the task is solved and the constraint is satisfied with large margin — but the
  constraint is satisfied **trivially** (it is log-only and the policy never approaches it), and the impulse
  is **matched** to a reference strike rather than **maximized** beyond it.*
- So of the three goals, **1 is genuinely done**, and **2 and 3 are "achieved" only in a weak sense** — which
  is exactly why the open decisions (Slides 10, 13) matter: maximization needs a better trajectory/reward,
  and the constraint needs to either bind legitimately or be honestly reported as non-binding.

*Speaker note:* this is the single most honest slide — put the "matched not maximized" and "satisfied but
vacuous" language up front so the rest of the talk reads as "here's why, and here's what we do about it."

---

## Slide 5 — How we got here: the nf1 → af1 arc

- **Before (`nf1`, pre-fix):** success **0.54**, depth 28.8 mm, delivered 0.39× i_ref, Λ/cap **0.91–1.12
  (some seeds OVER cap)**, with **5 of 6 seeds running the full 200-step episode cap** (parked; nf1 predates the current 1000-step / 20 s episode config). The policy
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

## Slide 6b — How we measure the impulse (the instrument, before the finding)

Before any claim about the constraint, this is *how* Λ is measured — because the measurement choices are
where the honesty lives.

**What we integrate.** Per arm joint j, over a 50 ms sliding window at the 2 ms physics substep:
$$\Lambda_j=\textstyle\sum_{\text{window}}\underbrace{\mathbf 1[\text{hammer–nail contact}]}_{\text{contact-mask}}\cdot\big|\,\underbrace{Q_{\text{qfrc},j}-B_j}_{\text{baseline-subtracted}}\,\big|\cdot\,\Delta t_{\text{phys}}$$

**What `qfrc_constraint` is.** MuJoCo solves contact as a convex optimization (Gauss's principle) and returns
`qfrc_constraint = efc_Jᵀ·efc_force` — the solved constraint reaction projected into joint space. Two facts
make this the right instrument:
- It gives **exact per-joint attribution *with* chain coupling** — the reaction at *every* joint from a
  contact at the hammer face, impossible to get on hardware for free. This is why we use the simulator's
  internal solver quantity, not an external estimate.
- MuJoCo reports it in **force units = impulse ÷ h**, so `Σ|qfrc|·Δt` reconstructs the per-joint **impulse**
  exactly — the quantity the solver natively works in.

**The catch, and the two masks that fix it.** Raw `qfrc_constraint` is an *aggregate* over **all** constraint
rows — contact **plus dof-friction plus joint-limits** (no per-type split in the field). On this scene the
contaminant is **dof-friction (~41–48%, end-effector-dependent) + joint limits — NOT a weld** (the training
env has `neq=0`, zero equality constraints; the weld exists only in the standalone viewer scene). So we (1)
**contact-mask** (integrate only substeps where the hammer–nail contact sensor fires) and (2)
**baseline-subtract** (freeze the pre-contact off-contact reaction and subtract it), isolating the contact-caused reaction.

**Three quantities, cross-checked (defense against "your number is a solver artifact"):**
| quantity | what it is | role |
|---|---|---|
| **robot-side Λ** (`qfrc_constraint`) | per-joint solved reaction, chain-coupled | **the enforced constraint** |
| **object-side ∫F·dt** | net contact force on the *nail* — friction/weld-free *by construction* | clean cross-check (= i_ref 0.6094 N·s) |
| **ContactRow** (`efc` rows) | isolates *only* the hammer↔nail rows (Jᵀf) | log-only cleanest per-joint attribution |

**The honest caveat we state out loud** (this is the bridge to Slide 7): the contact *force profile* is a
modeling choice (`solref`/`solimp`), so the impulse integral is momentum-pinned **only for a completed
ballistic event released inside the window**. Our slow position-controlled reference is a *protocol-truncated
press*, so Λ is **solref-fragile** — we report `solref`/`solimp`/`timestep` as provenance, and the object-side
∫F·dt moving in **lockstep** (−76%) proves the shift is *physical*, not an instrument bug.

**Sim-to-real:** the same per-joint reaction is recoverable on the real Z1 with a **generalized-momentum
external-torque observer** (De Luca/Haddadin) — no joint-torque sensors needed — so the sim quantity has a
standard hardware counterpart. Company we keep: Kang 2025 (terminate on τ_load), Ma 2025 (bound arm current),
CaT (bound foot force).

*Speaker note:* the one line to say — *"we don't bound the raw solver force (a modeling artifact); we bound the
contact-masked, baseline-subtracted per-joint impulse, and we cross-check it against the friction-free
object-side integral — which is exactly what an external-torque observer would measure on the real arm."*

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
- Corroborating evidence: commanding *this straight-down strike* 8× harder never raises contact speed
  past ~1.4 m/s, and Λ stays non-monotonic, never exceeding 0.36 of cap — the signature of a
  drive-through press, not a collision (a real collision shows 5–20× peak/mean force ratio; this
  shows 1.4–1.7×, with zero rebound).
- *(Note for consistency with Slide 10: the ~1.4 m/s clamp is specific to the straight-down trajectory,
  NOT a torque/effort limit — that attribution was corrected this week. The press character here does
  not depend on it; it rests on the solref −76% lockstep + the peak/mean force ratio + zero rebound.)*

**PROVEN** (`2026-07-17_phase0_diagnostics.md`, diagnostics A1/A2). The press-vs-ballistic conclusion is
robust; only the *cause* of the straight-down speed clamp was reattributed (Slide 10).

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

**PROVEN.** This table is the evidence base for decision (e) on Slide 13.

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

## Slide 10c — The definitive answer: a causal dissection of the maximization reward (NEW, strongest result)

We trained a **controlled ablation** (24 policies, reward weight 0→24×, GPU on Vega) and validated it over
**1,140 rollouts** with domain-varied resets. It settles the impulse-maximization question:

- **The forward-swing is a *learned* maneuver caused by the impact-max reward** — not kinematic, not the
  imitation prior. Turn the reward off → every seed strikes straight (90% of rollouts); turn it to 24× →
  it swings (93%). Robust across initial conditions. *(`mx_dose_response.png`, `before_after_trajectories.png`.)*
- **The two reward terms drive DISSOCIABLE strategies** (`dc_decomposition_box.png`): `impact_progress`
  (speed reward) → a wind-up **swing**; `delivered_impulse` (impulse reward) → a straight **hard press**
  (highest joint force, 43 N). Different-looking, same underlying lever.
- **The essential fact:** under fixed impedance, delivered impulse is `∫F·dt` against a nail that stays put,
  so the reward can only buy **contact *time*, not contact *speed*.** We paid the policy **24× to strike
  faster and it went *no faster*** (speed flat ~1.4 m/s); a regression over all 1,140 rollouts shows two
  scalars — dwell + force — predict delivered impulse (**R²=0.84**), while speed and the dramatic swing add
  ≈0. `corr(speed, delivered) = −0.26`.
- **Longer training doesn't help** — 3× iterations *destabilized* (a seed collapsed to 0% success), no ceiling
  break. *(Caveat: that's specific to our current no-DR/no-curriculum recipe — with DR + curriculum, longer
  training returns.)*

**What this means for the thesis (honest, and it tightens Slide 10):** on fixed impedance the maximization
objective **structurally collapses onto contact duration** — RL never found the whip (fastest strike ever,
across every config, was 1.81 m/s vs the ~4.2 m/s kinematic opportunity). **But "RL didn't find it" ≠ "it
can't be found"** — the closed-loop whip search (same fixed gains) is the unrun prerequisite. So this is
evidence the fixed-impedance *reward landscape* has a wide cheap "press-longer" ridge that dominates the
narrow "strike-faster" peak — a **landscape** problem, to be resolved *within FIC* before any impedance talk.

*Speaker note:* this is the slide to be proudest of — it's a clean causal experiment with an honest,
self-correcting method (four claims were revised mid-investigation when a controlled check beat a plausible story).

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

## Slide 14 — Proposed next steps: PERFECT FIXED IMPEDANCE FIRST (VIC stays deferred)

The whole plan stays inside fixed-impedance control. The two thesis halves — *maximize impact* and *bound it
safely* — turned out to be **coupled**: the constraint is vacuous **because** the strike is a gentle press.
Fix the strike and the constraint becomes real. In order:

1. **Closed-loop whip search (FIC, the linchpin).** Can a fixed-impedance policy, through the real DiffIK
   action space at the *same* PD gains, reach materially past ~1.4 m/s toward the ~4.2 m/s kinematic ceiling?
   CPU trajectory-opt first, then a training run with a wind-up curriculum. This decides whether FIC impact
   can be maximized *at all* — and whether the constraint can be made to bind. (Not VIC: no `set_gains`.)
2. **Make the impulse constraint bind + enforce (FIC, the thesis headline).** With an impactful strike, turn
   on soft-CaT enforcement (`imp_max_p>0`, decision (e)) and/or a harder nail, and demonstrate it bounds
   per-joint impulse *while* the policy maximizes impact — the actual contribution. (Deep-check: the deployed
   reward loads **j1** hardest, so that's the joint enforcement would bind.)
3. **Domain randomization + curriculum (FIC).** Robustness / sim2real — and where longer training iterations
   legitimately return (our "1500 hurts" result was purely a bare-fixed-env artifact).
4. **VIC — explicitly deferred** until FIC is perfected *and* you sign off. It's an efficiency/landscape
   refinement, not a proven necessity (Slide 10c) — and it belongs last.

---

## Slide 15 — What I'm asking you

- Which of the three options on decision (e) — ballistic / windowed-press / split — matches what
  you want the thesis to claim "impact-safe" means?
- Are you comfortable with CaT as the enforcement mechanism, given it's proven on velocity but
  unproven on impulse (because the impulse quantity has never bound)?
- I'm committing to **perfecting fixed impedance before any VIC** (maximize the strike, then make the
  constraint actually bind + enforce, then DR/curriculum). Are you aligned that VIC stays last?
- Is a harder/modified nail target (to make the constraint bind legitimately) in scope, or does
  changing the task risk diluting the thesis's grounding in the real hardware target?
- Given Slide 10c (RL never found the whip, but it's an exploration-not-physics limit), are you happy
  with the **closed-loop whip search** as the immediate next FIC experiment?

---

## Slide 16 — Mathematical formulation (one page; full version in a companion doc)

Everything above in exact notation. The full, code-line-cited version (37-D observation, DiffIK + PD,
all reward formulas, all three impulse accumulators, the caps derivation, the CaT/GAE equations) is
`docs/results/2026-07-20_MATH_FORMULATION.md` — every equation carries a `file:line` citation and was
spot-checked against the code. The essentials:

**The problem is a state-wise constrained MDP** (γ = 0.99, horizon ≤ 1000 control steps):
$$\pi^\* \in \arg\max_\pi\ \mathbb E\!\Big[\textstyle\sum_t \gamma^t r_t\Big]\quad\text{s.t.}\quad \Lambda_{t,j} \le L_j\ \ \forall t,\ \forall j\in\{1..6\}\ \ \text{a.s.}$$
The **return** rewards object-side delivered impulse (impact ↑); the **constraint** bounds the
robot-side per-joint reaction Λ (safety). These pull in opposite directions — that tension *is* the thesis.

**Reward** (dt-scaled, `r_t = 0.02·Σ w_k r_{k,t}`): 8 terms (the 7-term base + `delivered_impulse` on the constraint arm) — Gaussian `approach` (0.1) /
`nail_driven` (0.5), ratcheted `nail_depth_delta` (600), gated `impact_progress` (8), `completion`
(100), `delivered_impulse = 𝟙[progress]·ΔI_t/I_ref` (2, `I_ref=0.6094 N·s`), and penalties
`action_rate` (−0.01) / `joint_pos_limits` (−10).

**Three distinct impulse quantities** (all at the 2 ms substep, contact-masked):
- **Λ** (constraint-read, robot-side): 50 ms sliding window of baseline-subtracted `|qfrc_constraint|` — a **press/fatigue integral**, not clean ballistic momentum (this is the Slide 7 finding, stated formally).
- **I_t** (reward, object-side): episode-cumulative ∫F·dt with a 25-substep re-arm debounce; the maximization target.
- **ContactRow** (log-only diagnostic): per-event Jᵀf, consumed by nothing.

**Caps** (fixed at manufacturer values): `L_j = τ_j^rated · κ · Δt`, κ=2 (HD repeated-peak), Δt≈27.3 ms
→ `L = (1.64, 3.28, 1.64, 1.64, 1.64, 1.64) N·m·s`. *(Note the honest mismatch: caps use ~27.3 ms, Λ integrates 50 ms — this is exactly Khadiv decision (e).)*

**Soft-CaT** turns a violation into a termination probability and discounts only positive reward:
$$\delta_{t,j}=\Big[p_{min}+\text{clip}\big(\tfrac{c_{t,j}}{c^{max}_{t,j}},0,1\big)(p_{max}-p_{min})\Big]_{c>0},\quad \delta_t=\max_j\delta_{t,j},\quad r_t^{CaT}=(1-\delta_t)\,r_t^{+}+r_t^{-}.$$
**Current status: `imp_max_p = 0 ⟹ δ_t^{imp} ≡ 0`** — the impulse constraint is **log-only** (measured, not
enforced). Flipping `imp_max_p > 0` is part of Khadiv decision (e); the machinery is already wired and proven on the velocity constraint.

*Speaker note:* don't walk through equations live — put this up, say "the tension between the impulse
reward and the Λ constraint is the whole thesis in one line," and point at the companion doc for anyone
who wants every `file:line`.

---

## Appendix — exact numbers + caveats

**Best policy (af1), 3 seeds, `git_hash=6b447bf`, `imp_max_p=0`:**
- success: 1.00 / 1.00 / 1.00 · nail depth: 32.0 / 32.0 / 32.0 mm · ep_len: 8 / 7 / 7 steps
- delivered/i_ref (sampled): 1.21× / 0.87× / 0.64× → mean **0.90×**
- worst Λ/cap (sampled): 0.64 / 0.64 / 0.31 → mean **0.53**
- invariants: `impossible_success=0`, `lambda_dead=0` all seeds

**Progress arc:**
- nf1 (pre-fix): success 0.54, depth 28.8 mm, delivered 0.39×, Λ/cap 0.98 (range 0.91–1.12, some over
  cap), ep_len ~168 mean (5 of 6 seeds parked at the 200-step cap)
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

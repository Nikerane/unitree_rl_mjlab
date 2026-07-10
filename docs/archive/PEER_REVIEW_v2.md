> ⚠️ **ARCHIVED 2026-07-05** — superseded by nothing — historical peer-review record (2026-06-02).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Peer Review — Reward Design for RL Hammering & Repetitive Impact

**Review type:** `academic-paper-reviewer`, `full` mode (5-reviewer panel + editorial synthesis)
**Review bar:** technical report / design paper
**Date:** 2026-06-02
**Manuscript (reviewed as one integrated submission):**
- `docs/research/hammering_reward_design_deep_dive_v2.md` (primary)
- `docs/research/reward-design/DEEP_RESEARCH_REPORT.md`
- `docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md`

**Panel (independent, blind to each other):** Editor-in-Chief · R1 Methodology · R2 Domain · R3 Perspective · Devil's Advocate.

**Editor-supplied empirical note (NOT in the manuscript; from the authors' exploratory smoke run this session — a verified datapoint provided to all reviewers):** driving the hammer straight down with a constant action deterministically produced **one sustained contact** that drove the nail ~1.3 mm → 66 mm in a single continuous push (a *press*, not discrete strikes); `impact_progress` fired **once** (~3.88) then 0 while `nail_depth_delta` rewarded the continued drive; the episode reached the success threshold. In sim, a press appears to solve the task without learned "striking."

---

## ⮕ Editorial Decision: **MAJOR REVISION**

Unanimous across the panel (EIC, R1, R2, R3 all "Major"; Devil's Advocate "cannot Accept — CRITICAL"). Per IRON RULE #4, the DA CRITICAL forbids Accept regardless. **Not a Reject:** at the design-paper bar the issues are addressable by reframing + one cheap experiment, and the foundations (momentum thesis, integrity posture, clean implementation) are strong.

### Panel at a glance

| Reviewer | Recommendation | Standout score | Harshest score |
|---|---|---|---|
| EIC | Major | Integrity/Honesty **88** | Scope & Framing **55** |
| R1 — Methodology | Major | Reproducibility **74** | Experimental-design (ablation) **48** |
| R2 — Domain | Major | Literature **88** | Physics-correctness **68** |
| R3 — Perspective | Major | Clarity **82** | Practical-deployability **38** |
| Devil's Advocate | **CRITICAL — not Accept** | — | — |

### 🔴 The consensus finding (5/5, independent) — and the DA CRITICAL

**The headline `impact_progress` term does not do what the manuscript claims.** It is built and motivated as the defense against "slow-press instead of swing" (v2 §11 marks that row `⚠️→✅`; §6.1 calls it "the highest-value single idea"). But:

- `first_contact` fires on **exactly one step** (contact onset), so during a sustained press the term pays **once, then zero** — while `nail_depth_delta` is **path-independent** (v2 §7.3:389: it "telescopes to `weight·final_max_depth`") and rewards the entire press.
- The editor-supplied datapoint confirms the prediction: a constant downward action drove the nail 1.3→66 mm in **one sustained press** to success; `impact_progress` fired once (~3.88) then zeroed.
- The double-gate therefore defends against **scraping** (the Δdepth gate) but leaves **pressing as a global optimum** (R3: "makes the press *unrewarded*, not *non-optimal*").

Reached independently by: **EIC** Concern 2 · **R1** Weaknesses 1–2 · **R2** Weakness 1 (+ the deeper W2–3: the sim nail may be quasi-statically solvable, so striking isn't required unless nail physics are recalibrated to demand impulsive loading) · **R3** Weakness 2 · **Devil's Advocate [CRITICAL]** ("falsifiability/motivation collapse — the sim solves the task by the exact behavior the term is built to discourage, and no falsifiable in-sim prediction distinguishes the term's presence from its absence").

### Other multi-reviewer findings (MAJOR)

- **Claims-vs-evidence / hypothesis-as-result framing** — EIC C1 (no "evidence-status" banner), R1 W3 (ablation "what it teaches" cells assert outcomes with zero data), DA (a 17-question / 14-ablation hypothesis catalog presented as "exceeding a baseline" it never measured).
- **Augment-not-replace self-violation** — DA [MAJOR] + EIC: the corpus's own discipline ("add a term only when a failure is observed"; `DEEP_RESEARCH_REPORT.md:75,108`) was broken by shipping `impact_progress` up front, before observing the failure and before running Q1 (`test_single_strike.py`), which the doc says "gates everything."
- **Ablation matrix confounded + underpowered** — R1 W4–W5: most v2 §10 rows move >1 factor; no seeds/CIs/metrics. The impl-spec's clean A0/A1/A3 two-factor design should be primary.
- **Why RL, not model-based?** — R3 W1 [MAJOR]: the manuscript cites model-based planners that already hammer real nails (Vu 2026, Ti 2024) then re-derives their physics as reward terms for a weaker, sim-only learner without justifying the paradigm. Residual-RL-over-a-scripted-swing (its own Idea #14) should arguably be the headline.

### Disagreements (arbitrated)

- **Gap defensibility:** R2 finds the narrowed gap defensible; the DA calls it narrowing-until-unique sitting where there's no implementation. **Arbitration:** both hold — legitimate as a **research direction**, not yet claimable as a **contribution** (repetition §5.C, sim-to-real DR are unimplemented).
- **The 2000→600 rebalance (#1):** R1 verified the arithmetic (132→~40, restoring completion-dominance); the DA notes its value-cliff justification is undercut by the manuscript's own correct retraction (v2 §2.3#4: the cliff is bounded/GAE-smoothed). **Arbitration:** defensible on the **per-step-spike** argument (a +132 single-step spike > +100 completion inverts the hierarchy), distinct from the discounted-cliff argument — but should be **tested** (A0 vs A1), not asserted.

### ✅ Strengths the panel agrees to preserve

- The **momentum-not-force thesis** is physically correct and well-grounded (EIC, R2, R3, DA); `m_eff = 1/(n̂ᵀJM⁻¹Jᵀn̂)` reproduced correctly (R2).
- **Exemplary citation-integrity / honesty posture** — tiered evidence tags, disclosed prior fabrication, self-correction (EIC 88, R2 88, credited even by the DA).
- **Clean, verifiable implementation** — TDD, axis-verified-from-XML, dynamic-weight validation (R1, R2, DA Observations).
- The **position-only action-space critique** and retraction of non-actionable orientation/torque recommendations (R2, R3).

---

## 🗺️ Revision Roadmap (prioritized)

**Tier 1 — decision-blocking (must fix to clear Major Revision)**

1. **Confront the press head-on (resolves the CRITICAL).** Pick and argue one: (a) **recalibrate nail physics** so quasi-static pressing fails — rate-dependent resistance / static-friction breakaway — then let `nail_depth_delta` alone induce striking (R2's preferred, most robust); or (b) reward **pre-impact axial momentum over an approach window** gated on the following contact, so a zero-approach-velocity press earns nothing (R2 option a); or (c) accept pressing as a valid sim solution and **drop the striking framing** (R3). Add an Adroit-style `−0.1/step` time penalty as cheap pressure regardless (R2). **And run the falsifying experiment:** A1 (5-term) vs A3 (7-term) — does behavior shift press→strike, and does success-rate change? (R1, DA).
2. **Resolve Q1 now and fold it in.** Report `test_single_strike.py` output (or formally adopt the press datapoint). Per the doc's own protocol this gates everything repetitive (R1 W8, DA).
3. **Add an "Evidence status & scope" banner** at the top of v2; downgrade every `→✅` and every causal "improves learning" cell to "hypothesized; untested" (EIC C1, R1 W3, DA).

**Tier 2 — strengthen (should fix)**

4. De-confound the ablation matrix: adopt the impl-spec's clean A0/A1/A3 factorial; specify ≥N seeds, mean±CI, per-hypothesis metric+threshold (R1 W4–W5).
5. Add a **"Why RL vs model-based / hybrid"** section; consider promoting residual-RL-over-scripted-swing to the headline design (R3 W1).
6. Address the **augment-not-replace self-violation** explicitly — justify the exception (gated on Q1) or defer the term (DA).
7. **Anchor the core physics** on stronger/older peer-reviewed sources (Wang & Kheddar RSS 2019; Stronge, *Impact Mechanics*); add a consolidated "verify-before-publication" citation checklist (R2 W5, EIC C5, DA).
8. **Safety:** either descope the deployability/recoil motivation or move the "cheap, strong" terminating safety envelope into the shipped baseline (R3 W4).

**Tier 3 — polish (minor)**

9. Re-ground `v_expected` from observed `v_axial` percentiles; the `1.0` default makes the "O(1) normalization" a no-op divide-by-one (R1 W6, R2 W6, DA).
10. One canonical **as-implemented 7-term table**; mark the 9-term SPEC and v1 §4.1 as superseded (EIC C4).
11. γ-horizon mismatch (0.99 vs 1000-step episodes): correct or explicitly defer with rationale (DA, R1).
12. Derive the 4 mm settle dead-zone from a logged no-contact rollout rather than hardcoding (R1 W7).
13. Label `validate_rewards.py` Phase I as a smoke test, or make it state-injected (R1 W9).
14. Tighten length; lead with the as-shipped change + its single justification (EIC C6).

---
---

# Appendix — Individual Reviewer Reports (Phase 1, independent & blind)

## A. Editor-in-Chief

### Summary of submission
A three-part technical-report package on reward design for an RL hammering task: a Unitree-Z1 arm driving a nail (success = nail_slide qpos ≥ 0.07 m) via position-only DifferentialIK. The PRIMARY document (v2 deep dive) extends a prior local-corpus synthesis (DEEP_RESEARCH_REPORT); the third (IMPL_SPEC) specifies the two shipped changes — rebalancing `nail_depth_delta` 2000→600 and adding a double-gated `impact_progress` event term (weight 8). The v2 thesis is well-formed: for a stiff position-controlled arm, the only controllable, transferable impact quantity is end-effector axial momentum (`m_eff·v_axial`), not simulated contact force (a solver artefact). From it the report derives a task decomposition (reward machine), a taxonomy, three escalating proposals (A/B/C), a safety section, an ablation matrix, a failure-mode/reward-hacking checklist, plus its own DA pass, Limitations, and a tiered bibliography. There is NO RL training evidence — only reward-function unit tests and a 50-iteration CPU smoke run. The report is generally honest about this and tier-tags every load-bearing citation.

### Overall assessment & contribution
Strong design-and-synthesis document with a genuine, defensible thesis and an exemplary honesty posture, undermined by two structural problems: (1) a scope/promise mismatch between an expansive 18-idea / 9-term-spec vision and a four-line shipped change with zero training evidence (the "two voices" issue, acknowledged but unresolved); and (2) the manuscript's framing is contradicted by a known datapoint its own failure-mode taxonomy should have anticipated. The core contribution — reframing the impact lever as controllable momentum rather than commanded force on a position-only action space — lands. The most serious EIC concern is anti-confirmatory: the press drove the nail 1.3→66 mm in one sustained contact while `impact_progress` fired once then zeroed and `nail_depth_delta` happily rewarded the push, because depth reward is path-independent (v2:388-389) and dominant. The contribution stands as a design paper; the "✅ defended" claims must be downgraded to "hypothesised, untested."

### Strengths
- Clear, defensible central thesis (momentum-not-force), threaded consistently: v2:24, :166-177, §2.3 challenge at :71.
- Exemplary citation-integrity posture: six-tier evidence legend (v2:9-16), per-claim tagging, Limitations naming residual fabrication risk by tier (v2:563, :570); continuous honesty trail from OPUS_AUDIT.md:48-55 (H3 fabricated-author finding) into v2.
- "Two voices" tension surfaced with a verdict, not buried: DEEP_RESEARCH_REPORT.md:14 ("trust the code, not the spec").
- Strong escalation logic: Proposal A/B/C (v2:228,:294,:309) explicitly gated on observed failure and on Q1.
- Implementation spec is tight: TDD ordering, axis verified against XML (IMPL_SPEC:38), deliberate finite-diff-not-`site_vel_w` choice (:42), concrete acceptance criteria (:111-115).
- The DA pass is substantive and self-correcting — overturns the report's own "near-empty niche" claim with a logged external sweep (v2:554).
- The companion literature notes reframe the novelty claim correctly, with honest CN/JP coverage caveats.

### Concerns
1. **[MAJOR] Promise-vs-delivery scope mismatch.** v2 promises 18 ideas / taxonomy / 3 proposals / RM / 14-row matrix, but the delivered artifact is two weight edits with no training data; the abstract/exec-summary doesn't foreground that essentially none is validated. WHERE: v2:19-39 vs IMPL_SPEC:8-14. FIX: add an "Evidence status & scope" banner at the top of v2.
2. **[MAJOR] Failure-mode checklist over-claims defenses as achieved.** The "slow-press" case is contradicted by the datapoint. WHERE: v2:502 ("⚠️→✅"), v2:498, against v2:388-389. FIX: change "→✅" to "(hypothesised; untested)"; add a row acknowledging a single sustained press satisfies success without striking and that gating `impact_progress` alone does not prevent this.
3. **[MINOR→borderline MAJOR] Dense/sparse arithmetic inconsistency across docs**, none reconciled to the press datapoint. WHERE: DEEP_RESEARCH_REPORT.md:110 (+500→37.5), v2:217/:27 (2000→132; 600→30-40), IMPL_SPEC:20-23. FIX: reconcile in one place; note a single press collects the full path-independent depth total regardless of weight.
4. **[MINOR] Three overlapping prescriptive layers, no canonical "what's true now" table inside v2.** WHERE: DRR:14,:118-129 (6-term live table, now stale at 7), RECOMMENDED_REWARD_SPEC.md:15-24 (9-term, weights differ). FIX: a single authoritative as-implemented (7-term) table at the top of v2.
5. **[MINOR] Citation-integrity residual liability.** Load-bearing-sounding claims rest on E2*/E3; same work cited at different tiers across docs. WHERE: v2:563, :613-624, DRR:217. FIX: a "what to verify before external publication" checklist.
6. **[MINOR] Readability/length.** 627 dense lines; highest-value content diluted across §1/§4/§9/§12. FIX: cross-reference rather than restate; lead with the as-shipped change.

### Scores
Originality/Contribution **72** · Significance **64** · Scope & Framing **55** · Clarity **66** · Integrity/Honesty **88**

### Preliminary recommendation
**Major revision** — strong thesis and exemplary honesty, but the scope banner (C1) and the press-vs-"striking" over-claim (C2) must be fixed before it reads as a sound pre-registration rather than results. **Confidence 4.**

---

## B. Reviewer 1 — Methodology

### Summary
A design paper for a reward formulation on a position-only DiffIK Z1 hammering task, plus a "validate-without-training" verification methodology. The central methodological contribution — treating reward terms as deterministic functions and unit-testing them against scripted state sequences (REWARD_VALIDATION_METHODOLOGY.md; validate_rewards.py 9 phases) — is sound, honestly scoped, well-executed. The authors are explicit that this catches term-implementation bugs but NOT learnability, local optima, reward-hacking, or sim-to-real (REWARD_VALIDATION_METHODOLOGY.md:33-40; DRR:205). Citation-integrity discipline is exemplary. However, the core reward-design claim is undermined by the editor's note, and the manuscript's own logic predicts it: a single sustained "press" fires the double-gated `impact_progress` exactly once and drives the nail to success. The double-gate defends against scraping (Δdepth gate) but NOT against slow/single-press solutions — the failure mode it is repeatedly motivated by (§11, ⚠️→✅). The A1-vs-A2 rebalance arithmetic is correct but rests on a path-independence property that makes `nail_depth_delta` indifferent to press-vs-strike. The ablation matrix (v2 §10) is confounded, under-specified on power/seeds/metrics, and several cells assert causal outcomes with zero data.

### Strengths
- Reward-as-deterministic-function thesis is correct; bug-class table concrete (REWARD_VALIDATION_METHODOLOGY.md:6-28).
- Honest scoping of what validation does NOT catch (:33-40; DRR:205).
- Script implements the methodology including stateful `reset()` (Phase F) and dynamic weight reads (validate_rewards.py:169-178,:49-52).
- Phase I is a real end-to-end gate test (:207-240).
- Double-gate correctly implemented; unit tests cover the right adversarial cases (rewards.py:167-202; test_impact_progress_reward.py:76-161).
- Finite-diff velocity choice defensible & documented (IMPL_SPEC:42-44; OPEN_QUESTIONS.md:123-127).
- Rebalance arithmetic reproducible/correct (verified w=2000→132, w=600→39.6).
- Tiered citations + documented prior fabrication catch (v2:9-16; OPUS_AUDIT.md:48-55).
- Self-correction of an earlier over-claim (Q11 value cliff), v2 §2.3#4 (:79-80).

### Weaknesses / required changes
1. **[MAJOR]** Double-gated `impact_progress` does not defend against the press/single-sustained-contact solution. The Δdepth gate stops scraping; nothing against a slow press, because `first_contact` fires once and the depth gate is satisfied by any >ε advance. WHERE: v2:502, :353, :177; rewards.py:200-202. FIX: demote ✅ to ⚠️; distinguishing press from strike requires a velocity *floor* gate, an air-time/wind-up precondition, or impulse-over-a-window, not depth-advance-at-contact; add press as a first-class failure mode with its own ablation.
2. **[MAJOR]** `nail_depth_delta` path-independence makes it provably indifferent between one press and many strikes; with the press solving the task, the entire dense+sparse stack is maximized by a press. WHERE: v2:388-389, :183. FIX: state that the Proposal-A stack does not make striking optimal if a press reaches success.
3. **[MAJOR]** Causal learning claims asserted without training data ("balance ↑ stability, ↓ variance", "teaches swing-that-advances"). WHERE: v2 §10 rows A1/A3/A5 (:473,:475,:477), §1 item 5 (:27). FIX: reframe as "predicted effect"; the body repeatedly violates the Limitations disclaimer (:561).
4. **[MAJOR]** Ablation matrix confounded — most variants change >1 factor vs Proposal A; only A0↔A1 and A3↔A4 are clean. WHERE: v2:470-485 vs IMPL_SPEC clean factorial (:107-109). FIX: adopt the IMPL_SPEC two-factor design as primary.
5. **[MAJOR]** No experimental-design parameters anywhere (seeds, iterations, CIs, statistical test). WHERE: v2:453-465; DRR:162-166,:184-189. FIX: ≥N seeds/cell, mean±CI, pre-registered metric+threshold.
6. **[MINOR]** `v_expected=1.0` "O(1)" claim unsupported — analytic v_max≈2.5 m/s (§7.7), observed ~3.88; weighted ×8 → ~25-31 in one step. WHERE: v2:219,:405; IMPL_SPEC:46-48. FIX: set `v_expected`~2.5-4 m/s or re-justify weight=8.
7. **[MINOR]** 4 mm settle dead-zone is a hardcoded magic constant, justification asserted not derived. WHERE: rewards.py:100-115; v2:497; REWARD_VALIDATION_METHODOLOGY.md:124. FIX: derive from a logged no-contact rollout (New-Q17).
8. **[MAJOR]** Reproducibility gaps: `verify_reward_setup.py` (Q5), `test_single_strike.py` (Q1), and the smoke run are referenced as if results exist, but no outputs/logs reported. Q1 "gates everything repetitive" (:540) yet absent — and the editor's note appears to answer it. WHERE: v2:404-405; OPEN_QUESTIONS.md:8-21; IMPL_SPEC:92-96. FIX: report the actual output and fold its consequence in.
9. **[MINOR]** Phase I depends on real physics/emergent behavior — a behavioral smoke test smuggled into the "validation-without-training" suite. WHERE: validate_rewards.py:207-240. FIX: label it a smoke test or make it state-injected.

### Detailed comments
- REWARD_VALIDATION_METHODOLOGY.md:124 ("tests for consistency, not correctness in the abstract") is the single most important sentence; elevate to the methodology abstract.
- v2 §2.3#4 (:79-80): retraction of the `100/(1−γ)` over-claim is correct (termination truncates the bootstrap) — verification posture working.
- v2 §4.7.1 (:216): "target per-term episode-sum bands" is the correct replacement for the audited-broken §8 arithmetic — but unverified (no logged bands reported).
- A3-vs-A4 (:475-476) is the one well-posed reward-hacking experiment (gate-on vs gate-off) — but tests the *scrape* hack, not the *press* hack.
- A2 "normalisation removes need to re-weight" is asserted, not argued; normalization rescales the whole advantage stream and doesn't by itself restore completion-dominant hierarchy.
- γ-horizon mismatch (§7.6 :401, γ=0.99 → ~100-step horizon vs 1000-step episodes) is legitimate but itself a causal claim made without data.

### Scores
Soundness/Validity **62** · Reproducibility/Verification **74** · Experimental-design (ablation) **48** · Rigor-of-claims-vs-evidence **55** · Clarity **80**

### Recommendation
**Major** — verification methodology is publishable, but the central reward design must confront the press solution, and the ablation matrix needs de-confounding, power specification, and hypothesis-not-result framing. **Confidence 4.**

---

## C. Reviewer 2 — Domain Expert

### Summary
A literature-rich, physics-aware reward-design report. The central physics thesis — reward geometric outcomes (nail depth) and end-effector momentum `m_eff·v_axial`, never simulated contact force — is correct and well-grounded. The effective-mass formula `m_eff = 1/(n̂ᵀJM⁻¹Jᵀn̂)` (v2 §4.1, refs Vu et al. 2026) is reproduced accurately and matches the composite-rigid-body / impact-mechanics literature (Wang–Dehio–Kheddar 2022; Stronge restitution via Vu §III-C). The literature sweep is strong: the gap claim — "sim-to-real RL for repetitive, momentum/impact-explicit nail-driving on a manipulator" — is defensible and appropriately narrowed, correctly distinguishing quasi-static Adroit/D4RL hand-press from arm-strike physics. However, the report contains a load-bearing design error the editor's note exposes: `impact_progress` (single-step `first_contact` gate) does NOT defend against the slow-press exploit it claims to neutralize (§11 ⚠️→✅). A sustained press makes `first_contact` true for one step while depth advances over many; the term fires once and the path-independent depth-delta (§7.3) pays the press in full. The momentum framing is correct but, as implemented, does not motivate striking over pressing.

### Strengths
- Momentum-not-force thesis physically correct and textbook-grounded (v2 §1.1, §4.3:166-177; Vu §III-C via notes §2.3:69,:193).
- `m_eff` formula reproduced correctly/consistently (v2:152,:355 vs notes §2.3:60); speed-vs-inertia tradeoff flagged unresolved (New-Q13:543).
- Position-only action-space critique correct and consequential; corroborated by Romanyuk 2019 (notes §2.5:103; v2 §8.1:427); challenge to orientation-gate/torque/impedance recommendations sound (§2.3:71).
- Gap claim defensible and honestly narrowed (DA-pass:554; notes §2.6:134, §2.7:173).
- Citation-integrity exemplary (tiering; fabrication-history disclosure; demoted AutoMate claim).
- Sim-to-real reasoning correct and properly conditioned — geometry necessary but not sufficient; DR of contact params is what transfers (§2.3 item 5:82-83).

### Weaknesses / required changes
1. **[MAJOR]** `impact_progress` does not defend against slow-press — a load-bearing physics/design error. `first_contact` true only on contact onset; a press pays at most one small bonus (Δdepth in the first step may be <ε → zero), while path-independent depth-delta pays the press the same as a strike. WHERE: v2:502, IMPL_SPEC:35,41-44, v2:171. FIX: (a) reward *pre-impact* axial momentum over an approach window gated on a following contact (a press has near-zero approach velocity → earns nothing); or (b) credit Δdepth only within a post-`first_contact` window decaying for sustained contact; or (c) add Adroit-style −0.1/step now (§5.A:292). At minimum, retract the §11 "⚠️→✅".
2. **[MAJOR]** Velocity ceiling makes single-strike feasibility — and whether striking is even necessary — physically suspect, never reconciled with the press. v_max≈2.5 m/s (§7.7:405); the press (v_axial→0 after contact) also clears it → the sim nail (frictionloss=0.3, damping=8, taken on faith, Limitations §5:565) offers so little resistance that neither momentum nor striking is required. WHERE: v2:405, IMPL_SPEC:48. FIX: compute what static force the nail requires vs what the arm delivers quasi-statically (Adroit's nail absorbs ~15 N, notes §2.6:115); if a press solves it, state nail physics must be recalibrated BEFORE impact rewards can be validated.
3. **[MAJOR]** No rate/velocity-dependent nail-resistance model — the mechanism that would make striking necessary. WHERE: v2 §4.3,§8; harvested idea exists only as DR noise (notes §3 idea#2:183). FIX: add a static-friction breakaway / rate-dependent damping so quasi-static pressing is penalized by physics; then `nail_depth_delta` alone induces striking — more robust than gating reward terms.
4. **[MAJOR]** Path-independence undercuts the claim that depth-delta rebalancing + a gated impact term produces striking. The report itself (§7.3:389) says only RM/air-time/impact terms produce striking — but #1+#2 add only the (broken) impact term, not air-time/RM. WHERE: v2:389 vs §1.1/§12 framing. FIX: state in §12 that #1+#2 alone are NOT expected to produce striking over pressing.
5. **[MINOR]** Vu et al. 2026 is the single load-bearing source for the momentum thesis + `m_eff` formula. WHERE: v2 References:589. FIX: add Wang & Kheddar (RSS 2019) and Stronge, *Impact Mechanics* (2018) — older, stable anchors (in notes §2.3-2.4 but absent from v2 References).
6. **[MINOR]** `v_expected=1.0` asserted without grounding against reachable impact speed. FIX: set from `verify_reward_setup.py` percentiles of observed `v_axial` at contact.

### Detailed comments
- v2 §4.3:176-177 specifies "Δdepth over a *post-impact window*" but the implementation measures Δdepth over a *single step* at first contact — aligning them would partially address W1.
- v2 §2.3#4: the `100/(1−γ)`→bounded-`O(100)` correction is right.
- v2 §4.5:194 / notes idea#11: rebound-credit is moot on the welded rigid Z1 (stores no spring energy) — belongs in a future-compliance section.
- IMPL_SPEC:38: axis `n̂=(0,0,-1)` verified against nail_block_scene.xml:26 — good.

### Missing references
- **Wang & Kheddar (2019), Impact-Friendly Robust Control, RSS 2019** — foundational impact-momentum-QP paper; stable peer-reviewed anchor (see W5).
- **Stronge (2018), Impact Mechanics (2nd ed.)** — authoritative basis for the restitution/momentum model behind "never reward contact force."
- **A rate/velocity-dependent contact or wood-fracture model** — the physical premise that makes striking necessary; its absence is why the press exploit is a surprise.

### Scores
Physics/Technical-correctness **68** · Literature-coverage **88** · Novelty/Gap-defensibility **80** · Domain-contribution **70** · Clarity **85**

### Recommendation
**Major** — general physics correct and literature excellent, but the central design does not, as implemented, defend against the press exploit it claims to solve; this and the un-confronted quasi-static-solvability of the sim nail require revision. **Confidence 4.**

---

## D. Reviewer 3 — Cross-Disciplinary Perspective

### Summary
A careful, literate reward-design report. Within its frame (model-free PPO + reward shaping) it is unusually self-aware: it diagnoses its own action-space mismatch (§2.3), reframes the impact lever as momentum not force (§4.3), and audits its citations. From the outside view, it never confronts the question its own bibliography raises: it cites model-based impact planners that already hammer real nails (Vu 2026, Ti 2024, Xu 2025), then re-derives their physics as reward terms for a weaker, sample-hungry, sim-only optimizer — without asking whether RL is the right tool. The editor's note is fatal to the premise as stated: a single sustained press drove the nail to success and `impact_progress` fired once then zeroed. The double-gated term (the report's "highest-value single idea," §6.1) does not prevent the press; it merely fails to reward it. The whole "striking vs pressing" motivation may be an artifact of a welded-hammer, position-only, gravity-on sim. Deployability is asserted as future work but the safety design is entirely on the deferred side.

### Strengths
- Honest action-space diagnosis; orientation-gate is "moot," fix is an action-space change (v2 §2.3#1:70-71, §1 item 9:31, §12 items 13/15:534,536).
- Correct momentum reframing grounded in composite-rigid-body impact (v2 §1 item 2:24, §4.3:166-177).
- The manuscript already contains the alternative paradigm — residual-RL-over-scripted-swing (Idea #14, §6:367; §12:530) and admittance wrapping (§8.3:435).
- Tiered citations + explicit [PROPOSAL]/[ANALOGY] tagging (v2:9-15; Limitations §3:563).
- Implementation spec genuinely deployable as engineering (IMPL_SPEC:76-116).

### Weaknesses / required changes
- **[MAJOR]** Paradigm choice never argued. Cites model-based `m_eff` maximization (Vu 2026) and optimal-control nail-hammering (Ti 2024) then re-derives the same quantities as reward terms for model-free RL with no justification, no real-robot result. WHERE: v2:24,:152,:172, References:589,610. FIX: add a "Why RL, why not model-based / hybrid" section with the falsifiable hypothesis RL is supposed to buy; if the honest answer is "not clearly better," reframe as residual RL over a model-based swing (Idea #14) as the primary design.
- **[MAJOR]** Editor's note shows a press solves the sim & `impact_progress` fires once then zeros — contradicting the central motivation. The double-gate makes the press *unrewarded*, not *non-optimal*: depth-delta (600) still pays it, completion (+100) still terminates. WHERE: v2 §6 item 1:353, §11:502, §4.4:183,389. FIX: confront the press — accept it (drop striking framing → task shrinks), or make pressing physically impossible/penalized, or make depth-delta contingent on impulsive loading. Reconcile §7.7 (v_max≲2.5 m/s) with the observed press.
- **[MAJOR]** Experimental platform (welded hammer + position-only + gravity-on) taken as given, never challenged as a *scientific* decision; the platform may be incapable of exhibiting controlled dynamic striking — the press is the natural optimum. WHERE: v2 §3 phase table:124, §2.3#1:71; CLAUDE.md (weld active). FIX: add a platform-justification subsection (why welded vs grasped, position-only vs minimal torque, gravity-on); if the platform can't disambiguate press from strike, say so as a limitation gating downstream conclusions.
- **[MAJOR]** Deployability is the stated goal but every safety mechanism (PPO-Lagrangian/CPO, terminating envelope, thermal/RUL, impulse caps, admittance) is deferred; the shipped design has zero impact safety. WHERE: IMPL_SPEC:12-14; v2 §8:409-435, §12:528-529. FIX: descope deployability, or move the terminating safety envelope (the report's own "cheap, strong," §8.2:430) into the implemented baseline.
- **[MINOR]** "Hammering" treated as the end task; broader framing (stand-in for general impulsive manipulation? what transfers?) absent. WHERE: DA item 2:554. FIX: a "Generalization beyond nails" subsection separating platform-independent contributions (momentum-gated event reward, double-gating, event-driven RM) from nail-specific ones (depth-delta, settle dead-zone).
- **[MINOR]** Leans on Adroit `hammer` as proof the skeleton "works," but Adroit *presses* quasi-statically — internally inconsistent given the press observation. WHERE: v2 §5.A:292, DA item 2:554. FIX: state Adroit grounds the *shaping skeleton* only, and its press is precisely the failure mode this work must avoid.

### Cross-disciplinary opportunities & alternative framings
- **Hybrid model-based + residual RL as the primary design** (Idea #14 promoted): the base primitive *is* a swing, sidestepping the press optimum, buying safety + sample efficiency + a clean S2R story.
- **Reframe as impulsive-manipulation control**, hammering as one instance — widens significance.
- **Treat the press as a diagnostic, not a bug**: the sim under-constrains the problem (gravity-assisted quasi-static drive of a damping=8 nail). Characterizing *when* a task genuinely requires impact vs admits a quasi-static solution is itself a contribution.
- **Borrow deployment-safety frame from impact-aware control** (van Steen reference-spreading; Xu recoil result) → impedance/feedback scheduling around contact, which maps to admittance wrapping (§8.3) — arguably more important than any reward term and currently buried.

### Scores
Significance/Impact **52** · Paradigm-appropriateness **45** · Practical-deployability **38** · Breadth/Transfer **55** · Clarity **82**

### Recommendation
**Major** — strong design survey, but must justify RL-vs-model-based and confront the press result before the central premise stands. **Confidence 4.**

---

## E. Devil's Advocate

### Strongest Counter-Argument (steelmanned, then attacked)
**Steelman.** The spine (v2 §1.2, §4.3, §6.1) is rigorous: on a position-only DiffIK arm you cannot command force, only an EE position target; therefore the controllable, transferable impact quantity is axial momentum `m_eff·v_axial`, and rewarding instantaneous MuJoCo contact force chases a `solref`-dependent solver artefact (van Steen 2024). Double-gating speed on `first_contact ∧ Δdepth>ε` is a sound anti-reward-hacking construction (Skalse 2022 / Pan 2022). As reward engineering hygiene, defensible.

**Attack.** The thesis equivocates between (A) "momentum is the controllable impact lever" (a physics fact, true) and (B) "therefore the policy must learn to *strike*, and we must reward striking" (a behavioral prescription, **unsupported and now falsified in-sim**). The editor's run shows a constant downward action drove the nail 1.3→66 mm in **one continuous press** to success — no swing, no retract-restrike. `impact_progress` fired once (~3.88) then zero, while `nail_depth_delta` (weight 600) rewarded the entire drive. So in sim the task is solved by the exact behavior the new term is built to discourage, and the term contributes one transient pulse to a 1000-step success episode. Its motivation ("slow-press instead of swing") is a sim-to-real hypothesis about hardware with zero hardware data, and the manuscript itself files slow-press as a mere ⚠ partial failure (v2 §11) — yet shipped the term up front. The central argument does not establish that `impact_progress` is needed in sim; at best it is insurance against an unobserved sim-to-real failure, untestable on current evidence.

### Issue List
- **[CRITICAL]** Falsifiability/motivation collapse of the headline term (Soundness). IMPL_SPEC:35 + rewards.py confirm `fc = compute_first_contact()` — true only on contact onset; the sim-optimal solution is a single sustained press, so `impact_progress` is a one-shot pulse while `nail_depth_delta` (weight 600, hammer_env_cfg.py:158) rewards the whole press. No falsifiable in-sim prediction distinguishes "with" from "without" the term; v2 §11 lists slow-press as only ⚠. A4 in the matrix tests gate-vs-no-gate (scraping), NOT term-vs-no-term on the press.
- **[MAJOR]** Self-violation of "augment-not-replace / add-on-observed-failure" (DRR:75,108; CLAUDE.md). `impact_progress` added up front, before observing the failure and before Q1 (`test_single_strike.py`), which v2 §7.7/§12-5 says gates everything and should run first. IMPL_SPEC:3 "design approved"; the "deliberate, literature-backed exception" is post-hoc.
- **[MAJOR]** Goalpost-shifting on the gap claim (v2 DA-pass:554 admits "near-empty niche" was overturned; re-narrowed to repetitive momentum/impact nail-driving — the *least* supported region: repetition is untested §5.C, sim-to-real DR deferred IMPL_SPEC:12).
- **[MAJOR]** Hypothesis catalog, not a standalone falsifiable result. 17 open questions, 14-row matrix, all deferred to future GPU training (Limitations §1). Delivered/verified: two edits + unit tests + a 50-iter smoke run ("the loop runs," IMPL_SPEC:92). Framed as "exceeding a baseline" (v2:7) with no behavioral improvement demonstrated.
- **[MAJOR]** Citation integrity: load-bearing claims chain through weak tiers. ~11 [E2*] "not re-fetched" (v2:613-624), prior fabrication (Meta-World H3, DRR:77); the hammering-specific `m_eff` / "64% recoil" lean on Xu 2025 / Vu 2026 / Ti 2024 [E2]; ARMADA carries the M1 venue flag; DrEureka/RTW venue-unverified. The most quotable claims rest partly on flagged tiers (v2:570).
- **[MINOR]** `v_expected=1.0` makes the normalization cosmetic (IMPL_SPEC:48 "numerically identical … at the default") — a divide-by-one; real scaling lives in `weight=8.0`, itself an untuned guess.
- **[MINOR]** γ-horizon mismatch identified as "currently-wrong" (§7.6) but uncorrected (rl_cfg.py:41 γ=0.99; hammer_env_cfg.py:247 20 s); for the single-press solution the whole multi-strike horizon argument is moot.
- **[MINOR]** Unfalsifiable hedging in §11 — grades the new term ✅ against an unobserved failure while the general principle (line 509) is "assume every dense term is hackable."

### Ignored Alternative Explanations / Paths
1. **The press *is* the correct policy in sim, possibly on hardware** — Adroit's own hammer env (cited approvingly) is a press; a controlled push may be sim-optimal AND more S2R-robust (no impact transient, lower peak torque). The report assumes striking beats pressing; never argues it.
2. **`nail_depth_delta` alone may already be sufficient** — path-independence means it neither prefers nor punishes pressing; with completion terminating, the 5-term pre-#2 baseline might solve the task without `impact_progress`. The A1-vs-A3 comparison is never run.
3. **Lowering 2000→600 may slow learning for no benefit** — the value cliff is benign by the report's own analysis (§2.3#4, DRR:110); the rebalance trades early-signal strength (A1 "weaker early signal") for a problem that may not exist. The simpler A2 (keep 2000, return-normalize) is mentioned but not chosen, unjustified by data.
4. **Action-space change over reward engineering** — the report repeatedly says the real lever for orientation/compliance/recoil is an action-space change; if striking matters, scripted-swing-plus-residual addresses it more directly than a one-shot event reward.

### Missing Stakeholder Perspectives
- **The PPO optimizer** — no analysis of how a term non-zero on ~1 step/episode (~31 magnitude) interacts with GAE/advantage normalization; potentially noise, not gradient. The variance argument is applied to `completion` (Q11) but never to the new term.
- **The hardware/S2R engineer** — every sim-to-real claim is deferred; the stakeholder who would benefit from striking-over-pressing has zero evidence.
- **The reproducer** — [E2*] author lists not re-fetched.
- **The future maintainer** — 17 questions + 14 ablations, no prioritization of the single cheapest falsifying experiment (A1 5-term vs A3 7-term: strike vs press).

### Observations (non-defects)
- The code matches the spec; `ImpactProgressTerm` implements the double-gate, max-tracked prev_depth, finite-diff velocity with `_init` mask, `clamp_min(0)` exactly. Clean engineering.
- The Q11 rebuttal is correct and code-verified (success → `nail_fully_driven` termination truncates the bootstrap; cliff is O(100) local, not 100/(1−γ)).
- "Never reward simulated contact force" is well-grounded (van Steen [E1], DrEureka).
- `_SETTLE_OFFSET=0.004` is a real, documented anti-hacking defense.
- Self-tiering of citations + disclosure of the prior fabrication is honest, above bar.
- Including its own DA pass is good practice; its items 1–5 are real but softer than the in-sim-falsification problem above (which it does not raise).

### Verdict
**Yes — one CRITICAL issue forbids "Accept" as written:** the falsifiability/motivation collapse of the headline `impact_progress` term. The sim solves the task by a sustained press — the behavior the term is built to discourage — under which the term fires once and is otherwise inert while `nail_depth_delta` carries the episode. Not fatal to a *design paper* if reframed as an explicitly-hypothetical insurance term gated on a Q1/A1-vs-A3 result not yet run; but as currently framed (a justified, shipped contribution that "exceeds the baseline"), it is unsupported/borderline-unfalsifiable and must be revised before acceptance.

---

*Generated by the `academic-paper-reviewer` skill (full mode, 5-reviewer blind panel + editorial synthesis), reviewing the corpus as a technical-report submission. Reviewers were independent subagents; the editorial synthesis traces every consolidated point to a specific Phase-1 report. The "editor-supplied empirical note" is a real datapoint from this session's CPU smoke run, not part of the manuscript.*

# Deep-Research Report — Reward Design for the Z1 Hammer-Nail Task

**Pipeline:** ARS deep-research (`full` mode, local-corpus variant)
**Date compiled:** 2026-05-27
**Corpus:** 6 markdown files + 4 Python validation scripts in `docs/research/reward-design/`
**Project:** mjlab Unitree-Z1-Hammer (branch `hammer-z1`), DifferentialIK action space, nail at z=0.102 m, success threshold ≈ 0.071 m of nail travel.

---

## Executive Summary

The corpus is internally healthy after Opus 4.7's second-pass audit (`OPUS_AUDIT.md`), but it speaks with *two* voices: a **prescriptive 9-term spec** (`RECOMMENDED_REWARD_SPEC.md` §4) and an **executed 6-term baseline** (`hammer_env_cfg.py`, validated by `validate_rewards.py`). The audit's verdict — *"trust the code, not the spec"* — is correct and load-bearing for every recommendation below.

The **strongest defensible formulation today** is the implemented 6-term baseline at its **live weights** (`approach` +0.1, `nail_driven` +2.0, `nail_depth_delta` +2000, `completion` +100, `action_rate` −0.01, `joint_pos_limits` −10.0), launched against the ablation order in §5 of the spec, with the 3 deferred terms (`approach_contact_gated`, `impact_velocity_bonus`, `air_time_bonus`) gated on **observed failure modes** rather than added upfront. Note: these live weights have drifted from the spec — see §4.1. `validate_rewards.py` now reads weights dynamically from the env config and passes all 8 phases against this set on mjlab 1.4.

Open questions that *cannot* be resolved further from documentation:
- **Q1** (single-strike feasibility) — has a script ready (`test_single_strike.py`); resolving it determines whether `air_time_bonus` is essential or optional.
- **Q11** (value-function discontinuity from +100 completion bonus) — added by the audit; only training data will answer.
- **Q12** (decimation-induced approach-gate flicker) — also audit-added; only relevant after `approach_contact_gated` is enabled.

---

## 1. Research Question and Scope

**RQ:** What is the strongest defensible reward formulation for the Z1 hammer-nail RL task, given (a) the literature cited in `REWARD_LITERATURE.md`, (b) the open empirical questions in `OPEN_QUESTIONS.md`, (c) the contradictions surfaced in `OPUS_AUDIT.md`, and (d) the implemented baseline validated by `validate_rewards.py`?

**In scope.** Reward-term selection, weighting, ablation order, validation methodology, identification of unresolved questions and contradictions.

**Out of scope.** Network architecture, PPO hyperparameters, domain randomisation distributions, sim-to-real calibration (Q7), and the GPU training campaign itself.

**Methodological note.** This is a local-corpus deep-research run — Phase 2 (Investigation) collapses to thorough reading of the existing materials rather than an external literature sweep. The bibliography was *already* compiled and audited; the audit corrected the one fabricated citation (Meta-World, H3) and flagged three preprint-vs-venue claims (M1). I do not re-verify those claims here.

---

## 2. Corpus Map

| File | Role | Authority |
|------|------|-----------|
| `RECOMMENDED_REWARD_SPEC.md` | Prescriptive design (9 terms, full pseudocode) | Spec — secondary to code per audit verdict |
| `REWARD_DESIGN_MATRIX.md` | Term-by-term feasibility matrix | Secondary to code |
| `REWARD_LITERATURE.md` | Annotated bibliography (APA 7) | Primary for justifications |
| `REWARD_VALIDATION_METHODOLOGY.md` | Unit-testing principle for reward functions | Primary for *what's testable without training* |
| `OPEN_QUESTIONS.md` | 12 enumerated open questions (10 original + Q11/Q12 from audit) | Primary for unresolved items |
| `OPUS_AUDIT.md` | Critical second-pass on the above | Primary for contradictions and severity |
| `validate_rewards.py` | 8-phase reward unit test — all pass | **Ground truth for current behaviour** |
| `verify_contact_sensor.py` | Confirms `hammer_head` geom resolves | Ground truth for sensor wiring |
| `verify_reward_setup.py` | Random-policy reward-fire sweep + percentile printer | Required pre-training gate |
| `test_single_strike.py` | Resolves Q1 empirically | Run before deciding `air_time_bonus`'s necessity |

---

## 3. Synthesis Across Sources

### 3.1 Where the corpus converges (high-confidence claims)

1. **Progress framing is the right primary signal.** Wu et al. 2021 (DREM, contact-rich manipulation), the legacy Gym design (`depth_delta × 500`), and the implemented `NailDepthDeltaTerm` all agree: reward **improvements** in nail depth, not absolute position. The Gaussian-on-absolute-depth `nail_driven_reward` has near-zero gradient at 0 mm (`exp(-6.25) ≈ 0.002`) — useless when the policy hasn't yet made contact.

2. **Potential-based reward shaping (PBRS) is the theoretical guard.** Ng, Harada & Russell (1999) prove only `F(s,s') = γΦ(s') − Φ(s)` preserves policy invariance. The current `hammer_approach_reward` is *not* PBRS — it can bias the policy toward hovering. `nail_depth_delta` is closer to PBRS in spirit (rewards change in a potential function). This is the principal theoretical justification for replacing the always-on Gaussian if hovering is observed.

3. **Geometry-based outcomes transfer; contact forces do not.** van Steen et al. (2024) measured 3.1% post-impact velocity error in calibrated MuJoCo, and Ma et al. 2024 (DrEureka) confirmed that rewards exploiting simulation-specific artefacts (precise contact normals) hurt sim-to-real. Implication: `nail_depth_delta` and `impact_velocity_bonus` (finite-differenced position) are safe; a `contact_force_reward` is not. The audit's M2 finding (missing `torque_sum_sq_penalty`) is reasonable but does not contradict this — torque penalties are on the *robot* side, not on the *contact-physics* side.

4. **Event-gated bonuses, not continuous "near contact" rewards.** D'Ambrosio et al. 2023 (table tennis, 35+ reward components tried) and Kim et al. 2025 (ARMADA) both used **+1 at the moment of contact**, not a dense approach signal. The hammer task's analogue is `impact_velocity_bonus × first_contact`, gated by `ContactSensor.compute_first_contact()`.

5. **Validation without training is necessary and feasible.** `REWARD_VALIDATION_METHODOLOGY.md` explicitly recognises Eureka's evaluation step as the closest published analog. All 8 phases of `validate_rewards.py` pass, which means *for the 6 terms currently implemented*, the reward stack is internally consistent (no missed resets, no stale state, no weight-magnitude errors, no missing clamps). Any failure during training is attributable to *learnability*, not reward-term bugs.

### 3.2 Where the corpus contradicts itself (resolved)

The Opus audit caught five disagreements and resolved them in-place. They are noted here only because each has a downstream implication for the recommended formulation.

| Audit ID | Issue | Resolution | Implication |
|----------|-------|------------|-------------|
| C1 | Pseudocode `super().__init__(cfg, env)` would `TypeError` | Spec patched to `super().__init__(env)` | Anyone copying §3.1 or §3.3 pseudocode verbatim must use the patched version. The implementation already uses the correct call. |
| H1 | Spec §1 said "Replace ..."; code Augments | Spec §1 rewritten; original recommendations demoted to "future replacements" | **The augment-not-replace strategy is the user's explicit minimum-viable approach** (saved as `feedback_minimum_viable_reward.md`). Any future work should respect it unless a failure mode is observed. |
| H2 | MATRIX said completion bonus could re-fire without a guard | SPEC was correct: termination on success prevents re-fire | No guard needed in the implementation; `validate_rewards.py` Phase H confirms exactly-once firing. |
| H3 | Meta-World citation had 5 fabricated authors | Verified against arXiv:1910.10897 and corrected | This is the single instance of citation fabrication. All other arXiv IDs are claimed verified. |
| M1 | Three preprints (RTW, DrEureka, ARMADA) cited as venue-published | Marked "preprint — venue not verified" | Argument from authority is weaker for these; use them as engineering inspiration, not peer-reviewed evidence. |

### 3.3 Where the corpus has acknowledged gaps

These survive the audit and constitute the unresolved research surface.

**Empirical gaps that require running the env (resolvable today, no GPU needed):**
- **Q1** — Single-strike feasibility. Script: `test_single_strike.py`. **Run this before training.** It determines whether `air_time_bonus` is essential (single-strike infeasible) or merely useful (single-strike works).
- **Q5** — `min_air_time` / `max_air_time` thresholds. Script: `verify_reward_setup.py` already prints recommended percentiles from 200 random-policy steps. Run once before first training and re-run after partial training.

**Empirical gaps that require GPU training (defer):**
- **Q2** — Does retract-and-restrike emerge from `nail_depth_delta + impact_velocity_bonus` alone, or is `air_time_bonus` load-bearing? Ablation Step 3 vs Step 4.
- **Q3** — Optimal `impact_velocity_bonus` weight. Grid search [1, 5, 10, 20, 50].
- **Q6** — Approach std (0.10 vs 0.15). Two-run ablation, measure time-to-first-contact.
- **Q11** *(audit-added)* — Does the +100 completion bonus produce a value-function discontinuity that destabilises PPO? Train at +100 vs +50 and watch the value-loss and KL curves around first successes.
- **Q12** *(audit-added)* — Does decimation=10 cause `approach_contact_gated` to miss intra-step bounces, mixing approach reward with impact rewards on the same step? Only matters once the gated approach term is enabled.

**Gaps that require hardware (defer to sim-to-real phase):**
- **Q7** — Nail physics calibration. Needs a real-world strike measurement.
- **Q9** — Single-strike vs multi-strike learnability + transfer.

**Missing standard impact-RL terms (M2, audit-added):**
- `time_penalty` (−0.01/step). Standard episodic-RL guard against the policy reaching the goal in the very last step. Trigger to add: policy succeeds but episode lengths cluster near max.
- `torque_sum_sq_penalty` (`sum(τ²)` on arm joints). Standard RSL-RL legged-locomotion term; complements the existing `joint_pos_limits` and `action_rate`. Trigger to add: sim-to-real transfer fails due to actuator saturation, or arm exhibits "torque chatter" in rollouts.
- `orientation_alignment` (multiplicative quat gate). Used by Meta-World v3 hammer. Trigger to add: policy approaches the nail from a non-axial direction and strikes fail because the hammer face misses the nail head.

### 3.4 Where this report extends the corpus

Two pieces of synthesis are mine, not the corpus's:

**A — The "augment-not-replace" baseline + ablation order together form a coherent research protocol.** The spec gives them in different sections (§1 and §5) and the audit's H1 patch only fixes §1. Read together, they prescribe: (i) train the 6-term baseline (Step 0); (ii) only proceed to Steps 1–7 if a specific failure mode (slow press, hovering, no swing) is observed. This is the correct interpretation of `feedback_minimum_viable_reward.md`. Adding all 9 terms upfront would conflate the open weight-tuning questions Q3, Q6, Q11 with the question "does the reward shape produce the right behaviour?"

**B — Q11 (completion-bonus discontinuity) is partially mitigated by the existing `nail_depth_delta` weight of +500.** `500 × 0.075 m = 37.5` of cumulative dense progress reward over a full nail drive. The +100 completion bonus is therefore ~2.7× the cumulative dense reward — large, but not the 1000× cliff that a sparse-only design would produce. The audit's framing of Q11 as a "huge" cliff at γ near 1 is technically correct (`100/(1−γ)` grows large as γ→1), but the immediate reward-step jump is bounded by the design's existing dense shaping. The empirical concern is real but unlikely to be catastrophic at the current weights. Watch the value-loss curve; reduce to +50 if you see spikes around first successes.

---

## 4. Recommended Reward Formulation (defensible, minimum-viable)

### 4.1 The Step-0 baseline (use this for the first training run)

This is the *implemented* baseline, validated end-to-end by `validate_rewards.py`. **Do not add terms before running this.** Weights are **live values from `hammer_env_cfg.py`** (verified 2026-05-27 via `reward_manager.get_term_cfg`). They have drifted from the spec — trust this table over `RECOMMENDED_REWARD_SPEC.md` §1.

| Term | Live weight | Was in spec | Source | Notes |
|------|------------|-------------|--------|-------|
| `approach` (Gaussian on dist) | **+0.1** | +0.5 | Legacy + Eureka exp-shape | Lowered from spec. Hovering risk still applies; monitor. |
| `nail_driven` (Gaussian on absolute depth, std=0.03) | +2.0 | +2.0 | Meta-World, current mjlab | Near-zero at 0 mm; kept for near-goal fine gradient. |
| `nail_depth_delta` (stateful progress) | **+2000** | +500 | DREM (Wu 2021) | Increased 4× from spec. Primary learning signal at 0 mm. |
| `completion` (sparse +1 at threshold) | +100 | +100 | Meta-World, D'Ambrosio 2023 | Termination prevents re-fire. Watch Q11. |
| `action_rate` (L2 of action delta) | −0.01 | −0.01 | RSL-RL (Rudin 2022) | Sim-to-real smoothness. |
| `joint_pos_limits` (soft penalty) | **−10.0** | −1.0 | HPRS (Berducci 2024) | Strengthened 10× from spec. Safety boundary. |

**Implication of the weight drift:** `nail_depth_delta` now dominates the dense reward landscape (4× larger), and `approach` is much weaker (5× smaller). This shifts the policy's incentive earlier toward *making contact and advancing the nail* rather than *getting close to the nail*. The hovering risk from `approach` is mitigated by its small weight — partial implicit fix for the audit's hovering concern, though not the principled gated-approach solution. The stronger `joint_pos_limits` (−10) raises the cost of safety violations relative to task reward; consistent with HPRS hierarchy (safety > target > comfort).

### 4.2 Deferred terms — add one at a time, only on observed failure

| Trigger observation | Add term | Weight | Source |
|---------------------|----------|--------|--------|
| Policy hovers at nail_top, doesn't strike | `approach_contact_gated` (replaces `approach`) | +0.5, std=0.10 | Ng 1999 (PBRS), D'Ambrosio 2023 (event-gated) |
| Policy presses slowly instead of swinging | `impact_velocity_bonus` (stateful, finite-diff) | +10 | ARMADA (Kim 2025) |
| Policy strikes once but doesn't retract for another swing — *and* Q1 confirms single-strike is infeasible | `air_time_bonus` | +3 | RSL-RL feet_air_time (Rudin 2022) |
| Episode lengths cluster near max | `time_penalty` | −0.01/step | Standard RL |
| Actuators saturate or rollout shows torque chatter | `torque_sum_sq_penalty` | −0.001 | RSL-RL legged locomotion |
| Policy strikes from wrong angle, hammer misses nail head | `orientation_alignment` (quat gate) | multiplicative | Meta-World v3 (Yu 2020) |
| Visible jerk in rollouts | Raise `action_rate` to −0.02 | — | RSL-RL |
| Reaching goal too late in episode | Add `time_penalty` (above) | — | — |

### 4.3 Pre-training verification (mandatory)

Per `RECOMMENDED_REWARD_SPEC.md` §6 (M5-corrected):

```bash
python docs/research/reward-design/validate_rewards.py       # 8 phases must pass
python docs/research/reward-design/verify_contact_sensor.py  # ContactSensor wiring
python docs/research/reward-design/verify_reward_setup.py    # random-policy reward-fire sweep
```

And **before deciding whether `air_time_bonus` is essential**:

```bash
python docs/research/reward-design/test_single_strike.py     # Q1 resolution
```

`verify_reward_setup.py` must print `=== Reward setup verified. Safe to start training. ===` before `train.py` is launched.

Enable per-term wandb logging (rsl_rl logs `Episode_Reward/<term>` automatically). Watch the first 100 iterations:
- `Episode_Reward/nail_depth_delta` > 0 by iteration 5 — else the term is silent (broken).
- `Episode_Reward/completion` > 0 by iteration ~200 — else the policy isn't reaching success even occasionally.
- Value-loss curve should not show a discrete jump on the iteration of the first success — if it does, that's Q11 evidence; halve `completion` weight to +50.

### 4.4 Ablation order (only if Step-0 baseline insufficient)

Follow `RECOMMENDED_REWARD_SPEC.md` §5 as written — it's the audit-clean version. Summary:

```
Step 0: 6-term baseline (above)
Step 1: replace approach → approach_contact_gated     IF hovering
Step 2: add impact_velocity_bonus                     IF slow press
Step 3: add air_time_bonus                            IF no retraction AND Q1 confirms multi-strike needed
Step 4: add joint_vel penalty                         IF rollouts unsmooth
Step 5: tune completion weight (Q11)                  IF value-loss instability
Step 6: add near-goal Gaussian back if dropped        IF nail depth precision matters at termination
```

---

## 5. Devil's Advocate Pass

Two challenges to this recommendation, considered and answered.

**Challenge 1 — "The 9-term spec was the engineering deliverable. Why not deploy it?"**

Because the spec admits its weights are estimates (M4 audit finding: §8 arithmetic doesn't balance), and at least three of the nine terms have open weight-tuning questions (Q3 for `impact_velocity_bonus`, Q6 for `approach_contact_gated` std, Q11 for `completion`). Launching all nine at once means any training failure is ambiguous: bad shape? bad weight? bad combination? The 6-term baseline is *validated* (Phase 8 passes), the ablation order is published, and each added term has a clear trigger condition. This is straightforward debugging hygiene, not conservatism.

**Challenge 2 — "If single-strike turns out to be infeasible (Q1), then `air_time_bonus` is essential and deferring it wastes a training run."**

This is the only place where "defer until observed" loses time. The cost is a Step-0 run that the policy fails on, after which Step 3 is added. The mitigation is to **run `test_single_strike.py` before any training**. If it returns `max_depth < 30 mm` at all approach heights, fold Steps 1–3 into Step 0 immediately. This is cheaper than running PPO on a known-impossible single-strike target.

---

## 6. Limitations of This Synthesis

1. **No independent literature search performed.** The bibliography was already audited; this report inherits its grade. The H3 author-fabrication finding (Meta-World) is the only confirmed citation-integrity issue; M1 (three preprints presented as venue-published) survives — RTW, DrEureka, and ARMADA's venue claims are not re-verified.

2. **No training data.** Every "if X is observed" trigger is hypothetical. Some failure modes may never appear (single-strike may just work); others may co-occur (the policy may both hover and press slowly), forcing simultaneous term additions despite the prescribed sequential discipline.

3. **No hardware data.** Q7 (nail physics calibration) is purely deferred. Sim-to-real transfer is a separate research surface that this report does not address.

4. **The `validate_rewards.py` ground-truth is only as good as its scripted state sequences.** It tests phases A–H against pre-stated expectations. If the mental model of "what should happen at depth=0.010 m" is wrong, validation passes but the reward is still buggy in a more abstract sense. This is acknowledged in `REWARD_VALIDATION_METHODOLOGY.md`.

5. **AI-assisted research.** The audited corpus was originally produced by Sonnet 4.6, audited by Opus 4.7, and synthesised here. The audit caught one fabricated citation; further fabrication risk is non-zero, especially in M1-flagged entries.

---

## 7. References Cited (Internal Corpus)

Citations are reproduced exactly as they appear in `REWARD_LITERATURE.md` after the H3 audit correction. Three entries (Wang et al. 2025, Ma et al. 2024 DrEureka, Kim et al. 2025 ARMADA) carry the M1 caveat that their published venue is not independently re-verified.

- Berducci, L., Aguilar, E. A., Ničković, D., & Grosu, R. (2024). HPRS: Hierarchical potential-based reward shaping from task specifications. *Frontiers in Robotics and AI*, 11. arXiv:2110.02792
- D'Ambrosio, D. B., et al. (2023). Robotic table tennis: A case study into a high speed learning system. *RSS 2023*. arXiv:2309.03315
- Kim, J., Kim, J., Lee, D., Jang, Y., & Kim, B. (2025). ARMADA: A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation. *RSS 2025 — venue not re-verified*. arXiv:2502.16908
- Kim, Y., Lee, J., Choe, J., Cho, H., & Kang, Y. (2023). Not only rewards but also constraints. arXiv:2308.12517
- Kumar, V., Todorov, E., & Levine, S. (2016). Optimal control with learned local models. *ICRA 2016*.
- Luo, Y., et al. (2022). Dense2Sparse reward shaping. *IROS 2022*. arXiv:2003.02740
- Ma, Y. J., et al. (2023). Eureka: Human-level reward design via coding LLMs. *ICLR 2024*. arXiv:2310.12931
- Ma, Y. J., et al. (2024). DrEureka: Language model guided sim-to-real transfer. *arXiv preprint* — venue not verified. arXiv:2406.01967
- Mu, T., Liu, M., & Su, H. (2024). DrS: Learning reusable dense rewards for multi-stage tasks. *ICLR 2024*. arXiv:2404.16779
- Narang, Y., et al. (2022). Factory: Fast contact for robotic assembly. *RSS 2022*. arXiv:2205.03532
- Ng, A. Y., Harada, D., & Russell, S. J. (1999). Policy invariance under reward transformations. *ICML 1999*, 278–287.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to walk in minutes using massively parallel deep RL. *CoRL 2022*. arXiv:2109.11978
- Tang, B., et al. (2023). IndustReal: Transferring contact-rich assembly tasks from simulation to reality. *RSS 2023*. arXiv:2305.17110
- van Steen, J., Stokbroekx, D., van de Wouw, N., & Saccon, A. (2024). Impact-aware robotic manipulation: Quantifying the sim-to-real gap for velocity jumps. arXiv:2411.06319
- Wang, L., Xu, T., Lu, Y., & Xiao, X. (2025). Reward training wheels. *arXiv preprint* — venue not verified. arXiv:2503.15724
- Wu, Z., Lian, W., Unhelkar, V., Tomizuka, M., & Schaal, S. (2021). Learning dense rewards for contact-rich manipulation tasks. *ICRA 2021*, 2339–2345. arXiv:2011.08458
- Yu, T., Quillen, D., He, Z., Julian, R., Narayan, A., Shively, H., Bellathur, A., Hausman, K., Finn, C., & Levine, S. (2020). Meta-World. *CoRL 2020*. arXiv:1910.10897

---

## 8. AI-Assisted Research Disclosure

This synthesis was compiled by Claude (Sonnet 4.6) executing the ARS `deep-research` pipeline in `full` mode on a local corpus. No external literature search was performed. The corpus itself was produced by Sonnet 4.6 (initial drafts) and audited by Opus 4.7 (`OPUS_AUDIT.md`). Citation integrity inherits the audit's findings: 1 confirmed fabrication corrected (H3), 3 preprint-vs-venue ambiguities flagged (M1), all other arXiv IDs nominally verified. The H3 finding should give the reader calibrated caution about every other unverified claim — including in this report.

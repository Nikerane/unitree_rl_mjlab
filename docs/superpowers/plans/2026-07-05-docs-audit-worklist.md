# Docs Audit Worklist

_Date: 2026-07-05_

Assembled from the docs-consolidation audit. One row per doc: verdict, count of stale claims, inbound-reference count. Per-cluster sections carry the exact stale-claim line/quote/evidence/replacement, followed by an inbound-reference map and the deduped poison-phrase candidate list. Quote and replacement text is reproduced verbatim from the audit — this is an assembly, not an edit.

## Summary table

| path | verdict | #stale claims | inbound-ref count |
|---|---|---|---|
| docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md | fix | 10 | 13 |
| docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md | fix | 5 | 19 |
| docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md | fix | 1 | 14 |
| docs/research/reward-design/OPEN_QUESTIONS.md | fix | 3 | 18 |
| docs/research/reward-design/ORIENTATION_ROBUST_SIM2REAL_ARM.md | current | 0 | 5 |
| docs/research/reward-design/NAIL_PRECISION_CURRICULUM.md | current | 0 | 2 |
| docs/research/reward-design/CAT_DEEP_DIVE.md | merge | 5 | 10 |
| docs/research/reward-design/JOINT_VELOCITY_BOUND_RESEARCH.md | merge | 1 | 9 |
| docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md | archive | 0 | 6 |
| docs/research/reward-design/OPUS_AUDIT.md | banner-check | 0 | 7 |
| docs/research/reward-design/PEER_REVIEW_v2.md | banner-check | 0 | 6 |
| docs/research/reward-design/REAL_HAMMER_PLAN.md | record | 0 | 4 |
| docs/research/reward-design/REWARD_LITERATURE.md | merge | 4 | 13 |
| docs/research/reward-design/REWARD_VALIDATION_METHODOLOGY.md | archive | 0 | 13 |
| docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md | archive | 0 | 12 |
| docs/research/hammering_lit_sweep_RUNBOOK.md | archive | 0 | 1 |
| docs/research/hammering_literature_notes.md | merge | 2 | 13 |
| docs/research/impact_tracking_rl_litreview.md | merge | 3 | 6 |
| docs/research/hammering_reward_design_deep_dive_v2.md | record | 0 | 15 |
| docs/research/tracking_impact_impulse_design_research.md | record | 0 | 12 |
| docs/FUTURE_UPDATES.md | archive | 0 | 6 |
| docs/HANDOVER.md | archive | 0 | 8 |
| docs/VEGA_TRAINING_PLAN.md | fix | 2 | 9 |
| docs/results/README.md | record | 0 | — |
| docs/results/2026-06-17_b_strike.md | record | 0 | — |
| docs/results/2026-06-17_velocity_bound_ablation.md | record | 0 | — |
| docs/thesis/README.md | fix | 1 | — |
| docs/archive/README.md | banner-check | 0 | — |
| docs/archive/BASELINE_AUDIT.md | banner-check | 0 | — |
| docs/archive/DEEP_RESEARCH_REPORT.md | banner-check | 0 | — |
| docs/archive/IMPACT_TRACKING_REWARD_SPEC.md | banner-check | 0 | — |
| docs/archive/RECOMMENDED_REWARD_SPEC.md | banner-check | 0 | — |
| docs/archive/REWARD_DESIGN_MATRIX.md | banner-check | 0 | — |
| CLAUDE.md | fix | 1 | — |
| README.md | current | 0 | — |
| thesis_synthesis.md | record | 0 | 7 |
| thesis_direction_update.md | record | 0 | 8 |
| thesis_handoff_brief_original.md | record | 0 | 7 |
| docs/superpowers/plans/2026-06-17-z1-single-strike-fix.md | archive | 0 | 1 |
| docs/superpowers/plans/2026-06-17-r_imit-tracking-reward.md | archive | 0 | 0 |
| docs/superpowers/specs/2026-06-17-z1-strike-not-press-redesign-design.md | archive | 0 | 6 |
| docs/superpowers/plans/2026-07-05-docs-consolidation.md | current | 0 | — |
| docs/superpowers/specs/2026-07-05-docs-consolidation-design.md | current | 0 | — |

MISSING FROM AUDIT (added by completeness check): docs/superpowers/specs/2026-07-05-docs-consolidation-design.md.

### Memory rows (counted separately)

| path | verdict | #stale claims |
|---|---|---|
| .claude/…/memory/MEMORY.md | fix | 2 |
| .claude/…/memory/cat-constraints-as-terminations.md | current | 0 |
| .claude/…/memory/diffik-maxdq-velocity.md | current | 0 |
| .claude/…/memory/hammer-sim-audit-fixes.md | fix | 1 |
| .claude/…/memory/hammer-site-off-face.md | record | 0 |
| .claude/…/memory/impulse-cat-c0-findings.md | current | 0 |
| .claude/…/memory/impulse-cat-deep-review.md | current | 0 |
| .claude/…/memory/l6-hammer-fixture-ee.md | current | 0 |
| .claude/…/memory/nail-precision-curriculum.md | current | 0 |
| .claude/…/memory/orientation-robust-vicon-arm.md | current | 0 |
| .claude/…/memory/qfrc-constraint-weld-pollution.md | fix | 2 |
| .claude/…/memory/scene-xml-physics-desync.md | current | 0 |
| .claude/…/memory/soft-cat-scale-positives-decision.md | current | 0 |
| .claude/…/memory/softcat-velocity-result.md | current | 0 |
| .claude/…/memory/z1-arm-gravcomp.md | current | 0 |
| .claude/…/memory/z1-hardware-limits.md | current | 0 |
| .claude/…/memory/z1-velocity-bound-finding.md | current | 0 |

### Sibling-repo rows (counted separately)

| path | verdict | #stale claims |
|---|---|---|
| /Users/nikerane/repos/safe_impact_manipulation/CLAUDE.md | current | 0 |
| /Users/nikerane/repos/safe_impact_manipulation/README.md | current | 0 |
| /Users/nikerane/repos/safe_impact_manipulation/hammer_z1_env/README.md | fix | 1 |

---

## Cluster: impulse-CaT / soft-CaT plans

### docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md (fix)

Notes: Status line (line 3) 'C0 IMPLEMENTED' still correct, but C5 also shipped (deep-review). §5 and §7 still frame Pinocchio as a C0 gate/prerequisite — it is deferred. All J_limit/delivered/window numbers are gripper-era and must carry the EE-dependent 're-derive' caveat. Inbound-ref value (CLAUDE.md points here as the impulse plan) intact. §6b NEAR_NAIL re-solve (line 171) is now double-superseded by the L6 fixture.

**Claim (line 6)**
- Quote: `> **C0 findings:** contaminant is dof-FRICTION (~45% of raw Λ_j on joints 2,3), not the weld (~0.2); object-side delivered impulse 0.137 N·s is the clean weld/friction-immune ground truth; per-joint J_limit = [3.44, 6.88, 3.44…] N·m·s, reference strike at 4.7% of the worst-joint limit`
- Evidence: impulse-cat-c0-findings.md:20 — EE-dependent, drifted after L6 fixture: 0.107 N·s, J_limit [2.28,4.56,...], window 38 ms
- Replacement: `> **C0 findings (gripper-era measurement — EE-dependent, re-derive via derive_impulse_thresholds.py):** contaminant is dof-FRICTION (~41–48% of raw Λ_j, spread across the load-bearing joints), not the weld (~0.05); the object-side delivered impulse (~0.107 N·s on the L6-fixture tree, was 0.137 on the gripper) is the clean weld/friction-immune ground truth; per-joint J_limit (fixture-era [2.28, 4.56, …] N·m·s, was [3.44, 6.88, …] pre-L6) puts the reference strike well under the worst-joint limit`

**Claim (line 6)**
- Quote: `reference strike at 4.7% of the worst-joint limit (NON-binding for gentle strikes, binds for the ~21× more violent learned strikes).`
- Evidence: derive_impulse_thresholds.py:233-236 — binding-ness is computed live per run, not a constant; c0-findings.md:20 says never hardcode
- Replacement: `the reference strike sits at a small single-digit % of the worst-joint limit (NON-binding for gentle strikes, binds for the ~20×-more-violent learned strikes); the exact ratio is EE-dependent — re-run derive_impulse_thresholds.py at C2.`

**Claim (line 64)**
- Quote: `Z1_JOINT_IMPULSE_LIMIT: float = 0.1            # placeholder`
- Evidence: hook.py:54-79 + constraints.py:35 — placeholder is now guard-rejected under enforcement; joint_impulse_excess requires an explicit limit (no silent default)
- Replacement: `Z1_JOINT_IMPULSE_LIMIT: float = 0.1            # placeholder — NEVER a silent default: joint_impulse_excess requires an explicit limit, and CatSoftHook rejects this VALUE under enforcement (imp_max_p>0)`

**Claim (line 92)**
- Quote: `def joint_impulse_excess(env, limit=Z1_JOINT_IMPULSE_LIMIT, robot_cfg=_ARM_CFG) -> torch.Tensor:`
- Evidence: constraints.py:35 — signature is joint_impulse_excess(env, limit); limit is REQUIRED and the robot_cfg param was removed
- Replacement: `def joint_impulse_excess(env, limit) -> torch.Tensor:   # limit REQUIRED; no default, no robot_cfg (the accumulator's own robot_cfg fixes the joint set)`

**Claim (line 101)**
- Quote: `### 3.3 \`CatSoftHook\` diff — **two lines**, everything else unchanged`
- Evidence: hook.py:162-204 — the impulse path is a dedicated _add_impulse_constraint with a sparse-signal per-column self-seeding normalizer + log-only short-circuit, not a two-line .add()
- Replacement: `### 3.3 \`CatSoftHook\` diff — a dedicated \`_add_impulse_constraint\` (sparse-signal per-column self-seeding normalizer + \`imp_max_p==0\` log-only short-circuit), everything else unchanged`

**Claim (line 105)**
- Quote: `self._cat.add("joint_impulse_excess", c_imp, max_p=self._imp_max_p)   # get_probs() already MAXes`
- Evidence: hook.py:178-204 — impulse does NOT reuse CaT.add's shared EMA; it writes probs directly via a per-column violation-masked normalizer (velocity's batch-max EMA would saturate on the sparse impulse signal)
- Replacement: `self._add_impulse_constraint(env)   # writes cat.probs['joint_impulse_excess'] via a per-column self-seeding normalizer; get_probs() then MAXes it in (soft-OR)`

**Claim (line 119)**
- Quote: `| **Sparse-signal EMA collapse** (major) | Λ≈0 between strikes → batch-max EMA c_max decays to the 1e-6 floor → δ saturates at max_p on *every* contact (hard lottery, kills graded δ) | **do not inherit the velocity EMA**: mask the EMA update to contact steps / use a robust statistic (p95 of recent contact events) / **seed c_max from the C0 reference histogram**;`
- Evidence: hook.py:104-105,194-199 — fix shipped as CaT-style FIRST-VIOLATION per-column seeding; imp_seed is a decay FLOOR only, NOT a p95/excess statistic
- Replacement: `| **Sparse-signal EMA collapse** (major) | Λ≈0 between strikes → a batch-max EMA c_max would decay to the floor → δ saturates at max_p on every contact | **do not inherit the velocity EMA**: the shipped fix is CaT-style FIRST-VIOLATION per-column seeding (cmax = max(first over-limit batch-max, imp_seed floor)) with a per-column violation-masked EMA afterwards; imp_seed is a decay FLOOR only (any small value safe), NOT a p95/excess statistic;`

**Claim (line 132)**
- Quote: `- **C5 — combined soft-OR arm (optional, last).** velocity ∪ impulse in one CaT instance (one \`.add()\`; 12 columns).`
- Evidence: impulse-cat-deep-review.md:15 + env_cfgs.py:236-257 — C5 composition SHIPPED (cat_soft=True + cat_impulse=True → one hook, velocity ∪ impulse soft-OR)
- Replacement: `- **C5 — combined soft-OR arm (SHIPPED).** cat_soft=True + cat_impulse=True composes into ONE hook (velocity ∪ impulse soft-OR); report the column-contribution (argmax) histogram to show the impulse column actually drives terminations.`

**Claim (line 127)**
- Quote: `5. **Pinocchio cross-check decides the quantity here**, not later. *Tests:* accumulator sums over the contact window, zeros at boundaries, shape (B,6); off-contact ≈ 0; margin never a probability.`
- Evidence: impulse-cat-c0-findings.md:22 + derive_impulse_thresholds.py:218-225 — Pinocchio is DEFERRED (not installed); the MuJoCo-native object-side ∫F·dt is the accepted ground truth; the quantity decision was not gated on Pinocchio
- Replacement: `5. **The MuJoCo-native object-side ∫F·dt is the accepted ground truth; Pinocchio impulseDynamics cross-check is DEFERRED (not installed — decision pending).** *Tests:* accumulator sums over the contact window, zeros at boundaries, shape (B,6); off-contact ≈ 0; margin never a probability.`

### docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md (fix)

Notes: MERGE TARGET for CAT_DEEP_DIVE (absorbs it as a conceptual appendix per design spec §End state). The Decision 1–7 record and §1 reference-code line map are load-bearing and code-consistent (Decision 1 scale-positives matches hook.py:_NEG_TERMS + _compute_r_pos; Decision 7 timeout bootstrap matches). Fix the stale status + file-name/phase-count lines, then append the CAT_DEEP_DIVE body. C4 (line 309) is the '9 phases' poison hit.

**Claim (line 3)**
- Quote: `**Status:** pre-implementation (design approved, decisions recorded). 2026-06-17.`
- Evidence: src/tasks/hammer/cat/ + rl/ exist (constraint_manager.py, hook.py, cat_ppo.py, cat_storage.py); impulse-cat-deep-review.md — C0–C5 shipped, 270 tests green
- Replacement: `**Status:** IMPLEMENTED (soft-CaT C0–C5 shipped on branch soft-cat; 270 unit tests green). Design + decisions below are the authoritative record. 2026-06-17 (impl through 2026-07-05).`

**Claim (line 275)**
- Quote: `| \`src/tasks/hammer/cat/cat_hook.py\` | Non-terminating manager hook (Decision 5): calls the manager, stashes δ on \`env\` + into \`extras["cat_delta"]\`, never feeds \`reset_buf\`.`
- Evidence: the shipped file is src/tasks/hammer/cat/hook.py (class CatSoftHook), not cat_hook.py
- Replacement: `| \`src/tasks/hammer/cat/hook.py\` (\`CatSoftHook\`) | Non-terminating manager hook (Decision 5): calls the manager, stashes δ on \`env\` + into \`extras["cat_delta"]\`, never feeds \`reset_buf\`.`

**Claim (line 203)**
- Quote: `The \`cat_hook\` computes \`r_neg\` from the two known negative terms
(\`action_rate\`, \`joint_pos_limits\`) and stashes \`r_pos = r_total − r_neg\` into \`extras\` alongside δ;`
- Evidence: hook.py:35,149-160 — the class is CatSoftHook in hook.py; _NEG_TERMS = (action_rate, joint_pos_limits); computes r_pos = r_total − r_neg
- Replacement: `\`CatSoftHook\` (\`cat/hook.py\`) computes \`r_neg\` from the two known negative terms
(\`action_rate\`, \`joint_pos_limits\`) and stashes \`r_pos = r_total − r_neg\` into \`extras\` alongside δ;`

**Claim (line 309)**
- Quote: `CLAUDE.md pre-train gate (\`validate_rewards.py\` 9 phases + \`verify_contact_sensor.py\`) for the new`
- Evidence: validate_rewards.py:129-438 — phases A through M (13 phases), 'ALL PHASES PASSED'
- Replacement: `CLAUDE.md pre-train gate (\`validate_rewards.py\` phases A–M + \`verify_contact_sensor.py\`) for the new`

**Claim (line 276)**
- Quote: `| \`src/tasks/hammer/cat/curriculum.py\` | Port of \`modify_constraint_p\` (\`curriculums.py:21-42\`): lifetime ramp \`T:20→1/init_max_p\`, \`max_p=1/T\`, clocked by \`common_step_counter\`. Optional for v1. |`
- Evidence: src/tasks/hammer/cat/curriculum.py does not exist; C5 curriculum listed 'optional, after C4 green' and not shipped
- Replacement: `| \`src/tasks/hammer/cat/curriculum.py\` (NOT YET SHIPPED — C5 optional) | Port of \`modify_constraint_p\` (\`curriculums.py:21-42\`): lifetime ramp \`T:20→1/init_max_p\`, \`max_p=1/T\`, clocked by \`common_step_counter\`. |`

### docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md (fix)

Notes: MERGE TARGET for JOINT_VELOCITY_BOUND_RESEARCH (appendix per design spec). Body is a literature/mechanism synthesis (camps a/b/c, verified citations) and is overwhelmingly current — the constraint-TYPE argument, CaT-vs-Lagrangian verdict, and the 4.3–4.65 rad/s ablation reference all still hold and match velocity_bound.py + soft-cat decision. Only §3 phrasing conflates the impulse quantity with a per-substep instantaneous ceiling (it is per-contact-event). The Ma/Hutter N-P3O + Spoor λ-fragility + Vu/EMMT hammering citations are the ones the C0 findings lean on — keep verbatim. Git shows +25 lines already added (net-new external evidence, lines 243-266).

**Claim (line 114)**
- Quote: `we need
\`impulse_per_joint(t) ≤ limit\` and \`|q̇_j(t)| ≤ 3.1415\` at *every substep on every joint* (impulse
accumulated at 500 Hz).`
- Evidence: impulse_bound.py:24-30,68-146 — the shipped impulse constraint is a PER-EVENT PULSE over a contact-anchored window, not a per-substep instantaneous ceiling; velocity is per-substep, impulse is per-contact-event
- Replacement: `we need \`Λ_j\` (per-joint reaction impulse, per contact EVENT) \`≤ limit\` and \`|q̇_j(t)| ≤ 3.1415\` at *every substep on every joint* — velocity is the per-substep worst-case; the impulse is accumulated at 500 Hz over a contact-anchored window and read with per-event pulse semantics (impulse_bound.py).`

### docs/research/reward-design/CAT_DEEP_DIVE.md (merge → FAITHFUL_SOFT_CAT_IMPL_PLAN.md)

Notes: MERGE SOURCE → FAITHFUL_SOFT_CAT_IMPL_PLAN (CaT-as-incentive-vs-VIC-as-capability conceptual appendix per design spec). Load-bearing NOT-yet-absorbed content to carry into the merge: (1) the H1/Leziart CaT-to-humanoid precedent with a contact-phase-gated hand-force constraint (de-risks the impulse constraint); (2) the variable-impedance × CaT-on-impulse non-stationarity foundational risk (measurement-gaming via commanded K, m_eff depends on K); (3) the 5 unused-CaT-knobs list (real γ(1−δ), time-to-death curriculum, substep detection, per-constraint contact-active EMA, multi-constraint max-agg). During verbatim merge DROP every 'A3 is not real CaT / mjlab may not expose qfrc / reset() is a no-op / crude approximation we ship' line — all falsified by shipped code.

**Claim (line 10) — DROP**
- Quote: `**Implication:** before crediting any impulse-CaT design, implement (or confirm) the real \`γ(1−δ)\` rollout. The env-side termination we ship is the crude approximation.`
- Evidence: src/tasks/hammer/rl/cat_ppo.py + cat/hook.py exist — the faithful γ(1−δ) rollout (CatPPO + CatSoftHook) is SHIPPED; the naive sampled-hard is no longer 'the shipped mechanism'
- Replacement: DROP

**Claim (line 28) — DROP**
- Quote: `**Measurement is the gating unknown:** mjlab 1.4.0 may not expose per-substep \`qfrc_constraint\` on the Entity (it doesn't, per \`velocity_bound.py\`); fallback = the windowed \`Δq̇\` velocity-jump proxy (changes what \`Λ_max\` means).`
- Evidence: velocity_bound.py:22-24 + impulse-cat-c0-findings.md:16 — qfrc_constraint IS accessible per-DoF via entity.data._joint_dof_field('qfrc_constraint'); the 'not exposed' note was corrected; the Δq̇ proxy fallback was never needed
- Replacement: DROP

**Claim (line 27) — DROP**
- Quote: `close/reset at window end (\`reset()\` must actually clear state — A3's is a no-op).`
- Evidence: impulse_bound.py:111-118 — SubstepImpulseAccumulator.reset() clears pulse/running/baseline/peak buffers per episode (not a no-op)
- Replacement: DROP

**Claim (line 5) — DROP**
- Quote: `\`src/tasks/hammer/mdp/velocity_bound.py:100\` returns \`torch.rand_like(delta) < delta\` — a **sampled-Bernoulli HARD episode cut** (wired \`time_out=False\`).`
- Evidence: velocity_bound.py:130-162 is still the naive A3 (kept as ablation reference), but 'the env-side termination we ship' framing is stale — the faithful CatPPO/CatSoftHook is the shipped soft-CaT mechanism
- Replacement: DROP

### docs/research/reward-design/JOINT_VELOCITY_BOUND_RESEARCH.md (merge → CONSTRAINED_RL_LANDSCAPE.md)

Notes: MERGE SOURCE → CONSTRAINED_RL_LANDSCAPE (joint-velocity-bound research appendix per design spec). Load-bearing NOT-yet-absorbed content to carry: (1) the failure-mechanism analysis (4.3–4.65 rad/s overshoot is chain-coupled/off-axis windup no single-joint fix can brake) — grounds the CaT-incentive vs VIC-capability split; (2) the ranked options table A–E with the DcMotor effort_limit<saturation_effort numeric bug + 'A+B together = honest hard bound' correction; (3) 'velocity is the Z1 proxy, impulse is the thesis target' framing + the forbidden raw ‖q̇‖² repo rule. During merge DROP only the 'does mjlab expose substep qfrc_constraint' de-risking item (resolved). Rest is current.

**Claim (line 41) — DROP**
- Quote: `3. **De-risk the \`Λ_j\` measurement** independently (does mjlab expose substep \`qfrc_constraint\`? else commit to the \`Δq̇\` proxy).`
- Evidence: impulse-cat-c0-findings.md:16 + velocity_bound.py:22-24 — resolved: qfrc_constraint IS exposed per-DoF; the Λ_j accumulator is shipped in impulse_bound.py
- Replacement: DROP

---

## Cluster: open questions / training plans

### docs/research/reward-design/OPEN_QUESTIONS.md (fix)

Notes: Per design spec, prune answered items to a dated 'Resolved' tail. Q1/Q2/Q4/Q8/Q10 resolved; Q3/Q6/Q7/Q9/Q11/Q12 genuinely open (need training/hardware). The '6-term/8-phase' block (lines 183-202) is the most stale — predates impact_progress and the A–M expansion. Phase-H threshold text (0.070/0.071, line 209) also contradicts the 0.027 success threshold decided earlier in the same doc (line 41) — internal contradiction.

**Claim (line 185)**
- Quote: `Independent of all open questions above, the current 6-term reward stack (\`approach\`, \`nail_driven\`, \`nail_depth_delta\`, \`completion\`, \`action_rate\`, \`joint_pos_limits\`) has been validated end-to-end via \`validate_rewards.py\`. All 8 validation phases pass:`
- Evidence: hammer_env_cfg.py live weights (CLAUDE.md) — baseline is 7 terms incl. impact_progress; validate_rewards.py:129-438 runs phases A–M (13)
- Replacement: `Independent of all open questions above, the current 7-term reward stack (\`approach\`, \`nail_driven\`, \`nail_depth_delta\`, \`impact_progress\`, \`completion\`, \`action_rate\`, \`joint_pos_limits\`) has been validated end-to-end via \`validate_rewards.py\`. The A–H phases below cover the depth/completion terms; the full current gate runs phases A–M (impact_progress = Phase I, impulse-CaT = Phase M):`

**Claim (line 168)**
- Quote: `| Q1 | 🟢 Resolved | Real-hammer measure + V1 GPU training (arrays 36472565/36472566): trained single strike clears 27 mm at 100%, press does not emerge |`
- Evidence: l6-hammer-fixture-ee.md:22-23 — NEAR_NAIL is stale after the L6 fixture move; head site moved ~59 mm, so V1 depth/success figures predate the current EE
- Replacement: `| Q1 | 🟢 Resolved (gripper-era) | Real-hammer measure + V1 GPU training (arrays 36472565/36472566): trained single strike clears 27 mm at 100%, press does not emerge. NOTE: measured pre-L6-fixture; NEAR_NAIL re-solve pending, so depths may shift |`

**Claim (line 179)**
- Quote: `**Remaining blockers before first training run:** none. All resolvable-from-code questions are answered.`
- Evidence: l6-hammer-fixture-ee.md:22 — NEAR_NAIL re-solve + XML mirror + gate re-green are PENDING before any training/eval/gate on the current fixture EE
- Replacement: `**Remaining blockers before first training run:** the L6 fixture EE change requires a NEAR_NAIL re-solve + gate re-green before training on the current tree (see [[l6-hammer-fixture-ee]]). All resolvable-from-code reward questions are answered.`

### docs/VEGA_TRAINING_PLAN.md (fix)

Notes: Fix (still live as the GPU-campaign plan; V1 results are a dated record kept in place). Two genuine stale refs above (line 31 'active branch is hammer-z1' actually appears in HANDOVER not here — re-scoped: within VEGA_TRAINING_PLAN the branch is not restated, so only the Q13-Q15 ref at line 5 is a hard fix; the branch-name row is illustrative of the pattern to grep). NOTE: line 5 also mentions 'V0' and stages V0-V3 which are still the live plan. The V1/b_strike RESULT blocks (lines 56-159) are dated records and legitimately quote old numbers (delta 0.05->0.15, 8.6 steps, etc.) as history — do NOT touch those; they carry their own dates. The plan does NOT yet mention the impulse-CaT / soft-CaT V2 arms now built on soft-cat (SubstepImpulseAccumulator, cat_impulse); a live plan should point at IMPULSE_CAT_IMPL_PLAN.md for the current V2 campaign, but that is an addition not a stale-claim fix. Verify Q-numbering against the real OPEN_QUESTIONS.md before applying the line-5 fix.

**Claim (line 5)**
- Quote: `OPEN_QUESTIONS Q2/Q3/Q6/Q11/Q13–Q15) has been blocked on hardware.`
- Evidence: CLAUDE.md lists open questions as Q2, Q3, Q6, Q7, Q9, Q11, Q12; Q13-Q15 are not referenced anywhere current
- Replacement: `OPEN_QUESTIONS Q2/Q3/Q6/Q7/Q9/Q11/Q12) has been blocked on hardware.`

**Claim (line 31)**
- Quote: `The active branch is \`hammer-z1\`.`
- Evidence: git: current branch is soft-cat; l6-hammer-fixture-ee memory + env_cfgs.py cat_impulse live on soft-cat
- Replacement: `The active branch is \`soft-cat\` (the impulse-CaT + soft-CaT development branch; \`hammer-z1\` was the earlier Phase-0 branch).`

---

## Cluster: literature (merge sources → LITERATURE.md)

### docs/research/reward-design/REWARD_LITERATURE.md (merge → LITERATURE.md)

Notes: Merge source → LITERATURE.md via VERBATIM ASSEMBLY (design spec §118 policy). The bibliography ENTRIES + 'what to steal' engineering takeaways are the load-bearing content to preserve verbatim under a '[from REWARD_LITERATURE.md]' tag; the H3/M1 opus-audit HTML-comment markers (L70,107,118,145) are provenance and should travel with their entries. DROP the stale 'as-current' framing lines flagged above (they assert live config facts that code contradicts). Note: 'nail_slide_joint.qpos > 0.09 m' L123 is Meta-World's OWN threshold (a lit fact), NOT ours — KEEP, do not confuse with our NAIL_SUCCESS_THRESHOLD=0.027. Original archived after merge per spec §54.

**Claim (line 48) — DROP**
- Quote: `**What to steal:** For Z1 training: use \`nail_depth_delta × 500\` early, then consider reducing weight once the policy reliably drives the nail.`
- Evidence: hammer_env_cfg.py:184 weight=600.0 (2000→600 A1 rebalance); no 500 anywhere in the live config
- Replacement: DROP

**Claim (line 12) — DROP**
- Quote: `The current mjlab \`nail_driven_reward\` (Gaussian on absolute depth) and \`hammer_approach_reward\` (Gaussian on distance) are NOT potential-based — they bias the policy toward hovering near high-reward states.`
- Evidence: hammer_env_cfg.py:155-178 still has approach (w=0.1) + nail_driven (w=2.0) but the augment-not-replace baseline is now 7-term + impulse arm; the 'current' framing is a 2026-05-22 snapshot pre-nail_depth_delta/impact_progress
- Replacement: DROP

**Claim (line 32) — DROP**
- Quote: `See TODO in \`hammer_z1_env/env.py\` for implementation trigger.`
- Evidence: hammer_z1_env/env.py is the legacy sibling-repo mocap env, not the mjlab training path (src/tasks/hammer/); this TODO pointer is dead in the current architecture
- Replacement: DROP

**Claim (line 160) — DROP**
- Quote: `already at −1.0 in current config`
- Evidence: hammer_env_cfg.py:223 joint_pos_limits weight=-10.0 (not -1.0)
- Replacement: DROP

### docs/research/hammering_literature_notes.md (merge → LITERATURE.md)

Notes: Primary merge source into LITERATURE.md (target does not yet exist). Body is a clean annotated bibliography (Tandon, Vu 2026 QP effective-mass, Romanyuk RPT/MPT, Adroit/DAPG/D4RL, Tool-as-Interface, Robot Drummer, Karbasi, Teramae, Orbik) + 15 harvested reward-design ideas — verbatim-mergeable, source-attributed [from hammering_literature_notes.md]. Two notes for the merge: (1) the 'welded hammer' framing (lines 132/201/205) is now EE-stale at the assembly level — L6 replaced the Unitree gripper with a custom 3D-printed fixture bracket (chain link06->fixture->hammer, gripper meshes removed, jointGripper kept nq=7; UNCOMMITTED sibling-repo change per memory l6-hammer-fixture-ee) — but this is a literature-context phrase ('vs Adroit's grasped hammer'), so it stays as history in the merged bibliography, not a fix. (2) Idea #6/§2.3 effective-mass m_e,n and idea #10 Rhythmic-Contact-Chain are the load-bearing DIRECT citations; keep verbatim. Delete only exact duplicates of the 'field is narrow/gap = physics+embodiment' framing (repeated ~4x across §1/§2.6/§2.7).

**Claim (line 181) — DROP**
- Quote: `**Add a strike-quality / damage penalty** (in MuJoCo: nail lateral deflection / tilt, off-axis contact, or excessive lateral contact force).`
- Evidence: src/tasks/hammer/hammer_env_cfg.py:194 ships impact_progress (double-gated), not a damage penalty; deep_dive_v2 §11 marks orientation/off-axis moot on position-only DiffIK (orientation_weight=0.0, hammer_env_cfg.py:121)
- Replacement: DROP

**Claim (line 17) — DROP**
- Quote: `The Z1 project sits in the **intersection**: *learning* + *impact dynamics* + *arm* + *sim-to-real* + *repetition*. Frame the gap as physics + embodiment, not "nobody has learned to hammer."`
- Evidence: Framing-only literature synthesis; no code contradiction, but duplicated verbatim in §2.6/§2.7 and deep_dive_v2 DA-pass #2 — dedupe on merge
- Replacement: DROP

### docs/research/impact_tracking_rl_litreview.md (merge → LITERATURE.md)

Notes: Second merge source into LITERATURE.md. Body's annotated bibliography (DeepMimic, AMP, HMAMP, IQL, D4RL, Stronge, Wang/Dehio/Kheddar CRB, Wang&Kheddar QP, Vu 2026, Ti 2024, Stooke PID-Lagrangian, CAPS, DREM) + §4 citation-integrity corrections are the durable, verbatim-mergeable content. The STALE layer is the architectural FRAMING only: the title, RQ (§1), §3 synthesis, and §5 all assume 'generate-then-track / upstream trajectory + RL tracking' — an architecture the supervisor walked back (CLAUDE.md authoritative-direction note; superseded by tracking_impact_impulse_design_research.md D1 = weak annealed reward-level prior). On merge, keep the per-paper annotations, DROP the generate-then-track scaffolding sentences (title, RQ, Q1-Q4 framing prose). Also note §2C 'force-sensor-measured impulse feeding the reward' is superseded by the MuJoCo-native qfrc_constraint substep accumulator (impulse_bound.py) — the newer tracking_impact_impulse doc §2.3 already has the correct 'Σ|qfrc_constraint|·h at 500 Hz' measurement, so LITERATURE.md should carry that version, not this one.

**Claim (line 1) — DROP**
- Quote: `# Literature Review — Generate-then-Track RL for Impact-Explicit Robotic Hammering`
- Evidence: CLAUDE.md + thesis_direction_update: the two-stage generate-then-track / DeepMimic architecture was WALKED BACK by the supervisor; current arch is single-policy weak-annealed reward prior (tracking_impact_impulse_design_research.md D1)
- Replacement: DROP

**Claim (line 9) — DROP**
- Quote: `What is the most defensible learning-and-control architecture and reward formulation for impact-explicit robotic nail-driving on a *position-controlled* manipulator, where the objective is to **maximize delivered impact momentum** (effective mass × axial velocity, \`m_eff·v\`) while **minimizing joint reaction impulse** — combining an upstream-generated strike trajectory with RL tracking, with a force sensor for both objective signals and observations?`
- Evidence: 'upstream-generated strike trajectory with RL tracking' = the walked-back generate-then-track framing; superseded by weak-prior single-policy (tracking_impact_impulse_design_research §2.2/D1)
- Replacement: DROP

**Claim (line 51) — DROP**
- Quote: `The intersection that is the project's novelty`
- Evidence: 'a **force-sensor-measured impulse** feeding both the impact reward and the CRB-based recoil prediction' — superseded: impulse ground truth is MuJoCo-native qfrc_constraint accumulated at substep rate (impulse_bound.py), not a force-sensor; Pinocchio/CRB cross-val deferred (memory impulse-cat-c0-findings)
- Replacement: DROP

---

## Cluster: memory index + one-liners

### .claude/…/memory/MEMORY.md (fix)

Notes: Index hooks reviewed against each file. Two hooks carry the pre-L6/gripper-era numbers (45% friction, 0.137 N·s) that c0-findings' own 2026-07-04 UPDATE supersedes with EE-dependent fixture-era values. All other one-liners (impulse-cat-deep-review, l6-hammer-fixture-ee, cat-constraints, velocity/softcat/gravcomp/etc.) match their files. The deep-review hook (line 18) correctly says 'PER-EVENT PULSE (not persistent latch)' — current.

**Claim (line 13)**
- Quote: `UPDATE: C0 empirics show the contaminant is dof-FRICTION (~45%), not the weld (~0.2); see [[impulse-cat-c0-findings]]`
- Evidence: impulse-cat-c0-findings.md:20 UPDATE 2026-07-04 revises to 41-48% spread across joints 1-3,5 (EE-dependent, was ~45% on 2,3)
- Replacement: `UPDATE: C0 empirics show the contaminant is dof-FRICTION (efc type mjCNSTR_FRICTION_DOF, ~41-48% and EE-dependent), not the weld (~0.05); see [[impulse-cat-c0-findings]]`

**Claim (line 15)**
- Quote: `object-side ∫F·dt=0.137 N·s clean GT; J_limit non-binding for gentle strikes (binds at ~21× more violent)`
- Evidence: impulse-cat-c0-findings.md:20 UPDATE 2026-07-04: fixture-era delivered=0.107 N·s (0.137 was gripper-era); numbers are EE-dependent
- Replacement: `object-side ∫F·dt≈0.107 N·s (EE-dependent, gripper-era was 0.137) clean GT; J_limit non-binding for gentle strikes (binds at ~21× more violent); re-derive via derive_impulse_thresholds.py`

### .claude/…/memory/hammer-sim-audit-fixes.md (fix)

Notes: Records the 2026-06-16 sim audit. Findings #4 (obs-clamp not solreflimit), #5 (clip_actions=1.0 bounds, DiffIK clip a no-op) still accurate. Only the gate-count line is stale (9→A–M) — same 'nine phases' poison as elsewhere. The gripper `<contact><exclude>` and gripperMover notes predate L6 but are described as that-session mechanical fixes, not current EE state.

**Claim (line 46)**
- Quote: `\`validate_rewards\` 9 phases incl. real strike`
- Evidence: validate_rewards.py now runs Phase A through M (grep shows Phases A,B,C,D,E,F,G,H,I,J,K,L,M); '9 phases' is stale
- Replacement: `\`validate_rewards\` phases A–M incl. real strike`

### .claude/…/memory/qfrc-constraint-weld-pollution.md (fix)

Notes: The body's central thesis — 'the weld DOMINATES qfrc_constraint / it's the weld baseline' — is the falsified 'weld pollution' framing the crib flags as STALE, and unlike diffik-maxdq-velocity or c0-findings this body carries NO in-file dated correction marker (the friction correction lives only in the MEMORY.md index line + c0-findings). Line 16's sub-correction is separate and CURRENT: qfrc_constraint IS exposed via _joint_dof_field, confirmed at velocity_bound.py:23-24. If instead archived as a 'contaminant investigation origin' record, superseded_by impulse-cat-c0-findings — but as a live reference it needs the inline weld→friction correction.

**Claim (line 10)**
- Quote: `the Z1 hammer XML runs an **active weld equality** ... \`qfrc_constraint = Jᵀ·efc_force\` sums over **all** active constraints (weld + joint-limits + contacts), so the weld reaction is on the arm DoFs at *every* substep and **dominates** the brief nail-impact spike.`
- Evidence: impulse-cat-c0-findings.md:13 + impulse_bound.py:19-21: the dominant off-contact contaminant is dof-FRICTION (efc mjCNSTR_FRICTION_DOF, ~41-48%), the weld contributes only ~0.05 when gravity-compensated
- Replacement: `the Z1 hammer XML runs an active weld equality ... \`qfrc_constraint = Jᵀ·efc_force\` sums over all active constraints (weld + dof friction + joint-limits + contacts). CORRECTION (C0, 2026-06-18): the DOMINANT off-contact contaminant is dof-FRICTION (efc mjCNSTR_FRICTION_DOF, constant ±frictionloss on moving joints, ~41-48% of raw Λ_j and EE-dependent), NOT the weld — the weld contributes only ~0.05 to gravity-compensated arm DoFs. The contact-sensor gate + optional baseline-subtract still handle both; see [[impulse-cat-c0-findings]].`

**Claim (line 12)**
- Quote: `accumulating \`qfrc_constraint\` over the full control step therefore measures "how hard the weld drags the arm" (≈ gross commanded motion), NOT the impact impulse ... The plausible "typical 0.01–0.2 N·m·s" magnitude is likely the weld baseline, not the impact.`
- Evidence: impulse-cat-c0-findings.md:13-14: contaminant is friction (~45%), and the shipped accumulator is contact-anchored (impulse_bound.py:69-146), not full-step; object-side clean GT ≈0.107 N·s (EE-dependent)
- Replacement: `accumulating \`qfrc_constraint\` over the full control step therefore measures gross commanded motion (dominated by dof-FRICTION, per the correction above), NOT the impact impulse. The shipped accumulator is contact-anchored at substep rate (impulse_bound.py); the weld/friction-immune object-side ground truth ≈0.107 N·s (EE-dependent, re-derive via derive_impulse_thresholds.py).`

---

## Cluster: sibling repo

### /Users/nikerane/repos/safe_impact_manipulation/hammer_z1_env/README.md (fix)

Notes: Only one dangling pointer into unitree_rl_mjlab: docs/BASELINE_AUDIT.md has been archived (now docs/archive/BASELINE_AUDIT.md) so line 7 dangles now and would still point at an archived doc after consolidation; retarget to the new docs/README.md index (the design's single current-truth index) which is co-listed right after with `docs/research/reward-design/`. All other unitree_rl_mjlab pointers resolve: scripts/train.py, scripts/play.py, scripts/list_envs.py, src/tasks/hammer/{nail_block.py,hammer_env_cfg.py,mdp/,config/z1/{env_cfgs.py,rl_cfg.py},rl/runner.py} all present; HammerOnPolicyRunner + ONNX export (rl/runner.py:26,31-38) and the 3-DoF DiffIK sanity snippet still accurate. EE note: no gripper-holds-hammer claim to flag — lines 33-34 say '7 (arm) + 1 (gripper) = 8 DOF' (jointGripper is genuinely KEPT per L6 memory, nq unchanged) and 'hammer is fixed to the EE — no grasp needed' (still a rigid weld). The stale-vs-L6 items (scene diagram omits the l6_hammer_fixture bracket in the link06->fixture->ee_center_body->hammer chain; head described as 'box 5x2x2.5 cm'+'capsule 22 cm' now a real claw-hammer mesh) are SIBLING-REPO scene geometry = out of scope under LIGHT depth, and that fixture change lives in the sibling asset XML (uncommitted), not this repo.

**Claim (line 7)**
- Quote: `**\`Unitree-Z1-Hammer\`**. Docs: \`unitree_rl_mjlab/docs/BASELINE_AUDIT.md\`,`
- Evidence: docs/BASELINE_AUDIT.md absent from live tree; only docs/archive/BASELINE_AUDIT.md exists (find . -iname '*baseline_audit*')
- Replacement: `**\`Unitree-Z1-Hammer\`**. Docs: \`unitree_rl_mjlab/docs/README.md\` (index),`

---

## Cluster: thesis / project root pointers

### docs/thesis/README.md (fix)

Notes: Banner-only body per rubric; verdict=fix ONLY for pointer lines to files that WILL MOVE. The single moving-file pointer is line 26 (CAT_DEEP_DIVE -> FAITHFUL_SOFT_CAT_IMPL_PLAN). Content claims are recorded as evidence (NOT fixed, per instruction): (a) line 39 C3 '4.3-4.65 rad/s' worst-case velocity is a legit ablation result, keep; (b) line 138 '37 CPU unit tests + verify_cat_soft.py + smoke_cat_soft.py all green' is now STALE — the 2026-07 review put the suite at 270 tests green (impulse-cat-deep-review memory) and added Phase M / derive_impulse_thresholds.py; C0-C3 is now C0-C5 with the impulse arm + composition. This is a content claim so it stays a note, not a fix, but flag it: the '37 tests / C0-C3 / only GPU results remain' status in §6 predates the entire impulse-CaT + soft-OR-composition body of work now on soft-cat. (c) §6 says the soft-CaT *implementation* is DONE and only GPU results remain — still broadly true for the velocity arm, but the IMPULSE arm (the thesis headline C2) is only at C0 log-only, which §6 doesn't mention. Recommend a status refresh in a later pass, but body edits are banner-only for the thesis doc.

**Claim (line 26)**
- Quote: `Detail: \`CONSTRAINED_RL_LANDSCAPE.md\` §3–4, \`CAT_DEEP_DIVE.md\`.`
- Evidence: docs consolidation design line 28/55: CAT_DEEP_DIVE is merged into FAITHFUL_SOFT_CAT_IMPL_PLAN.md then archived
- Replacement: `Detail: \`CONSTRAINED_RL_LANDSCAPE.md\` §3–4, \`FAITHFUL_SOFT_CAT_IMPL_PLAN.md\` (absorbs the former CAT_DEEP_DIVE).`

### CLAUDE.md (fix)

Notes: FULL fix-depth per instructions (CLAUDE.md gets rewritten later). Verified CURRENT: 7-term baseline (hammer_env_cfg.py has exactly approach/nail_driven/nail_depth_delta/impact_progress/completion/action_rate/joint_pos_limits); all live weights match code (0.1/2.0/600/8/100/-0.01/-10); 'all phases A-M' matches validate_rewards.py (Phase A line 129 through Phase M line 365-377, Phase I=impact_progress, Phase M=impulse-CaT arm); unit-test list (test_impulse_bound/test_impulse_constraint/test_delivered_impulse_reward/test_cat_soft_hook) all exist; every referenced path resolves (test_single_strike.py, verify_contact_sensor.py, render_reference.py, DEEP_RESEARCH_REPORT.md in archive, etc.); the 2026-06-17 Z1-primary direction update is current architecture. ONLY stale line is line 14 'weld-pollution blocker'. NOTE spec §L3 authority contract is NOT yet present in CLAUDE.md (Task 7 adds it) — expected, not a stale claim.

**Claim (line 14)**
- Quote: `note its top decision-to-confirm (soft-CaT vs the docs' CMDP/Lagrangian) and the weld-pollution blocker it resolves.`
- Evidence: src/tasks/hammer/mdp/impulse_bound.py:18-22 — qfrc_constraint contaminant is dof-FRICTION (~45%), weld is tiny (~0.05); IMPULSE_CAT_IMPL_PLAN.md:6 C0 findings confirm.
- Replacement: `note its top decision-to-confirm (soft-CaT vs the docs' CMDP/Lagrangian) and the qfrc_constraint-contamination blocker it resolves (the contaminant is dof-FRICTION, ~41-48% on joints 2/3, not the weld — the weld contribution is tiny ~0.05).`

---

## Cluster: non-obvious notes (no stale claims to fix, but flagged)

The following rows carry no line-level fixes but have load-bearing notes for the consolidation (archive banners, merge-carry content, or record framing). Verbatim from the audit:

### docs/research/reward-design/ORIENTATION_ROBUST_SIM2REAL_ARM.md (current)
Design/idea-capture for a future arm; explicitly 'NOT yet implemented, build after fixed-impedance impulse result' — matches CLAUDE.md sequencing and memory [[orientation-robust-vicon-arm]]. hammer_env_cfg.py:121 orientation_weight=0.0 is a forward-looking pointer, not a stale fact. Vu/EMMT (hal-05516105) citation matches CONSTRAINED_RL_LANDSCAPE. No code contradictions. Soft note: references the NEAR_NAIL reset pose the L6 fixture invalidated, but its own action is 'build later,' so not load-bearing-stale.

### docs/research/reward-design/NAIL_PRECISION_CURRICULUM.md (current)
Brainstorm/idea-capture (2026-06-23), kept per design spec End-state list. Feasibility claim (mujoco_warp runtime geom_size resize for primitive geoms) matches memory [[nail-precision-curriculum]]. Explicitly pre-spec, notes 'independent of the in-flight L6-fixture work' and the pending NEAR_NAIL re-solve. No code contradictions.

### docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md (archive)
Superseded_by: src/tasks/hammer/hammer_env_cfg.py (impact_progress term shipped; live weights) + docs/research/hammering_reward_design_deep_dive_v2.md. Archive per spec (shipped). Body is banner-only; do NOT extract line fixes. Contains STALE phase counts ("existing 8 phases pass" L87, "8 + Phase I" L113) — real gate is A-M (validate_rewards.py:129-377); banner must warn. UNABSORBED load-bearing content that would be lost to archive: the design rationale for both shipped changes (why nail_depth_delta 2000→600: +132 single-step spike inverts completion=100 hierarchy, L20-23; ImpactProgressTerm formula/axis-verification/finite-diff-not-site_vel_w choice, L34-52) is NOT captured in any current living doc — only the code implements it. Recommend a one-line pointer from a living doc to this term's rationale before archiving. Design spec §57 lists it for archive. NAIL_SUCCESS_THRESHOLD/live-weight facts here are superseded by code.

### docs/research/reward-design/OPUS_AUDIT.md (banner-check)
Superseded_by: n/a — dated audit record (2026-05-22); kept as historical evidence per spec §57 archive-with-banner. NOT yet in docs/archive/ (archive README L17 explicitly lists OPUS_AUDIT as 'still live'). Design spec §57 says ARCHIVE; body is banner-only. NO banner present, and docs/archive/README.md does NOT describe it (it is under the 'Still live (NOT archived)' section, L16-20) — both need fixing when moved. STALE-as-current numbers inside (all frozen 2026-05-22 findings, legitimately historical once bannered): 'all 8 phases pass' L10, '6-term baseline' L10/L34-37 (now 7-term + impulse arm), '9-term SPEC' references. These are dated audit findings so they survive under a 'research record'/archive banner, but the doc currently reads as live. Load-bearing content already absorbed elsewhere: H3 fabricated-author correction lives verbatim in REWARD_LITERATURE.md:118-121; augment-not-replace strategy is in CLAUDE.md. Nothing unabsorbed and load-bearing.

### docs/research/reward-design/PEER_REVIEW_v2.md (banner-check)
Superseded_by: n/a — dated peer-review record (2026-06-02); kept as historical evidence per spec §57. NOT yet in docs/archive/ (archive README L17 lists PEER_REVIEW_v2 as 'still live'). Design spec §57 says ARCHIVE; body is banner-only per spec. NO banner present; docs/archive/README.md does NOT describe it. Reviews DEEP_RESEARCH_REPORT.md which is ALREADY archived (docs/archive/DEEP_RESEARCH_REPORT.md) — inbound references to 'DRR' are pre-archive. STALE-as-current phrase 'validate_rewards.py 9 phases' L128 (real: A-M). These are frozen review findings, legitimately historical under a banner. Load-bearing content: the CRITICAL press-exploit finding is already absorbed into TRACKING_IMPACT_IMPULSE_IMPL_PLAN (T3 one-payout window) and OPEN_QUESTIONS Q1 — nothing unabsorbed and load-bearing that would be lost.

### docs/research/reward-design/REAL_HAMMER_PLAN.md (record)
Superseded_by: n/a — executed T0.5 record (INTEGRATED & RECALIBRATED 2026-06-17); kept for provenance. Executed plan → keep as dated evidence-record with banner variant 'dated research record, kept for provenance'; body untouched (do NOT extract line fixes). Design spec §60 lists REAL_HAMMER_PLAN for archive 'if executed' — it IS executed (status L3 'INTEGRATED, striking & RECALIBRATED'). Numbers verified against code: NAIL_SUCCESS_THRESHOLD=0.027 (hammer_env_cfg.py:12,213) matches L7 — CURRENT. STALE-as-current inside: 'validate_rewards.py (10 phases)' L33 (real gate is A-M = 13 phases; legit as a 2026-06-17-dated figure under banner), '155 pytest pass' L7 (now 270). EE CAVEAT: this doc describes the gripper-era claw-hammer mount (grasp #10, head 0.5 kg); the L6 fixture-bracket EE change (chain link06→fixture→hammer, gripper removed) is NOT reflected here and post-dates it — banner should note the EE has since changed and NEAR_NAIL re-solve is pending. UNABSORBED load-bearing: the recalibration invariant 'press-stall < threshold ≤ best clean strike' + I_ref≈0.32 N·s + best-strike=28.3 mm rationale (L7,31) — verify these are mirrored in OPEN_QUESTIONS Q1 before this becomes a pure record.

### docs/research/reward-design/REWARD_VALIDATION_METHODOLOGY.md (archive)
Superseded_by: docs/research/reward-design/validate_rewards.py (module docstring + phase comments A-M are the live source of truth). Design spec §58 archive 'if validate_rewards.py docstring suffices' — the script's phase comments (validate_rewards.py:129-377, Phases A-M) now supersede this prose. Body is banner-only; do NOT extract line fixes. STALE-as-current inside: every '× 500' arithmetic (Phase C L76-77 '+0.010 × 500 = +5.0', Phase D L83, Phase E L90) uses the OLD nail_depth_delta weight 500 — live weight is 600 (hammer_env_cfg.py:184), so the asserted expected values are wrong against the current gate; and the doc pre-dates the impact_progress/imitation/impulse-CaT phases so its 6-phase A-F walkthrough is a subset of A-M. Banner must warn these numbers are stale. UNABSORBED load-bearing: the methodology PRINCIPLE ('reward = deterministic function, unit-test without training'; the bug-class table L18-27; 'tests consistency not correctness' L124) is conceptual guidance NOT captured in the script docstring — if archived, a one-line pointer or a short 'why' block should survive in a living doc so the rationale isn't lost.

### docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md (archive)
Superseded_by: docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md (soft-CaT impulse machinery, the shipped path) + FAITHFUL_SOFT_CAT_IMPL_PLAN.md. Design spec §59 archive with 'absorbed portions noted'. Body is banner-only; do NOT extract line fixes. This plan's T3/T4/T5 impulse machinery was SUPERSEDED by the soft-CaT approach actually shipped (impulse_bound.py SubstepImpulseAccumulator + cat/hook.py + IMPULSE_CAT_IMPL_PLAN.md) — the plan proposed a windowed-impulse REWARD term + Lagrangian escalation, whereas the code implements per-event-pulse Λ_j + soft-CaT termination (env_cfgs.py:199-250). STALE-as-current: 'validate_rewards.py 10/10 phases' L5 and '(9 phases)' L27 (real: A-M=13); '147 unit tests' L5 (now 270); I_ref≈0.32 N·s L16 is gripper-era (fixture-era is different, EE-dependent — re-derive via derive_impulse_thresholds.py). UNABSORBED load-bearing that would be lost: (1) the A-BASE/A-PRIOR/A-RES/C-PEN/C-LAG/I-MOM ablation TABLE (L117-124) and the degenerate-tracking guard metrics (press-watchdog, deviation-norm, beat-the-reference L66-69) — the experimental design; (2) the A-TRACK r_imit tracking-prior arm (T2, weak annealed reward prior) is IMPLEMENTED (rewards imitation term + reward_curriculum in hammer_env_cfg.py:246-267, Phase K in validate_rewards.py) but its rationale/decision-changelog lives ONLY here. Recommend the ablation table + r_imit decisions be pointed-to or lifted into a living plan before archiving, else the tracking-arm design is orphaned.

### docs/research/hammering_lit_sweep_RUNBOOK.md (archive)
Superseded_by: docs/research/reward-design/LITERATURE.md. Pure re-runnable process playbook (workflow scripts, STOP checkpoints, integrity/tiering rules) — no code claims, so banner-only per archive rules. Load-bearing content NOT absorbed by any living doc that would be lost to archive: (1) the §6 tiering ladder [E1]/[E2]/[E2*]/[E3]/◔/[ANALOGY]/[PROPOSAL] and 'gray zone = FAIL' + 'never blend authors across papers' integrity rules that govern how every citation in the corpus was verified — LITERATURE.md should carry a one-line pointer to these; (2) the §1 EXCLUDE / §2 SATURATED corpus-boundary lists (what has already been searched: Adroit ecosystem, CN/JP empty, no native hammer in robosuite/ManiSkill) — needed so the next sweep starts where this ended. Both are provenance/method, not facts that contradict code, so archive-with-banner is correct; recommend archive/README.md note that this is the sweep methodology behind LITERATURE.md. No stale-vs-code claims (contains no impulse/CaT/gate machinery).

### docs/research/hammering_reward_design_deep_dive_v2.md (record → BANNER-ONLY)
Dated evidence-record (2026-05-29), banner-variant 'dated research record, kept for provenance' — body untouched, no line fixes extracted per rules. Its central RECOMMENDATIONS have since SHIPPED, so the doc reads as historically-forward-looking: it proposes impact_progress as '[PROPOSAL]/NEW' (lines 37,171,237) and nail_depth_delta=2000->600 rebalance (line 217/236) — both now LIVE (hammer_env_cfg.py:194 weight=8.0; line 184 weight=600.0), and validate_rewards.py now runs phases A-M not the '8 phases' this doc cites (line 45). These are not fixes (record body is frozen) but they ARE the poison-phrase seeds for the freshness guard so a future grep doesn't read this doc's proposal-tense as current. Load-bearing content already absorbed by living docs (IMPACT_PROGRESS_IMPL_SPEC, FAITHFUL_SOFT_CAT / IMPULSE_CAT plans, CONSTRAINED_RL_LANDSCAPE): the double-gated impact term, PBRS-approach, privileged critic, safety-as-constraints, reward-hacking checklist §11. Content NOT yet absorbed that would be lost if this weren't kept: the §10 ablation matrix (A0-A13) and §11 env-specific reward-hacking taxonomy are the most re-usable un-absorbed assets — keep as record. Banner-only.

### docs/research/tracking_impact_impulse_design_research.md (record → BANNER-ONLY)
Dated evidence-record (2026-06-10), ALREADY carries an in-file scope banner (line 7: 'EVIDENCE STATUS & SCOPE BANNER … No RL training was run … hypothesized-untested') — so it self-identifies; recommend upgrading that to the standard 'dated research record, kept for provenance' banner-variant, body untouched, no line fixes. This is the doc whose D1-D6 decisions define CURRENT direction (weak annealed reward prior; windowed axial impulse one-payout-per-event depth-gated; excess-over-threshold penalty from ratings ladder; CMDP/PID-Lagrangian target form), and its §2.3 substep-rate qfrc_constraint measurement is the CORRECT one that should feed LITERATURE.md (superseding impact_tracking_rl_litreview's force-sensor version). Caveats for the record banner / freshness guard: (1) it predates the soft-CaT decision — its §3.1/§4 present PID-Lagrangian/CMDP as the constraint machinery, whereas the constraint mechanism actually adopted is soft-CaT (γ(1-δ), Khadiv confirm pending; memory soft-cat-scale-positives-decision, impulse_bound.py/hook.py). Not a fix (record is frozen + it correctly frames CMDP as one option), but note it so the guard treats 'PID-Lagrangian as the chosen mechanism' as history. (2) All numeric thresholds here (Repeated-Peak ~2x rated, 10^4-event budget, window W) are correctly framed as to-be-derived (Q16, derivation recipes mandated in §5 DA-1), so they are NOT asserted-as-constant and pass the EE-dependence rule. Load-bearing un-absorbed content kept by this record: the §2.1-2.6 impact/tracking/impulse/VIC citation axes (Biemond, reference-spreading van Steen line, Yang&Posa impact-invariant, Khurana hitting-flux, Varin 2019 impedance-hammering, VICES, Ma badminton N-P3O) and the §4.1 D1-D6 decision table — these are the direct upstream of IMPULSE_CAT_IMPL_PLAN and the VIC future arm; the record is where they live until a living doc absorbs them. Banner-only.

### docs/FUTURE_UPDATES.md (archive)
Superseded_by: docs/results/2026-06-17_b_strike.md + docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md. Archive: nearly all items executed or superseded. Items 1/2a/5/6a are APPLIED (delta_pos_scale is now 0.15 not 0.05 per env_cfgs.py:78 + b_strike; frictionloss/goal 0.032 shipped). Item 4 impact_velocity_bonus explicitly superseded by impact_progress. Body is stale (still cites delta_pos_scale=0.05, 7-term reward, pre-impulse-CaT). Load-bearing content NOT yet absorbed elsewhere: item 2b (nail xy randomisation) and item 3 (starting-pose curriculum) remain genuinely open engineering TODOs with concrete implementation sketches — these should be carried into OPEN_QUESTIONS or a live TODO before archiving, or they are lost. Everything else is done. superseded_by chosen because b_strike records the delta/friction outcomes and the tracking plan owns the reward-term future work.

### docs/HANDOVER.md (archive)
Superseded_by: docs/thesis/README.md + docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md + CLAUDE.md. Archive: this is a 2026-06-09/10 cold-start handover whose entire framing is superseded. Wrong on multiple current-truth axes: (1) frames Z1 as a throwaway 'Phase 0 diagnostic' and says 'start fresh on the G1' — CLAUDE.md's 2026-06-17 DIRECTION UPDATE makes Z1 the PRIMARY platform where the full thesis machinery (soft CaT + impulse constraint + VIC) is built first; (2) §7 Path D 'start the G1 work' contradicts that; (3) active branch 'hammer-z1' is now 'soft-cat'; (4) nail range '0-7.5 cm' / '7 cm' threshold is stale (now 0.032 goal / 0.027-0.030 success per FUTURE_UPDATES + rewards.py NAIL_GOAL_DEPTH); (5) 'validate_rewards.py runs nine phases' is stale (now A-M = 13 phases, validate_rewards.py:129-377); (6) delta_pos_scale=0.05 assumptions throughout; (7) DEEP_RESEARCH_REPORT 'modified but uncommitted' / IMPACT_TRACKING 'UNTRACKED' pointers now point at docs/archive/. Load-bearing content NOT yet fully absorbed: the press-exploit narrative and the 'why the action space is the binding constraint' physics writeup (§2, §4) — but these ARE now captured in thesis/README.md C3 + results/*, so safe to archive. Whole doc is banner-only on archive; do not line-fix.

### docs/results/README.md (record)
Record/current: the README already states 'one Markdown file per campaign/run' and 'YYYY-MM-DD_<run-name>.md' — the dated-records-by-design convention the design spec (line 39) wants. No stale claims; the index rows (b_strike, velocity-bound ablation, a_base/a_track) correctly point at dated files. Banner-variant 'dated research record, kept for provenance' is appropriate but body is fine as-is. Nothing load-bearing at risk.

### docs/results/2026-06-17_b_strike.md (record)
Record (dated evidence, 2026-06-17). All numbers (4.3-4.65 rad/s worst-case, 1.2 m/s, delta 0.15) are legitimate historical measurements of that run — banner-only, body untouched per rubric. §6 points at '../research/reward-design/JOINT_VELOCITY_BOUND_RESEARCH.md' which the consolidation design (line 29) folds INTO CONSTRAINED_RL_LANDSCAPE.md — that inbound reference will break when JOINT_VELOCITY_BOUND_RESEARCH is merged+archived; note for the Phase-C link-check, but not a body edit here (records keep original prose). Load-bearing content already absorbed into thesis/README.md C3 and the velocity_bound_ablation record, so nothing lost.

### docs/results/2026-06-17_velocity_bound_ablation.md (record)
Record (dated evidence, 2026-06-17). A1-A4 numbers are that run's real measurements; banner-only. Two inbound refs will break under consolidation: '../research/reward-design/CAT_DEEP_DIVE.md' (line 45, being merged into FAITHFUL_SOFT_CAT_IMPL_PLAN per design line 28/55) and '../research/reward-design/JOINT_VELOCITY_BOUND_RESEARCH.md' (line 7, merged into CONSTRAINED_RL_LANDSCAPE) — flag for Phase-C link-check, not a record-body edit. Also note the file has a DUPLICATED 'Artefacts' section (lines 64-68 and 70-74) — a pre-existing copy-paste artifact; harmless but worth a one-line dedup if ever touched. Content (chain-coupled residual finding) is fully absorbed into thesis/README.md C3, so archive-safe as provenance.

### docs/archive/README.md (banner-check)
Archive index (banner-check scope). It has a header explaining the archive but NOT the standard '> ⚠️ ARCHIVED' first-line banner (it is the index, not an archived doc, so that is arguably fine). It describes 5 archived docs (DEEP_RESEARCH_REPORT, RECOMMENDED_REWARD_SPEC, REWARD_DESIGN_MATRIX, IMPACT_TRACKING_REWARD_SPEC, BASELINE_AUDIT) with why+superseded-by. STALE against the consolidation plan: the 'Still live (NOT archived)' list (lines 16-20) names OPUS_AUDIT, PEER_REVIEW_v2, REWARD_VALIDATION_METHODOLOGY, IMPACT_PROGRESS_IMPL_SPEC, hammering_lit_sweep_RUNBOOK, TRACKING_IMPACT_IMPULSE_IMPL_PLAN, REAL_HAMMER_PLAN as live — but the design spec (lines 57-61) marks most of those for archival. This index must be regenerated in Phase B with one row per newly-archived file (incl. HANDOVER, FUTURE_UPDATES, and the 5 merge-source docs). Not a code-staleness issue; a consolidation-bookkeeping gap.

### docs/archive/BASELINE_AUDIT.md (banner-check)
Superseded_by: docs/VEGA_TRAINING_PLAN.md + docs/HANDOVER.md. MISSING the '> ⚠️ ARCHIVED' first-line banner — starts directly with '# Z1 Hammer — Baseline Audit' (line 1). The archive/README describes it (row: '2026-05-22 point-in-time validation snapshot', superseded by VEGA_TRAINING_PLAN + HANDOVER). ACTION for Phase B: prepend the L2 banner. Load-bearing content not elsewhere: nothing — it is a 2026-05-22 smoke-test snapshot fully superseded by later results records.

### docs/archive/DEEP_RESEARCH_REPORT.md (banner-check)
Superseded_by: docs/research/hammering_reward_design_deep_dive_v2.md. Has a '⚠️ Thesis-reframing note (2026-06-10)' + a 'See also v2' pointer at the top, but NOT the standard '> ⚠️ ARCHIVED <date> — superseded by <path>' banner mandated by L2. The reframing note conveys the same do-not-act intent but the poison-guard/hook keys on the standard banner phrasing. archive/README describes it (superseded by v2 deep dive). ACTION: normalise to the standard ARCHIVED banner (can keep the reframing note below it). Content is fully superseded by v2; nothing load-bearing lost.

### docs/archive/IMPACT_TRACKING_REWARD_SPEC.md (banner-check)
Superseded_by: docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md. Has a '> ⚠️ HISTORICAL (2026-06-10)' banner block (lines 3-5) with do-not-implement guidance + superseded-by reasoning — satisfies the L2 self-identification intent, though the keyword is 'HISTORICAL' not 'ARCHIVED'. archive/README describes it (the generate-then-track architecture the supervisor walked back; superseded by TRACKING_IMPACT_IMPULSE_IMPL_PLAN). Recommend normalising 'HISTORICAL' -> the standard 'ARCHIVED' banner keyword for the freshness-guard/hook to match consistently. No load-bearing content lost (the surviving r_imit prior + delivered-impulse machinery live in the tracking plan + rewards.py).

### docs/archive/RECOMMENDED_REWARD_SPEC.md (banner-check)
Superseded_by: src/tasks/hammer/hammer_env_cfg.py + docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md. MISSING the standard '> ⚠️ ARCHIVED' first-line banner — starts with '# Recommended Reward Specification' then an html opus-audit comment (lines 8-11). archive/README describes it ('aspirational 9-term spec; implemented 7-term code is source of truth'). This is a known poison reservoir: CLAUDE.md explicitly warns the 9-term 'complete config' is aspirational vs the live 7-term. ACTION: prepend the ARCHIVED banner. Load-bearing content not elsewhere: none — the 'trust the code not the spec' verdict + augment-not-replace strategy is already captured in CLAUDE.md.

### docs/archive/REWARD_DESIGN_MATRIX.md (banner-check)
Superseded_by: docs/archive/RECOMMENDED_REWARD_SPEC.md. MISSING the '> ⚠️ ARCHIVED' first-line banner — starts with '# Reward Design Matrix' (line 1). archive/README describes it ('lightweight term table; duplicates RECOMMENDED_REWARD_SPEC §3-4', superseded by RECOMMENDED_REWARD_SPEC which is itself archived). ACTION: prepend the ARCHIVED banner. Note the superseded-by target is itself an archived doc — fine for provenance but the chain terminates in archive; the true current source is hammer_env_cfg.py. Content (the 🔄 replace / ➕ add term matrix) is aspirational and fully superseded; nothing load-bearing lost.

### README.md (current)
FULL fix-depth audit — no stale claims found. Reward design section is CURRENT against code: 7-term weighted sum (line 39) matches hammer_env_cfg.py; all weights match (approach +0.1, nail_driven +2.0, nail_depth_delta +600, impact_progress +8, completion +100, action_rate -0.01, joint_pos_limits -10); the changelog note (line 81) that strike_vel was removed is correct (no strike_vel in src/tasks/hammer/); #1 rebalance 2000->600 matches hammer_env_cfg.py:184. README does not touch impulse-CaT / Lambda / EE / gate-number territory so none of the CRIB poison phrases apply. Pointers (docs/research/reward-design/, docs/archive/, hammering_reward_design_deep_dive_v2.md §4.3/§5.A) all resolve. Generic upstream mjlab usage sections (train/play/deploy) are boilerplate and unaffected.

### thesis_synthesis.md (record)
Banner-only (record: 'dated research record, kept for provenance', 2026-06-10). Body untouched per instructions. STALE POINTERS/FRAMING (would mislead if acted on, absorb into index/CLAUDE before archiving originals): (1) repeatedly frames the Z1 as lacking variable impedance + impulse constraint and recommends 'start fresh on the G1' / 'don't retrofit VIC into the Z1' (§3, §7.5) — DIRECTLY superseded by CLAUDE.md's 2026-06-17 direction update (Z1 is now primary, build full machinery incl. impulse constraint + variable impedance ON the Z1). (2) references docs/HANDOVER.md (exists at docs/, will move to archive Task 5) and PEER_REVIEW_v2.md as live sources. (3) claims Z1 'has no impulse computation / no CMDP machinery' (§3) — now falsified: src/tasks/hammer/mdp/impulse_bound.py + cat/hook.py ship the substep impulse accumulator and soft-CaT. LOAD-BEARING content not yet in a living doc: the axis-by-axis before/after architecture table (§2.3) and the 4 open supervisor questions (§7) — thesis/README should own these before this record is relegated. Recommend record banner NOT contradict CLAUDE.md; if any pointer is repaired it is banner-only.

### thesis_direction_update.md (record)
Banner-only (record: 'dated research record, kept for provenance'). Body untouched. Content is the supervisor-conversation architecture (single policy, variable impedance, online RL) and is largely still the authoritative direction — CLAUDE.md line 11 already points to it as 'the current architecture'. No code-contradicting claims (it is forward-looking G1 architecture, not Z1 implementation status). The ONE tension with CLAUDE.md: §2 says 'still fixed-base arm first, locomotion as stretch' and treats G1 as the target platform, whereas CLAUDE.md's 2026-06-17 update makes the Z1 the primary build platform first — this is sequencing nuance, not a contradiction (G1 remains the ultimate target). LOAD-BEARING and not fully mirrored elsewhere: the CBO relocation reasoning (§3E) and the safety-hardness-as-explicit-choice framing (§3D) — thesis/README should carry these. Keep as record; do not banner-edit the body beyond the record banner (non-goal guard, spec §Risks).

### thesis_handoff_brief_original.md (record)
Banner-only (record: 'dated research record, kept for provenance'). Already carries a self-identifying NOTE (line 5) pointing to thesis_direction_update.md as the superseding architecture — good. Body untouched. Architecture sections (two-level SURE+RL, §arc phases 3-4) are explicitly superseded and self-flagged; conceptual pillars (impulse-not-energy, preparation-not-reaction, hitting flux — §A-H) survive and are the reason to keep it. No code-contradicting claims (it predates all Z1 impl). LOAD-BEARING not yet in a living doc: the must-read paper map with arXiv IDs (Bogdanovic 1907.07500, SURE 2602.06864, Konno 2011, Wang&Kheddar 2006.01987) and the conceptual-pillar derivations — these belong in thesis/README before the brief is relied upon only as history. The existing inline NOTE means the record banner is somewhat redundant but harmless.

### docs/superpowers/plans/2026-06-17-z1-single-strike-fix.md (archive)
Superseded_by: docs/archive/2026-06-17-z1-strike-not-press-redesign-design.md. Executed 2026-06-17 plan (superseded_by the design record 2026-06-17-z1-strike-not-press-redesign-design.md; all tasks landed per git log b_strike run). Archive with banner, no line fixes (body is history). STALE if read as current (why it must be quarantined): (1) says validate_rewards is 'phases A-K' / 'PASS for every phase A–K' (Steps 6, Task4) — gate is now A-M. (2) Task 2 max_dq velocity rail is SUPERSEDED by its own inline CORRECTION (max_dq is not a clean velocity limit; rail reverted) and by the design record. (3) delta_pos_scale=0.15 single-strike framing. LOAD-BEARING content not yet absorbed into a living doc: the clamped_nail_depth() helper rationale + Phase K/L overshoot-clamp gate (shipped as validate_rewards Phase L) and the diffik max_dq≈10·max_dq calibration — the latter lives in the diffik-maxdq-velocity memory but not in any living reward-design doc. superseded_by names the design record which is itself being archived; ultimate live home is docs/README index 'How we got here' timeline.

### docs/superpowers/plans/2026-06-17-r_imit-tracking-reward.md (archive)
Superseded_by: docs/archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md. Executed 2026-06-17 plan (A-TRACK/r_imit arm landed: ImitationPriorTerm + Unitree-Z1-Hammer-Track are in hammer_env_cfg.py:244 + rewards.py). Self-labels 'Transient artifact: delete after the work lands' (line 5) — archive rather than delete per spec non-goal (no deletions except graphify-out/). Banner, no line fixes. STALE if read as current: (1) it plans validate_rewards 'Phase K' for r_imit, but in the shipped gate r_imit is Phase K and the overshoot-clamp moved to Phase L, and the impulse arm is Phase M — the plan's phase-letter map is pre-final. (2) the ImitationPriorTerm docstring's per-episode contact latch here uses 'latch persists for the rest of the episode' language which is CORRECT for r_imit (episode latch) — do NOT confuse with the impulse_bound PULSE semantics; not a stale claim, just a collision risk for a poison-regex. LOAD-BEARING not yet in a living doc: the r_imit budget rule (cumulative w0*Sum < 0.35*completion) and the hover-at-apex known-risk — these live only in the term docstring + TRACKING_IMPACT_IMPULSE_IMPL_PLAN (itself being archived). superseded_by = the tracking design plan.

### docs/superpowers/specs/2026-06-17-z1-strike-not-press-redesign-design.md (archive)
Superseded_by: docs/README.md (How we got here timeline); direction update in CLAUDE.md supersedes the Phase-0/G1-out-of-scope framing. Executed 2026-06-17 design+decision record (the 'make it strike, kill pressing' redesign). Its own inline TRAINING RESULT + CORRECTION blocks already record that Option A (self-limiting, no rail) was FALSIFIED and delta_pos_scale=0.15 amplifies to 4.3-4.65 rad/s worst-case. Archive with banner (dated decision record), no line fixes. STALE if read as current: (1) TL;DR pt4 / §5 'set max_dq = 3.1415*dt = 0.0063' is superseded by its own CORRECTION (max_dq craters the arm; qvel≈10·max_dq). (2) §3 quotes honest-speed ceiling 3.44 m/s and the delta_pos_scale->head-speed sweep as current physics — these are EE/posture-dependent gripper-era measurements and predate the L6 fixture migration; must not be treated as constants. (3) 'Z1 hammer is the Phase-0 fixed-base diagnostic / G1 out of scope' framing (§1) is superseded by the 2026-06-17 CLAUDE.md direction update (Z1 primary, full machinery). LOAD-BEARING not yet in a living doc: the multi-lens 'don't build T3-T5 on the Z1' decision (§2) and the multi-strike Option-M trigger analysis (§6, work-vs-KE 0.8 J ceiling) — these are the durable rationale for why the impulse-CaT arm shipped as log-only; the docs/README 'How we got here' timeline should link this record.

### docs/superpowers/plans/2026-07-05-docs-consolidation.md (current)
This is the execution plan for THIS consolidation effort — explicitly out of audit scope per instructions ('the two 2026-07-05 consolidation docs = current, do not audit their claims'). No stale claims extracted. Its POISON_PHRASES list (Task 2) and classification match the spec; it is the live driver of the work. Left untouched.

### docs/superpowers/specs/2026-07-05-docs-consolidation-design.md (current)
MISSING FROM AUDIT — flag to controller. (Found by the completeness check; no row in the audit JSON. This is the second of the two 2026-07-05 consolidation docs, explicitly out of audit scope per instructions — treated as current, do not audit its claims.)

### Memory one-liners with no stale claims (current)

- .claude/…/memory/cat-constraints-as-terminations.md — Foundational reference for the CaT mechanism (Chane-Sane IROS 2024): δ=max_p·clip(c+/c_max), reward·(1−δ), soft/hard p_max. Matches velocity_bound.py CaTJointVelConstraint exactly. The 'moves to the G1 humanoid' framing predates the 2026-06-17 Z1-primary direction update but is stated as thesis-arc context (rehearsal→G1), not a code claim; the constraint machinery it describes is what shipped on the Z1. No falsified code fact.
- .claude/…/memory/diffik-maxdq-velocity.md — head speed ≈0.18·delta_pos_scale/dt; max_dq caps qvel at ~(kp/kd)·max_dq (kp/kd=10); delta=0.15 self-limits OPEN-LOOP only, closed-loop policy hits 4.3–4.65. All corroborated by env_cfgs.py:79-85 and z1-velocity-bound-finding. The description one-liner still says 'self-limits at 2.41 rad/s' but the body (line 16) explicitly FALSIFIES the closed-loop self-limiting with a dated correction, so it is a current record-with-correction.
- .claude/…/memory/hammer-site-off-face.md (record) — Dated (2026-06-15/16) root-cause record for the gripper-era hammer_head_site fix (grasp #10, face=c4, site 0.0326 0.132 -0.0004, NEAR_NAIL joint values, validate A–J). ALL of it is pre-L6: l6-hammer-fixture-ee.md moved the hammer body -0.015→-0.073 (~59mm) and the site/NEAR_NAIL are now STALE-pending-re-solve. Body is banner-only. Load-bearing content NOT yet in a living doc: the reliable-geometry method (env.sim.mj_model + mj_geomDistance, geom math outside MuJoCo's reframe is garbage) and face=c4/claw=c1+c2 decomp identity — would be lost; l6-fixture-ee should absorb the identity + method before this is archived.
- .claude/…/memory/impulse-cat-c0-findings.md — Authoritative C0 record. Carries the dated 2026-07-04 UPDATE (line 20) that supersedes ALL its own gripper-era numbers (57.3ms→38ms, J_limit [3.44,6.88..]→[2.28,4.56..], seed 0.16→0.1020, 0.137→0.107 N·s) with EE-dependent fixture-era values and the 'never hardcode, re-run derive_impulse_thresholds.py at C2' directive — exactly the correction-marker pattern the rubric preserves as CURRENT.
- .claude/…/memory/impulse-cat-deep-review.md — The single most current (2026-07-05) authority for the impulse-CaT arm and it matches branch soft-cat exactly: PER-EVENT PULSE, EPISODE-CUMULATIVE delivered + torch.maximum credit, construction guards + log-only hard invariant, cat_soft∪cat_impulse composition, first-violation seeding + imp_seed-as-floor, per-event cap event_window_substeps=25, _joint_count slice-safe, phases A–M, 270 tests, reduce='last' episode-peak metric.
- .claude/…/memory/l6-hammer-fixture-ee.md — Records the 2026-06-22 EE change (gripper→3D-printed fixture, chain link06→fixture→hammer, gripper meshes removed, jointGripper kept nq=7, hammer body pos -0.015→-0.073). Matches the crib exactly, including that it is an UNCOMMITTED sibling-repo change with NEAR_NAIL re-solve + mocap mirror + verify still PENDING. This is the doc that makes hammer-site-off-face's site/NEAR_NAIL numbers stale.
- .claude/…/memory/nail-precision-curriculum.md — Explicitly a brainstorm (2026-06-23), 'not yet specced/implemented', pointing to NAIL_PRECISION_CURRICULUM.md. Feasibility claim (mujoco_warp runtime geom_size per-env resize, primitives only) is a probe result, not a code-state claim. Independent of L6.
- .claude/…/memory/orientation-robust-vicon-arm.md — Planned future arm (2026-06-18), scoped AFTER the fixed-impedance impulse-CaT result — consistent with CLAUDE.md sequencing. 6-DoF vs VOT-5, Vicon-only state. The one concrete code anchor (orientation_weight=0.0 / position-only at hammer_env_cfg.py:121) is stated as current-baseline; env is still position-only DiffIK.
- .claude/…/memory/scene-xml-physics-desync.md — Nail gravcomp='1' fix + two-XML sync (nail_block_scene.xml training vs hammer_nail_scene.xml viz). Gravcomp-on-nail + frictionloss-gives-no-static-hold + test_nail_physics regression all remain valid physics facts. No code claim falsified.
- .claude/…/memory/soft-cat-scale-positives-decision.md — Decision-1 (scale-positives-only: reward=r_total−δ·r_pos, _NEG_TERMS unscaled) matches hook.py:32-35,149-160 and the MF-3 guard exactly; Decision-7 terminal/timeout convention matches. Status line 'C0–C3 implemented, 37 CPU tests, GO for C4' is a dated point-in-time snapshot (now 270 tests, C5 composed) but framed as status, not a current invariant.
- .claude/…/memory/softcat-velocity-result.md — Dated GPU result (2026-06-18, run 36557770): soft-CaT mean peak |q̇| 2.48–2.65 vs a_base 3.54, 100% success 1.21 m/s, but worst-case still >π → motivates VIC + substep impulse. Experiment measurements, not code claims.
- .claude/…/memory/z1-arm-gravcomp.md — gravcomp=1 on all robot bodies in get_spec (z1_constants.py), matches robosuite/real-Z1, fixes zero-action droop. The gripper-actuator side-effect note predates L6 gripper-mesh removal, but jointGripper + actuator are KEPT per l6-hammer-fixture-ee (nq=7), so the gripper-chatter fix is still live.
- .claude/…/memory/z1-hardware-limits.md — Real Z1 URDF limits: joint velocity 3.1415 rad/s all joints, torque 60 N·m joint2 / 30 N·m joints 1,3,4,5,6. Matches velocity_bound.py:41 Z1_JOINT_VEL_LIMIT=3.1415. Sim reproduces torque + position ranges, no native velocity limit.
- .claude/…/memory/z1-velocity-bound-finding.md — Dated 2026-06-17 Phase-0 finding: b_strike 4.3–4.65 rad/s >π; A1–A4 ablation, no single bound fixes chain-coupled worst-case; adopt CaT. Ablation arms committed as flag-gated tasks — confirmed live in env_cfgs.py. Results-record; no code claim falsified.

### Sibling one-liners with no stale claims (current)

- /Users/nikerane/repos/safe_impact_manipulation/CLAUDE.md — Sibling-repo graphify-usage rules only. No pointers into unitree_rl_mjlab, no EE claims, nothing that dangles after consolidation. The shared design deletes graphify-out/ in unitree_rl_mjlab; this file governs the SIBLING repo's own graphify-out/ (out of scope).
- /Users/nikerane/repos/safe_impact_manipulation/README.md — All pointers into unitree_rl_mjlab resolve: repo + `hammer-z1` branch (branch still exists), task `Unitree-Z1-Hammer` (registered in src/tasks/hammer/config/z1/__init__.py:10), scripts/train.py + scripts/play.py (present), hammer_z1_env/README.md, and unitree_rl_mjlab/docs/. No dangling refs. No gripper-holds-hammer claim.

---

## Inbound references

Count of inbound hits per doc/asset (filename -> hits):

- IMPULSE_CAT_IMPL_PLAN.md -> 13
- FAITHFUL_SOFT_CAT_IMPL_PLAN.md -> 19
- CONSTRAINED_RL_LANDSCAPE.md -> 14
- OPEN_QUESTIONS.md -> 18
- ORIENTATION_ROBUST_SIM2REAL_ARM.md -> 5
- NAIL_PRECISION_CURRICULUM.md -> 2
- CAT_DEEP_DIVE.md -> 10
- JOINT_VELOCITY_BOUND_RESEARCH.md -> 9
- IMPACT_PROGRESS_IMPL_SPEC.md -> 6
- OPUS_AUDIT.md -> 7
- PEER_REVIEW_v2.md -> 6
- REAL_HAMMER_PLAN.md -> 4
- REWARD_LITERATURE.md -> 13
- REWARD_VALIDATION_METHODOLOGY.md -> 13
- TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md -> 12
- hammering_lit_sweep_RUNBOOK.md -> 1
- hammering_literature_notes.md -> 13
- impact_tracking_rl_litreview.md -> 6
- hammering_reward_design_deep_dive_v2.md -> 15
- tracking_impact_impulse_design_research.md -> 12
- FUTURE_UPDATES.md -> 6
- HANDOVER.md -> 8
- VEGA_TRAINING_PLAN.md -> 9
- thesis_synthesis.md -> 7
- thesis_direction_update.md -> 8
- thesis_handoff_brief_original.md -> 7
- 2026-06-17-z1-single-strike-fix.md -> 1
- 2026-06-17-r_imit-tracking-reward.md -> 0
- 2026-06-17-z1-strike-not-press-redesign-design.md -> 6
- validate_rewards.py -> 41
- derive_impulse_thresholds.py -> 13
- verify_contact_sensor.py -> 18
- verify_reward_setup.py -> 19
- test_single_strike.py -> 21

Notes for the Phase-C link-check (inbound refs that will break on move/merge):
- CAT_DEEP_DIVE.md is referenced by CONSTRAINED_RL_LANDSCAPE.md:21, thesis/README.md:26, FAITHFUL_SOFT_CAT_IMPL_PLAN.md:17/193, results/2026-06-17_velocity_bound_ablation.md:45 — retarget to FAITHFUL_SOFT_CAT_IMPL_PLAN.md on merge.
- JOINT_VELOCITY_BOUND_RESEARCH.md is referenced by results/2026-06-17_b_strike.md:66, results/2026-06-17_velocity_bound_ablation.md:7, __init__.py:28, z1_constants.py:131 — retarget to CONSTRAINED_RL_LANDSCAPE.md on merge.
- REWARD_LITERATURE.md / hammering_literature_notes.md / impact_tracking_rl_litreview.md are merged into the new LITERATURE.md — retarget their inbound refs.
- hammer_z1_env/README.md:7 dangles at docs/BASELINE_AUDIT.md (now docs/archive/) — retarget to docs/README.md index.

---

## Poison-phrase candidates

Deduped from the poison_candidates fields across all rows (regex fragments as given):

- `4\.7% of the worst-joint limit`
- `0\.137 N[·.]s`
- `\[3\.44, 6\.88`
- `~?21× more violent`
- `57\.?3? ?ms`
- `imp_seed[^\n]*p95`
- `excess.p95`
- `9 phases`
- `nine phases`
- `cat_hook\.py`
- `pre-implementation`
- `4\.3.4\.65 rad/s`
- `All 8 validation phases`
- `8 validation phases`
- `6-term reward stack`
- `not real CaT`
- `naive sampled-hard`
- `mjlab 1\.4\.0 may not expose`
- `it doesn't, per`
- `not exposed on the Entity`
- `A3's is a no-op`
- `the crude approximation`
- `does mjlab expose substep`
- `existing 8 phases`
- `8 phases pass unchanged`
- `validate_rewards\.py \(8 \+ Phase I\)`
- `nail_depth_delta 2000\s*→\s*600`
- `all 8 phases pass`
- `6-term baseline`
- `validate_rewards\.py → \*\*all 8 phases`
- `9-term`
- `validate_rewards\.py 9 phases`
- `9-term SPEC`
- `50-iteration CPU smoke`
- `NAIL_SUCCESS_THRESHOLD.{0,8}0\.030\s*→\s*0\.027`
- `validate_rewards\.py \(10 phases\)`
- `155 pytest pass`
- `10 phases`
- `grasp #10`
- `nail_depth_delta × 500`
- `nail_depth_delta \* 500`
- `joint_pos_limits.{0,20}−1\.0`
- `already at −1\.0 in current config`
- `\+0\.010 × 500`
- `× 500 = \+5\.0`
- `0\.005 × 500`
- `8 phases`
- `validate_rewards\.py 10/10 phases`
- `validate_rewards\.py \(9 phases\)`
- `147 unit tests`
- `phases A–I \*\*\+ J/K/L/M`
- `I_ref ≈ 0\.32 N·s`
- `24-DoF hand`
- `welded hammer`
- `position-only DiffIK`
- `generate-then-track`
- `upstream-generated strike trajectory`
- `RL tracking`
- `force-sensor-measured impulse`
- `orientation_weight=0\.0`
- `impact_progress.{0,40}(NEW|PROPOSAL|deferred)`
- `nail_depth_delta=2000`
- `9 phases|8 phases|\(8 phases\)`
- `one payout per contact`
- `Repeated-Peak.{0,30}~?2. rated`
- `windowed axial impulse`
- `clip_actions=1\.0`
- `delta_pos_scale=0\.05`
- `frictionloss=?10`
- `7-term`
- `Phase 0`
- `diagnostic baseline`
- `start fresh on the G1`
- `single-policy, variable-impedance`
- `branch is \`hammer-z1\``
- `7-term reward`
- `0-?7\.5 cm`
- `7 cm`
- `Q13–Q15`
- `weld[- ]pollution`
- `weld pollution blocker`
- `\bReplace\b`
- `0\.137\s*N`
- `~?45%.{0,20}(friction|dof-FRICTION)`
- `not the weld \(~0\.2\)`
- `\b9 phases\b`
- `validate_rewards\D{0,6}9\b`
- `hammer_head_site.{0,30}0\.0326 0\.132`
- `NEAR_NAIL.{0,40}(joint1 -0\.119749|-0\.119749)`
- `weld.{0,20}(dominates|dominate)`
- `how hard the weld drags the arm`
- `weld baseline, not the impact`
- `latch persists for the rest of the episode`
- `do variable impedance fresh on the G1`
- `don't retrofit VIC into the Z1`
- `lacks the variable-impedance action space and the explicit impulse constraint`
- `CAT_DEEP_DIVE\.md`
- `37 CPU unit tests`
- `chain-coupled`
- `3\.1415 rad/s`
- `docs/BASELINE_AUDIT\.md`
- `head\s+box 5.2.2\.5 cm`
- `handle\s+capsule, 22 cm`

---

## Completeness check result

Command:
```
find docs CLAUDE.md README.md thesis_synthesis.md thesis_direction_update.md thesis_handoff_brief_original.md -name "*.md" | grep -v -E "deploy/|doc/|simulate/|\.pytest_cache/" | wc -l
```
Result: **43** in-repo md files found.

In-repo rows in this worklist: **43** (42 from the audit JSON + 1 added by the completeness check).
- MATCH: 43 found == 43 in-repo rows.
- The one file found but absent from the audit JSON — `docs/superpowers/specs/2026-07-05-docs-consolidation-design.md` — was added with verdict `current` and notes "MISSING FROM AUDIT — flag to controller" (it is the second 2026-07-05 consolidation doc, explicitly out of audit scope).

Memory rows: **17** (counted separately). Sibling-repo rows: **3** (counted separately). Rows total: **63**.

---

## Deep-review corrections (2026-07-05) — AUTHORITATIVE; these SUPERSEDE the rows above where they conflict

A 3-reviewer adversarial pass (quote fidelity + evidence reality + replacement correctness + poison-reintroduction) found 9 errors in the rows above and 3 missing fixes. **When a row above conflicts with an item here, apply the item here.**

### C-1 — IMPULSE_CAT_IMPL_PLAN.md:6 replacement reintroduces poison (bare `[3.44, 6.88]`, `0.137 N·s`)
The row-above replacement reprints gripper-era constants into a LIVING doc. Use this **poison-free** replacement instead (old text = the line-6 quote in the row above):
> `> **C0 findings (gripper-era measurement — EE-dependent, superseded after the L6 fixture; re-derive via derive_impulse_thresholds.py):** the off-contact contaminant is dof-FRICTION (efc mjCNSTR_FRICTION_DOF, ~41–48% of raw Λ_j, EE-dependent, spread across the load-bearing arm joints), not the weld (~0.05 gravity-compensated); the object-side delivered ∫F·dt is the clean weld/friction-immune ground truth; per-joint J_limit and the reference-strike ratio are fixture-dependent — do not quote the old gripper-era numbers, re-run derive_impulse_thresholds.py. efc-row isolation of the *specific* contact is not clean in mujoco_warp (deferred). Pinocchio cross-check SKIPPED (not installed — decision pending). max_p>0 NOT enabled (awaits Khadiv mechanism confirm + Pinocchio decision).`

### C-2 — CLAUDE.md:14 replacement reintroduces gripper-era joint attribution (`joints 2/3`)
c0-findings.md:20 (2026-07-04 UPDATE) moved friction to joints 1–3,5. Use this replacement (old = the line-14 quote in the CLAUDE.md row above):
> `note its top decision-to-confirm (soft-CaT vs the docs' CMDP/Lagrangian) and the qfrc_constraint-contamination blocker it resolves (the off-contact contaminant is dof-FRICTION, ~41–48% of raw Λ_j and EE-dependent, spread across the load-bearing arm joints — NOT the weld, whose contribution is tiny; re-derive per-joint figures via derive_impulse_thresholds.py).`

### C-3 — CAT_DEEP_DIVE.md DROP line numbers are wrong (would fail to match)
- DROP "…The env-side termination we ship is the crude approximation." is at **line 9**, not 10.
- DROP "velocity_bound.py:100 returns `torch.rand_like(delta) < delta` — a sampled-Bernoulli HARD episode cut" is at **line 7**, not 5.
Both DROPs remain justified (shipped CatPPO/CatSoftHook). Match on the quoted substring, not the line number.

### C-4 — JOINT_VELOCITY_BOUND_RESEARCH.md:41 is MISATTRIBUTED (do NOT apply to JVB)
The "De-risk the Λ_j measurement… does mjlab expose substep qfrc_constraint?" DROP quote lives in **CAT_DEEP_DIVE.md:41**, not JVB. **JVB has 0 stale DROP claims** — its line 15 ("log per-joint IMPULSE Λ_j = Σ|qfrc_constraint_j|·h") is CURRENT and must be kept. Reassign this DROP to CAT_DEEP_DIVE.md:41 (a 5th CAT_DEEP_DIVE drop).

### C-5 — REWARD_LITERATURE.md:12 DROP over-reaches (terms still ship)
`nail_driven_reward` (Gaussian-on-absolute) and `hammer_approach_reward` still ship (hammer_env_cfg.py:155-178). Do NOT assert they were removed. On merge into LITERATURE.md, drop ONLY the trailing live-config sentence ("The current mjlab `nail_driven_reward` … hovering near high-reward states"); KEEP the Ng et al. potential-shaping takeaway.

### C-6 — hammering_literature_notes.md:181 DROP evidence is wrong (not code-falsified)
No damage/strike-quality penalty ships; impact_progress is an impact reward, not a damage penalty. The valid reason to drop is **non-actionability**: orientation/off-axis is moot on the position-only DiffIK action space (orientation_weight=0.0, hammer_env_cfg.py:121; deep_dive_v2 §11). Correct the evidence wording accordingly (it's a non-actionable drop, not a falsified-by-code drop).

### C-7 — impact_tracking_rl_litreview.md:51 quote is a paraphrase (not verbatim)
Line 51 actually reads "…and (c) a **force-sensor-measured impulse** feeding both the impact reward and the CRB-based recoil prediction …". DROP is justified (superseded by MuJoCo-native qfrc_constraint accumulator). Use the verbatim `force-sensor-measured impulse` clause as the drop anchor, not "The intersection that is the project's novelty".

### C-8 — VEGA_TRAINING_PLAN.md:31 is MIS-SCOPED (do NOT apply to VEGA)
"The active branch is `hammer-z1`." is not in VEGA — it lives in **HANDOVER.md:31** (which is being archived, so the banner handles it). **VEGA has only ONE fix: line 5** (the Q13–Q15 → Q2/Q3/Q6/Q7/Q9/Q11/Q12 fix). Drop the VEGA:31 row.

### C-9 (NEW) — FAITHFUL_SOFT_CAT_IMPL_PLAN.md:304 also has stale `cat_hook.py` (audit missed it)
Line 304: `- **C3 — env hook + extras plumbing (sim, CPU).** Implement `cat_hook.py`; wire `cat_soft`. Test:` → replace `Implement \`cat_hook.py\`` with `Implement \`cat/hook.py\` (\`CatSoftHook\`)`. (Without this, the `cat_hook.py` poison phrase survives.)

### C-10 (NEW) — IMPULSE_CAT_IMPL_PLAN.md:87 quotes the stale `not exposed on the Entity` phrase
Rewrite line 87 so it does not reprint the exact stale phrase. Old = the line-87 sentence; New:
> `> `qfrc_constraint` **does** exist on the mujoco_warp 3.8.1 `Data` and is sliceable via the existing `entity._joint_dof_field('qfrc_constraint')` helper (`data.py:237-240`) — the `velocity_bound.py:22` comment claiming it was inaccessible is **outdated** and already corrected in C0. Prefer adding a clean `qfrc_constraint` `@property` to `data.py` (mirror `qfrc_actuator` at `data.py:414-423`). `torque_source ∈ {'constraint','actuator'}` param kept for the documented ablation.`

### C-11 (NEW) — OPEN_QUESTIONS.md 185–196 A–H table is triply-stale (`6-term`, `8 phases`, `× 500`, `0.070`)
Replace the block from line 185 through line 198 (the "Independent of all open questions…" intro + the A–H table + the "Methodology:" line) with the pointer paragraph below. This removes `6-term reward stack`, `All 8 validation phases`, every `× 500`, and the stale 0.070/0.071 completion threshold in one edit. Keep the `## Reward Stack Validation (2026-05-22)` header (dated section) and the two paragraphs after line 198 (Implication / Note on baseline composition).
> `Independent of all open questions above, the current 7-term reward stack (\`approach\`, \`nail_driven\`, \`nail_depth_delta\`, \`impact_progress\`, \`completion\`, \`action_rate\`, \`joint_pos_limits\`) is validated end-to-end by \`validate_rewards.py\`, which now runs phases A–M (impact_progress = Phase I, impulse-CaT = Phase M). See that script for the authoritative per-phase expected values — they track the live weights in \`hammer_env_cfg.py\` (e.g. \`nail_depth_delta = 600\`), so this doc no longer restates a per-phase table (the earlier 6-term / 8-phase / ×500 walkthrough was a 2026-05-22 snapshot and drifted). Run it before any GPU training.`

### C-12 — CAT_DEEP_DIVE.md link retargets (Task 4/5 link-check must hit ALL of these)
The `CAT_DEEP_DIVE.md` reference survives at **CONSTRAINED_RL_LANDSCAPE.md:21, FAITHFUL_SOFT_CAT_IMPL_PLAN.md:17 and :193, docs/thesis/README.md:26** (+ results/2026-06-17_velocity_bound_ablation.md:45). All must be retargeted to `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` on merge or the `CAT_DEEP_DIVE\.md` guard phrase stays RED.

### C-13 (CRITICAL) — verbatim provenance tags must OMIT the `.md` extension
The merge policy requires each assembled block carry a `[from <source>]` tag, and the merge
appendix headers (Task 4) name their source. If those tags print the full filename **with
`.md`** (e.g. `[from CAT_DEEP_DIVE.md]`, `## Appendix … [from CAT_DEEP_DIVE.md, verbatim]`),
they will (a) trip the `CAT_DEEP_DIVE\.md` freshness-guard phrase forever, and (b) trip Task 5
Step 3 / Task 8 link-checks — because those tags live in LIVING docs (LITERATURE.md, the
FAITHFUL/CONSTRAINED appendices). **Resolution:** every verbatim provenance tag and appendix
header uses the **bare doc name, no extension** — `[from CAT_DEEP_DIVE]`, `[from
REWARD_LITERATURE]`, `[from hammering_literature_notes]`, `[from impact_tracking_rl_litreview]`,
`[from JOINT_VELOCITY_BOUND_RESEARCH]`. Real file *pointers* ("see `CAT_DEEP_DIVE.md`") keep
`.md` and are the only thing the `.md`-anchored guard/link-checks target.
- **Task 5 Step 3 grep must use the `.md` extension** (`CAT_DEEP_DIVE\.md`, not bare
  `CAT_DEEP_DIVE`) so it flags un-retargeted file pointers, not the bare-name provenance tags.
- This is why the guard phrase is `CAT_DEEP_DIVE\.md` (with extension): it stays RED until the
  four surviving `.md` pointers (C-12) are retargeted, and goes GREEN even though the appendix
  keeps a `[from CAT_DEEP_DIVE]` bare-name tag.

### Freshness-guard note
`tests/test_docs_current.py` `POISON_PHRASES` is the curated, self-consistent list (RED now,
GREEN after Tasks 3–7): all 28 phrases were grep-confirmed to hit a living doc and traced to
their removal step. The plan's original 9-phrase base list was mostly invalid: `57.3 ms`, `one
switch`, `gripper holds the hammer`, `imp_seed…p95` live only in **memory / archive / .py**,
not the md living set, and `**9 phases**` never matches (the real text is unbolded `9 phases`).
Those are dropped; the curated set is in the test file, each phrase tagged with its removal task.

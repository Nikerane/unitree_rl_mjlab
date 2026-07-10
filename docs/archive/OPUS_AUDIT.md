> ⚠️ **ARCHIVED 2026-07-05** — superseded by nothing — historical audit record (2026-05-22).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Opus 4.7 Audit of Sonnet 4.6 Reward-Engineering Deliverables

**Date:** 2026-05-22
**Auditor model:** Opus 4.7
**Subject:** Sonnet 4.6 output in this session — literature review, reward design matrix, recommended spec, and implementation.

**Audit scope:** Critical second-pass. Not a redo. Honest engineering critique with severity tags. Where I cannot independently verify a claim (e.g. a venue acceptance), I say so rather than fabricate confirmation. Disagreements with Sonnet's research are flagged in this document, not silently rewritten in the source files.

**Verification runs:**
- `validate_rewards.py` → **all 8 phases pass**. Implementation (6-term baseline) is internally consistent.
- `verify_contact_sensor.py` → previously confirmed `hammer_head` geom resolves. ContactSensor is wired but not consumed by any reward term yet.

The implementation is the source of truth; documentation drift is the dominant issue.

---

## Findings

### 🔴 CRITICAL

#### C1 — Spec pseudocode would raise `TypeError` if copied as-is
- **Where:** `RECOMMENDED_REWARD_SPEC.md:54` (NailDepthDeltaTerm pseudocode) and `RECOMMENDED_REWARD_SPEC.md:187` (ImpactVelocityBonusTerm pseudocode)
- **What:** Both show `super().__init__(cfg, env)` inside `__init__(self, cfg, env)`.
- **Why it's wrong:** mjlab's base class `ManagerTermBase.__init__(self, env: ManagerBasedRlEnv)` (verified at `mjlab/managers/manager_base.py:65`) takes **only `env`**. The auto-instantiation passes `(cfg, env)` to the subclass, but the subclass must pass only `env` upward.
- **Why the implementation works anyway:** the actual code at `src/tasks/hammer/mdp/rewards.py:95` correctly calls `super().__init__(env)`. The spec is what's wrong, not the code.
- **Impact:** anyone implementing `ImpactVelocityBonusTerm` (currently deferred) by copying the spec pseudocode will hit a runtime error on env construction.
- **Patch action:** Fixed in `RECOMMENDED_REWARD_SPEC.md` with `<!-- opus-audit -->` markers.

---

### 🟠 HIGH

#### H1 — Spec §1 contradicts what is actually implemented
- **Where:** `RECOMMENDED_REWARD_SPEC.md:8–16` (Section 1 — "Keep / Drop / Replace") vs. `src/tasks/hammer/hammer_env_cfg.py:129–179`.
- **What the spec says:** "Replace `nail_driven_reward` with `nail_depth_delta`. Replace `hammer_approach_reward` with contact-gated version. Increase `action_rate` to −0.02."
- **What the code does:** **Augments** rather than replaces. Both Gaussian terms are still active at original weights. `action_rate` is still −0.01. The user explicitly chose this minimum-viable-change approach earlier in the session (saved as `feedback_minimum_viable_reward.md`).
- **Why it matters:** the spec's §1 reads as a directive ("Replace ..."), but the actual baseline is correctly preserving the old terms while adding progress + completion. Anyone reading the spec thinks the implementation is incomplete; anyone reading the code thinks the spec wasn't followed. Neither is true — the policy diverged intentionally.
- **Patch action:** Rewrote Section 1 in `RECOMMENDED_REWARD_SPEC.md` to reflect the augment-not-replace strategy, with the original "Replace" recommendation moved to "potential future change" status. `<!-- opus-audit -->` markers.

#### H2 — MATRIX vs SPEC contradict each other on completion-bonus guarding
- **Where:** `REWARD_DESIGN_MATRIX.md:43` vs. `RECOMMENDED_REWARD_SPEC.md:297–305`.
- **What MATRIX says:** "fires every step after success unless guarded. Guard: `(nail_depth >= thresh) & (prev_depth < thresh)` — requires stateful tracking, or just accept repeated bonus (mjlab terminates on success anyway)."
- **What SPEC says:** "mjlab terminates the episode on success, so this fires at most once. Shape: (B,)."
- **Which is correct:** SPEC is correct. `mjlab/managers/reward_manager.py:compute` runs BEFORE the termination check; the bonus fires exactly once at the success step, then `nail_fully_driven` terminates the episode, so the bonus cannot fire again on subsequent steps. No guard needed.
- **Validation evidence:** Phase H of `validate_rewards.py` confirms exactly-once firing with no guard.
- **Patch action:** Rewrote the MATRIX entry to remove the misleading guard text. `<!-- opus-audit -->` marker.

#### H3 — Meta-World citation includes fabricated authors *(RESOLVED 2026-05-22)*
- **Where:** `REWARD_LITERATURE.md:116`
- **What Sonnet wrote:** `Yu, T., Quillen, D., He, Z., Julian, R., Narayan, A., Shao, H., Nair, A., Chen, S., Bahl, S., Planche, B., & Levine, S.`
- **What arXiv:1910.10897 actually says (verified 2026-05-22):** `Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, Sergey Levine`
- **Fabricated names:** `Shao, H.`, `Nair, A.`, `Chen, S.`, `Bahl, S.`, `Planche, B.` (none are Meta-World authors)
- **Missing names that Sonnet dropped:** `Shively, H.`, `Bellathur, A.`, `Hausman, K.`, `Finn, C.` (Hausman + Finn are senior authors — significant omission)
- **Severity rationale:** Five fabricated and four missing authors in a 10-author paper. Citation was almost entirely wrong despite being presented confidently.
- **Patch action:** Replaced the citation with the verified list. The `<!-- opus-audit H3 -->` marker now records the fabricated/dropped names for traceability.

---

### 🟡 MEDIUM

#### M1 — Three preprint citations are presented as if peer-reviewed venues
- **Where:**
  - `REWARD_LITERATURE.md:140` — `Wang et al. 2025 (RTW), arXiv:2503.15724` (March 2025; venue not verified)
  - `REWARD_LITERATURE.md:106` — `Ma et al. 2024 (DrEureka), arXiv:2406.01967` (June 2024; venue not verified)
  - `REWARD_LITERATURE.md:70` — `Kim et al. 2025 (ARMADA), RSS 2025` — RSS 2025 has occurred by today's date (2026-05-22), but I cannot re-verify the venue claim inside this audit session
- **What's wrong:** These are cited as if accepted at named venues without verification. The earlier Sonnet phase used a web-search agent to verify Meta-World, IndustReal, Factory, DeepMind TT, RSL-RL, DM Soccer, and Wu et al. (DREM) — but did NOT call out the other three as preprint-only.
- **Why it matters:** propagates the fabricated-citation risk pattern of H3.
- **Patch action:** Added "(preprint — venue not verified)" markers to the three entries.

#### M2 — Standard impact-RL terms are missing from MATRIX and §4
- **Where:** `REWARD_DESIGN_MATRIX.md` and `RECOMMENDED_REWARD_SPEC.md:347–428`
- **Missing:**
  - **Time penalty** (`-0.01`/step) — pushes the policy toward earlier completion. Standard in episodic RL when termination on success has insufficient discount-factor pressure.
  - **Energy / torque² penalty** — `sum(τ²)` on arm joints. Standard in RSL-RL locomotion configs and sim-to-real manipulation. The SPEC mentions `torque_peak_penalty` (a hinge penalty above a limit), which is different and complementary.
  - **Orientation alignment gate** — discussed in `REWARD_LITERATURE.md:118` (Meta-World) as the multiplicative `quat_alignment` factor, but it never appears in MATRIX or SPEC §4 as a recommended term. If the policy approaches the nail from a wrong angle, the hammer face won't strike the nail head — this is a real failure mode for hammering.
- **Why it matters:** the spec presents itself as comprehensive but is missing terms a competent RL practitioner would consider standard.
- **Patch action:** Added a "Considered but deferred" subsection in SPEC §1 listing these three terms, with the trigger conditions for adding them.

#### M3 — `approach_contact_gated` may flicker under brief contacts
- **Where:** `RECOMMENDED_REWARD_SPEC.md:101–124` and the underlying contact-flag mechanism in `mjlab.sensor.ContactSensor`.
- **What:** The gate uses `(sensor.data.found > 0).any(dim=-1).float()`. With `decimation=10` substeps per control step, the `.found` flag at reward-compute time reflects the contact state at the **last** substep — but during a fast bounce (the hammer strikes and rebounds within the 10 substeps), the flag could be 0 by the time the reward is computed even though contact occurred. The approach reward would NOT be gated off for that step, despite a strike having happened.
- **Why it matters:** Could produce reward inconsistency where the policy is rewarded for both approach AND a successful strike on the same step. Not catastrophic (it's a small temporary bias), but worth knowing.
- **Mitigation option:** `ContactSensorCfg.history_length` (set to `decimation`) would buffer all substep contacts; the gate could check `(history > 0).any()` instead of just `.found`.
- **Patch action:** Documented in OPEN_QUESTIONS.md as Q12. Not patched in spec because the mitigation needs validation.

#### M4 — Weight-scale arithmetic in §8 doesn't balance
- **Where:** `RECOMMENDED_REWARD_SPEC.md:494–508`
- **What:** Per-term cumulative estimates and the "Total max per episode ≈ +218" don't reconcile:
  - `action_rate −0.02 × ~0.05 max/step × ~1000 steps = ~−50`, but row says `~−10`
  - `joint_vel −0.005 × ~6 (sum of 6 joints × 1 rad/s) × ~1000 = ~−30`, but row says `~−60`
  - The arithmetic at the bottom: `50 + 100 + 37.5 + 100 − 70 ≈ 218` — but actual term sum is `+approach + impact + air + depth_delta + completion − penalties` and the numbers don't add to 218 either
- **Why it matters:** Anyone using these as a sanity-check target during training will get confused. The estimates are reasonable order-of-magnitude but the table presents them with false precision.
- **Patch action:** Marked the §8 table as "approximate; verify by logging per-term episode sums" with `<!-- opus-audit -->`.

#### M5 — Pre-Training Verification §6 references an obsolete script
- **Where:** `RECOMMENDED_REWARD_SPEC.md:451`
- **What:** Points users at `verify_reward_setup.py`. That script does still exist, but `verify_contact_sensor.py` and `validate_rewards.py` are the actually-used verification tools (per this session's work). Running `verify_reward_setup.py` alone gives less coverage than the newer pair.
- **Patch action:** Updated §6 to recommend the newer pair, with `verify_reward_setup.py` kept as a comprehensive option.

---

### 🟢 LOW

#### L1 — IsaacGym mention in Eureka discussion
- **Where:** `REWARD_LITERATURE.md:100` mentions Eureka uses IsaacGym. Accurate description of Eureka, but readers may be confused since the current project uses mjlab/mujoco_warp.
- **Patch action:** None needed — the description is honest about Eureka's setup.

#### L2 — Legacy `env.py:200` comment differs from spec
- **Where:** `~/repos/safe_impact_manipulation/hammer_z1_env/env.py:200`
- **What:** Comment says impact_bonus "fires exactly once per episode" — accurate for legacy mocap-based design, but the spec recommends a `ContactSensor`-based version that fires on every re-strike. Anyone reading both will think one of them is wrong.
- **Patch action:** None — the legacy env is a reference implementation, not the deployment target. The SPEC and `OPEN_QUESTIONS.md` already document the upgrade.

#### L3 — Validation methodology and audit scope overlap
- **Where:** `REWARD_VALIDATION_METHODOLOGY.md` covers reward-term unit testing but does not mention "ablation order" or "training observability checks" — those are in the SPEC §5 and §6.
- **Patch action:** None — the separation is reasonable, but a cross-reference would help future readers.

---

## Summary table

| ID | Severity | Subject | Patched in source? |
|---|---|---|---|
| C1 | CRITICAL | Pseudocode `super().__init__(cfg, env)` bug | ✅ Yes |
| H1 | HIGH | Spec §1 contradicts implemented baseline | ✅ Yes |
| H2 | HIGH | MATRIX vs SPEC completion-bonus contradiction | ✅ Yes |
| H3 | HIGH | Fabricated Meta-World authors (5 wrong, 4 missing) | ✅ Verified against arXiv and corrected |
| M1 | MEDIUM | Preprint citations presented as venue-published | ✅ Marked preprint |
| M2 | MEDIUM | Missing standard terms (time, energy, quat-gate) | ✅ Added to deferred bucket |
| M3 | MEDIUM | Approach-gate flicker under brief contacts | ✅ Logged as Q12 |
| M4 | MEDIUM | Weight-scale §8 arithmetic | ✅ Marked approximate |
| M5 | MEDIUM | Verify-script reference outdated | ✅ Updated |
| L1 | LOW | IsaacGym mention | No action |
| L2 | LOW | Legacy comment vs spec | No action |
| L3 | LOW | Doc cross-reference | No action |

---

## What I did NOT do

- Did not redo the literature search. Sonnet's papers and "what to steal" assessments were left intact except where citation integrity demanded a flag.
- ~~Did not silently rewrite H3 (Meta-World author list)~~ — corrected after user request. Verified author list against arXiv:1910.10897 directly; 5 fabricated names removed, 4 dropped names restored.
- Did not change the implemented baseline in `rewards.py` or `hammer_env_cfg.py`. The implementation passes validation; the spec was the thing out of sync.
- Did not run training or test_single_strike.py — both would have low audit value (training needs GPU; single-strike was already deferred to "run when needed" per `OPEN_QUESTIONS.md`).

## Overall judgment

Sonnet's deliverable is **solid engineering work with documentation drift**. The implementation matches the user's minimum-viable strategy and passes validation. The literature review is mostly accurate (one fabricated author, three preprints presented as venue-published). The spec has one CRITICAL pseudocode bug and several places where directives don't match what was actually built. None of these are fatal — most are documentation-only fixes — and the underlying engineering choices are defensible.

Recommended posture going forward: **trust the code, not the spec**. Use `validate_rewards.py` as the source of truth for what the reward stack actually does. When adding a deferred term, read the implementation in `rewards.py` as the canonical pattern, not the pseudocode in §3 of the SPEC.

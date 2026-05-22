# Open Questions — Z1 Hammer Reward Design
## Things that need sim experiments, not more literature

Status legend: 🔴 OPEN | 🟡 SCRIPT PROVIDED (run to resolve) | 🟢 RESOLVED

---

### 🟡 Q1 — Does a single strike fully drive the nail?

**Question:** With the current nail physics (frictionloss=0.3 N, damping=8 N·s/m, range 0–0.075 m), what is the maximum nail depth achievable in a single strike given the Z1's maximum achievable end-effector velocity via DifferentialIK?

**Why it matters:** If a single hard strike can drive the nail 75mm, the repeated-strike reward design is unnecessary — the policy just needs to learn one good swing. If max single-strike depth is ~10–20mm, repeated striking is mandatory and the `air_time_bonus` + `impact_velocity_bonus` architecture becomes critical.

**Resolution:** Run `test_single_strike.py`. Script lifts the hammer to several approach heights, then commands a max-velocity downward strike, and reports the resulting nail depth per height. Interpretation:
- If any approach yields ≥ 70 mm depth: single-strike is achievable; repeated-strike rewards are optional.
- If all approaches yield < 30 mm: repeated striking is mandatory.

```bash
python docs/research/reward-design/test_single_strike.py
```

---

### 🔴 Q2 — Does the policy discover retract-and-restrike naturally?

**Question:** With only `nail_depth_delta` + `impact_velocity_bonus` (no `air_time_bonus`), does the policy learn to retract and re-strike, or does it get stuck pressing continuously?

**Why it matters:** If retraction emerges naturally, `air_time_bonus` is unnecessary complexity. If not, it's essential. The literature provides no answer — this is an open niche (Phase 2 agent confirmed no prior work on repeated hammering RL).

**Experiment:** Train ablation step 3 (approach + depth_delta + impact_velocity, no air_time). Record video of 100 rollouts. Count: (a) episodes with ≥2 distinct contact events, (b) episodes where the nail advances > 5mm.

**Cannot be resolved without running training.** Plan: keep `air_time_bonus` in the initial config (Step 4 of the ablation), and run a single ablation that disables it to confirm whether it's load-bearing.

---

### 🔴 Q3 — Optimal `impact_velocity_bonus` weight

**Question:** What weight on `impact_velocity_bonus` causes the policy to prefer swinging (high velocity at contact) over pressing (low velocity but continuous)? What weight causes the policy to flail wildly (too high)?

**Why it matters:** Weight=10 is an engineering estimate. Too low: no effect on policy behaviour. Too high: policy learns to maximize velocity at contact regardless of nail alignment (random flailing scores high).

**Experiment:** Grid search over weights [1, 5, 10, 20, 50]. For each, log: (a) mean impact speed at contact, (b) mean nail depth at episode end, (c) success rate.

**Cannot be resolved without running training.** Defer to first training campaign.

---

### 🟢 Q4 — Contact sensor geom pattern *(RESOLVED)*

**Question:** What is the exact geom name(s) of the hammer head in the Z1 MuJoCo XML? Does `"hammer_head.*"` resolve to the correct geoms?

**Resolution:** Inspected `hammer_z1_env/assets/z1_mocap_hammer.xml`. The hammer geoms are:
- `hammer_head` — the striking surface (use as ContactSensor primary)
- `hammer_handle` — handle, NOT for impact detection

Use **exact match** `pattern="hammer_head"` (NOT a regex with wildcards — there's only one matching geom and the wildcard adds nothing). The spec has been updated to reflect this. Also: `verify_reward_setup.py` checks `sensor.primary_names` at startup and fails loudly if the pattern resolves to zero geoms.

---

### 🟡 Q5 — `air_time` min/max thresholds

**Question:** What is the typical distribution of hammer head retraction duration in rollouts from a random policy vs. a partially trained policy? What `min_air_time` threshold separates meaningful retraction from random jitter?

**Resolution:** `verify_reward_setup.py` now collects 200 random-policy steps and prints percentiles of `last_air_time` and impact speed at every contact event. Recommended thresholds:
- `min_air_time` ≈ 50th percentile (median) of random-policy retraction times — bonus fires for above-average retraction.
- `max_air_time` ≈ 95th percentile — caps the bonus and prevents the policy from being rewarded for slow hovering.

Run the script and read the recommendation. Re-run after initial training with a partially trained policy to confirm thresholds remain sensible.

---

### 🔴 Q6 — Approach std tuning

**Question:** What `std` for `approach_contact_gated` gives a useful learning signal from the episode start without causing the policy to commit to a single approach direction?

**Current:** std=0.15 (very wide, covers ~15cm radius — basically always near 1.0 when the arm is anywhere near the nail). **Proposed:** std=0.10. **Risk:** too narrow → sparse approach reward → slow learning.

**Why it matters:** The approach reward is the only positive signal in the first few hundred episodes (before contact is made). Too narrow: policy can't find the nail. Too wide: approach reward doesn't discriminate good from bad approach paths.

**Experiment:** Train two runs: std=0.10 vs std=0.15. Compare time-to-first-contact (median step at which `compute_first_contact()` first fires per episode).

**Cannot be resolved without running training.** Defer to first ablation.

---

### 🔴 Q7 — Nail physics calibration for sim-to-real

**Question:** What frictionloss and damping values for the nail slide joint produce impact dynamics (post-impact velocity) that match a real nail being struck?

**Why it matters:** van Steen et al. (2024) showed MuJoCo's post-impact velocity error is ~3.1% with calibrated parameters. Uncalibrated: larger error, trained policy may drive the nail with wrong force profile on real hardware.

**Experiment:** If a real nail-driving setup exists, measure: (a) input hammer velocity before contact, (b) nail displacement after single strike. Tune `frictionloss` and `damping` in the XML until simulated displacement matches real-world measurement.

**Requires real hardware data — cannot resolve from simulation alone.** Defer until sim-to-real transfer is attempted.

---

### 🟢 Q8 — Stateful `NailDepthDeltaTerm` reset timing *(RESOLVED)*

**Question:** Does mjlab call `ManagerTermBase.reset(env_ids)` before or after the first observation is taken on episode reset?

**Resolution:** Read `mjlab/managers/reward_manager.py:100-114`. `RewardManager.reset(env_ids)` first zeroes `_episode_sums`, then calls `term_cfg.func.reset(env_ids=env_ids)` for all class-based terms. The next call to `RewardManager.compute()` runs against the already-reset state. So:

- `_max_depth[env_ids]` is set to 0 in `NailDepthDeltaTerm.reset()`.
- On the next step's `compute()`, `current_depth ≈ 0` (the env was just reset) and `_max_depth = 0` → `delta = 0`.

No spurious reward on the first step. **No debug assertion needed.**

---

### 🔴 Q9 — Repeated impact vs. single deep strike: which is harder to learn?

**Question:** Is training faster when the task can be solved by a single very hard strike (if physically achievable) or by many moderate strikes? Should we add domain randomisation on nail friction to force the policy to develop a robust multi-strike strategy?

**Why it matters:** If the policy discovers a single-strike solution early and it works, it may never explore repeated striking. Adding randomisation on frictionloss (range 0.1–0.5 N) forces the policy to develop a robust repeated-strike strategy that generalises to real-world nail variation.

**Experiment:** Train with and without friction randomisation. Compare: distribution of number of contact events per successful episode, sim-to-real gap on robot hardware.

**Cannot be resolved without running training + sim-to-real transfer.** Depends on Q1 resolution: if single-strike is impossible, this question is moot (the policy is forced into multi-strike). Otherwise, defer to sim-to-real validation phase.

---

### 🟢 Q10 — `site_vel_w` availability *(RESOLVED)*

**Question:** Is `robot.data.site_vel_w` populated during reward computation?

**Resolution:** Eliminated by design. `ImpactVelocityBonusTerm` is now a stateful `ManagerTermBase` that finite-differences `site_pos_w` instead of reading `site_vel_w`. Cost: one extra tensor copy per step. Benefit: independent of mjlab's lazy-evaluation behaviour. The spec has been updated. Original stateless variant retained as a footnote for users who explicitly want it after verification.

---

## Resolution Summary

| Q | Status | How resolved |
|---|---|---|
| Q1 | 🟡 Script provided | Run `test_single_strike.py` |
| Q2 | 🔴 Needs training | Ablation Step 3 vs Step 4 comparison |
| Q3 | 🔴 Needs training | Weight grid search |
| Q4 | 🟢 Resolved | Geom name = `hammer_head` (exact, verified in XML) |
| Q5 | 🟡 Script provided | `verify_reward_setup.py` outputs percentiles |
| Q6 | 🔴 Needs training | Two-run ablation |
| Q7 | 🔴 Needs hardware | Defer until sim-to-real |
| Q8 | 🟢 Resolved | mjlab reset ordering is correct; no action needed |
| Q9 | 🔴 Needs training + hardware | Defer; depends on Q1 |
| Q10 | 🟢 Resolved | Finite-differenced velocity eliminates the risk |

**Remaining blockers before first training run:** none. All resolvable-from-code questions are answered. Remaining open questions all require either running the env (Q1, Q5 — scripts ready) or training cycles (Q2, Q3, Q6, Q7, Q9 — defer to training campaign).

---

## Reward Stack Validation (2026-05-22)

Independent of all open questions above, the current 6-term reward stack (`approach`, `nail_driven`, `nail_depth_delta`, `completion`, `action_rate`, `joint_pos_limits`) has been validated end-to-end via `validate_rewards.py`. All 8 validation phases pass:

| Phase | What it tests | Result |
|---|---|---|
| A. Reset & hold | baseline / no spurious signals | `nail_depth_delta = 0` |
| B. Move action | no delta without contact | `nail_depth_delta = 0` |
| C. Force depth 0.010 m | progress fires correctly | `nail_depth_delta = 5.0` (= 0.010 × 500) |
| D. Hold at 0.010 m | no re-fire on same depth | `nail_depth_delta = 0` |
| E. Force depth 0.020 m | NEW delta fires, not cumulative | `nail_depth_delta = 5.0` (= 0.010 × 500) |
| F. Reset + depth 0.005 m | `NailDepthDeltaTerm.reset()` works | `nail_depth_delta = 2.5` (= 0.005 × 500) |
| G. Bounce-back 0.020 → 0.015 m | `clamp_min(0)` prevents negative | `nail_depth_delta = 0` |
| H. Completion bonus | sparse task reward at success threshold | `completion = 0` below 0.070, `= 100.0` at 0.071 |

Methodology: `REWARD_VALIDATION_METHODOLOGY.md`.

**Implication:** when GPU training becomes available, the reward stack is known correct. Any failure during training is attributable to policy learnability or env dynamics, not to a buggy reward term.

**Note on baseline composition:** `completion` (sparse task reward) was added to the baseline alongside the dense shaping terms, because completion is the actual task signal, not a shaping choice. Other terms from `RECOMMENDED_REWARD_SPEC.md` (`impact_velocity_bonus`, `air_time_bonus`, `approach_contact_gated`, `joint_vel_penalty`) remain in the deferred bucket — they are shaping terms to be added only when a specific failure mode is observed during training.

---

## Opus 4.7 Audit (2026-05-22) — New experiment questions

<!-- opus-audit: added during second-pass audit. See OPUS_AUDIT.md for the full
     finding table and severity assignments. -->

### 🔴 Q11 — Value-function discontinuity from large completion bonus

**Question:** With `completion_bonus = +100` arriving on the same step as the `nail_fully_driven` termination, does PPO's value function develop a discontinuity that destabilises training?

**Why it matters:** The dense shaping reward at the step before completion is small (~0.05–5.0 per step from depth_delta + Gaussians). Then the completion step contributes +100 at once, and the episode ends — so the value of the pre-completion state suddenly jumps by ~100/(1−γ) ≈ huge with γ near 1. This is exactly the cliff PPO can struggle with.

**Experiment:** Train two runs at completion weights `+100` and `+50`, holding everything else fixed. Compare:
- value-function loss curve (sudden spikes after first successes?)
- KL divergence per update (does the policy "swing wildly" around the first successes?)
- final success rate

If the +100 run shows instability around the first successes that the +50 run does not, **reduce completion weight**. The alternative is to spread the bonus across the last N steps before completion, but that adds complexity.

**Cannot resolve without GPU training.** Defer to first real training run.

### 🟡 Q12 — Approach reward jitter under brief contacts

**Question:** With `decimation=10` substeps per control step, can the hammer briefly contact the nail and bounce off within a single control step — leaving `sensor.data.found == 0` at reward-compute time despite the contact having occurred?

**Why it matters:** If `approach_contact_gated` is added (it's deferred but planned), the gate condition `(sensor.data.found > 0).any(dim=-1)` reads only the final substep's contact state. A bounce inside the substep window would leave the gate "off" — the policy would receive BOTH the approach reward AND any impact-event rewards on the same step. Mildly inconsistent.

**Experiment options:**
1. Set `ContactSensorCfg.history_length = 10` (one full control step) and replace the gate with `(sensor.data.force_history.norm(dim=-1) > 0).any(dim=(1, 2))` — captures all substeps.
2. Empirically: run a scripted fast-strike sequence, log the per-control-step contact state and the actual substep contact counts. If the fraction of mismatched steps is < 1%, ignore the issue.

**Cannot resolve without running the env with `approach_contact_gated` enabled.** Defer until that term is added.

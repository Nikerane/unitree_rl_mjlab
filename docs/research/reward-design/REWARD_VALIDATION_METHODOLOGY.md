# Reward Validation Methodology
## Testing reward correctness without training

---

## The principle

A reward function is a **deterministic function**: `(state, action, next_state) → scalar`. You can test it like any other deterministic function — feed it known inputs, assert specific outputs. Training is only required to evaluate **policy learnability**; it is not required to evaluate **reward correctness**.

This is the same logic as unit-testing any pure function. The fact that the reward is consumed by an RL algorithm doesn't make the reward itself untestable — the reward computation runs every step regardless of whether anyone is training on it.

---

## What this catches

Validation runs a scripted state sequence (or directly manipulates the env state) and asserts each reward term produces the expected value at each step.

| Bug | How validation catches it |
|---|---|
| Stateful term doesn't reset between episodes | Episode 2 step 1 returns nonzero reward from leaked state |
| Reward weight is wrong order of magnitude | Cumulative episode reward is 10× too big/small |
| `clamp_min(0)` missing on a delta reward | Negative deltas appear when the tracked quantity bounces back |
| Term reads wrong joint/site index | Reward never fires even with the physical event clearly occurring |
| Tensor shape mismatch on GPU vs. CPU | Crash or silent broadcast bug visible immediately |
| Term forgets to update its tracked state | Same delta fires every step instead of just once |
| Ordering of `clamp` and update is wrong | Reward fires once but never again for further progress |

These are exactly the bugs that produce a 5-hour training run with mysteriously flat reward curves.

---

## What this does NOT catch

- **Policy learnability** — whether PPO can extract a useful gradient from the reward signal. That's emergent behavior, requires training.
- **Local optima** — whether the reward landscape has dead ends that trap the policy. Training-specific.
- **Reward hacking** — whether a smart policy can find an unintended way to maximize the reward. Often only visible after training when you watch rollouts.
- **Sim-to-real gap** — whether reward gradients that work in sim transfer to hardware.
- **Inter-term interference** — emergent behavior from combining multiple shaping terms. Some interference is structural (catchable here), some is policy-mediated (not catchable here).

**Validation is necessary but not sufficient.** It catches the obvious bugs cheaply. It does not prove the reward is well-designed.

---

## Related work

The closest paper-level analog is **Eureka** (Ma et al., 2023, arXiv:2310.12931). Eureka uses GPT-4 to generate reward code, then evaluates each candidate by running it through IsaacGym against a fixed-quality policy and measuring a "fitness" score. The validation step in their pipeline is essentially what's described here: deterministic execution of the reward function against known trajectories.

The general practice has no single canonical citation — it's standard software engineering applied to RL components. Most production RL codebases include some form of reward unit testing; few publish the practice.

---

## Validation phases for the hammer task

For each scripted phase, run N steps and assert per-term reward behavior:

### Phase A — Reset and hold (zero action)
- Action: zeros
- Expected nail_depth_delta: **0 every step** (nail isn't moving)
- Expected nail_driven (Gaussian): small but constant (depends on initial depth, which is 0 → exp(-6.25) ≈ 0.002)
- Expected approach: constant (depends on initial hammer→nail geometry)
- Expected action_rate: 0 (no change between zero actions)

**Catches:** stale state from previous episode (delta fires on first step), unwanted spurious signals.

### Phase B — Move (nonzero action, no nail contact)
- Action: any consistent direction
- Expected nail_depth_delta: **still 0** (no contact, no nail movement)
- Expected approach: varies as hammer moves
- Expected action_rate: nonzero on first step (action changed from zeros), then small/zero if action stays constant

**Catches:** depth_delta firing without nail movement (would mean reading wrong joint).

### Phase C — Force nail to advance (write nail_slide joint position directly)
- Manipulate: `nail_block.data.joint_pos[:, nail_slide_idx] = 0.010` (10 mm)
- Expected nail_depth_delta: **+0.010 × 500 = +5.0** on the step that change is observed
- Expected nail_depth_delta: **0** on the next step (already at max_depth)
- Expected nail_driven (Gaussian): rises from ~0 toward ~exp(-(0.065)²/0.03²) ≈ exp(-4.7) ≈ 0.009

**Catches:** weight magnitude bugs, missing `_max_depth` update, missing clamp.

### Phase D — Force nail to advance again
- Manipulate: depth from 0.010 → 0.020
- Expected nail_depth_delta: **+0.010 × 500 = +5.0** (the new delta, not cumulative)

**Catches:** delta not actually computing delta (e.g., recomputing absolute depth and forgetting subtraction).

### Phase E — Reset and re-strike
- Call `env.reset()`
- Manipulate nail depth to 0.005 (5 mm)
- Expected nail_depth_delta: **+0.005 × 500 = +2.5** (fires because reset zeroed `_max_depth`)

**Catches:** `reset(env_ids)` not actually zeroing the tracked state. This is the highest-value test — it's the single most common bug for stateful RewardTerms.

### Phase F — Bounce-back robustness
- Manipulate: nail depth 0.020 → 0.015 (simulating elastic rebound)
- Expected nail_depth_delta: **0** (clamp_min(0) prevents negative reward)
- Expected `_max_depth` after this step: still 0.020 (max preserved)

**Catches:** missing or broken clamp_min(0).

---

## Reusing this pattern for future reward terms

When adding a new reward term (e.g., `impact_velocity_bonus`, `air_time_bonus`, `completion_bonus`), extend `validate_rewards.py` with phases that exercise its specific failure modes:

| New term | Validation phase to add |
|---|---|
| `impact_velocity_bonus` | Trigger ContactSensor.compute_first_contact() artificially. Check bonus fires only at the transition step, proportional to head velocity. |
| `air_time_bonus` | Hold "no contact" for varying durations, then trigger contact. Check bonus scales with last_air_time. |
| `completion_bonus` | Force nail depth past success threshold. Check bonus fires once (or every step until termination — depends on design). |
| `joint_vel_penalty` | Set joint_vel to known values, check penalty magnitude. |

**General template per new term:**
1. **Happy path:** event the term targets actually happens → assert fires correctly
2. **Negative case:** event doesn't happen → assert returns 0
3. **Edge case:** event happens repeatedly → assert behavior is as designed (re-fire? one-shot?)
4. **Reset case:** if term is stateful, exercise the reset() hook

---

## Limitations and honest caveats

- Validation requires you to **state your expectations** for each phase. If your mental model of "what should happen" is wrong, validation passes but the reward is still buggy. Validation tests for **consistency**, not **correctness in the abstract**.
- Writing the scripted state sequences requires understanding the env's API for setting `joint_pos` / `qpos` directly. mjlab supports this via `entity.data.joint_pos[...] = ...` followed by `env.scene.write_data_to_sim()` (or similar — see code for exact pattern).
- The Gaussian shapes (`approach`, `nail_driven`) are hardest to assert exact values on — they depend on geometry. Use approximate assertions (within a tolerance) for these, exact assertions only for delta/bonus/penalty terms with clean formulas.
- Validation does not eliminate the need for training. It eliminates one large class of bugs that would otherwise be diagnosed during training.

---

## Where this fits in the workflow

```
1. Implement reward term  ─→  2. Validation script  ─→  3. Smoke training (few iters)  ─→  4. Full training
                              [catches term bugs]      [catches env+training bugs]       [final answer]
```

Step 2 is fast and local. Step 3 needs minimal compute. Step 4 needs real compute.

The validation script is rerun whenever any reward term is added or modified.

# Reward + constraint code fault audit (2026-07-19)

Two independent reviewers (Fable 5 + Codex) audited the Z1 hammer reward/constraint machinery for latent
faults — bugs, reward-hacking surfaces, and enforcement-day landmines. **CONSENSUS** = both flagged
independently (high confidence). Findings I verified against code are marked ✅. **None of these actively
corrupt the banked Phase-0/Phase-2 results** — they are latent/future-facing (the enforcement ones are
dormant at `imp_max_p=0`; the farm surface was never discovered by PPO — delivered stayed ~0.87× i_ref in
every arm). Nothing applied; this is a report.

## Ranked

### F1 — delivered_impulse is a farm surface; the depth-gate delays, not prevents [P1, CONSENSUS, ✅ verified]
`rewards.py:262-279` + `impulse_bound.py:262-279` + `env_cfgs.py:344-368`
The `SubstepDeliveredImpulse` accumulator re-arms 25 off-substeps after a window closes and adds each new
window to episode-cumulative `_total` — so the per-event cap bounds the **rate** (≤50% duty), not the
episode **total**. Two exploits:
- **`depth_gate=False`** (the Phase-2 toggle I added): pays every increment ungated → a 50 ms-on/50 ms-off
  press cycle farms a fresh window every 100 ms, unbounded. ✅ verified mechanism.
- **`depth_gate=True`** (shipped default): ✅ verified — non-progress impulse is **escrowed** (`_credited`
  held on non-advancing steps, `cur` keeps climbing during a press), then the whole backlog is paid on the
  next `>eps` advance. The gate delays farming to the next nudge, doesn't discard it. My docstring's
  "press-farming stays bounded either way" is **wrong**.
- **`no_terminate` + `depth_gate=False`**: worst case — seated nail on the hard stop, press is
  constraint-force (no 30 N friction ceiling), farm for the rest of the 20 s episode. Two independent CLI
  flags, **no guard** on the combination.
Latent today: default is `depth_gate=True`, terminating arm bounds collection to the 28→30 mm window, and
PPO never discovered it (dg0 sweep was a no-op).
**Fix (recommended):** discard non-progress impulse — advance the credit baseline **every** step and pay only
impulse causally paired with that step's depth progress (don't escrow). Add `raise ValueError` on
`no_terminate and not depth_gate`. Since Phase 2 proved `depth_gate=False` is a no-op, deleting the ungated
path entirely is the lazy+safe option.

### F2 — IMP_J_LIMIT time-basis mismatch (caps 27.3 ms vs window 50 ms) [P1, CONSENSUS]
`env_cfgs.py:27-35,235` + `impulse_bound.py:74-113`
Caps derived at Δt≈27.3 ms; the shipped Λ integrates a 50 ms window (impact ~44 ms). Safe **now** because
`imp_max_p=0`; **breaks when** `--...imp-max-p 0.5` flips (one flag) → current legitimate strikes graded
against caps calibrated for a materially shorter integral. Guarded only by a comment (the `0.1` placeholder
guard does **not** cover this — `IMP_J_LIMIT` isn't the placeholder).
**Fix:** re-derive all six caps at the shipped 25-substep window; bind `(window_substeps, physics_dt)`
metadata to the caps and assert it matches the accumulator when `imp_max_p>0`.

### F3 — δ multi-read enforcement is outcome-dependent [P2, CONSENSUS]
`impulse_bound.py:98-104` + `hammer_env_cfg.py:302,309`
Window(25) > decimation(10) → one violation spans ~3 control reads; a **completing** strike is truncated by
reset to ~1 read while a non-completing one eats ~3 → enforcement pressure depends on task outcome and is
~3× stronger than the C2-calibrated per-read `imp_max_p=0.5`. Safe now (δ≡0); breaks on enforcement.
**Fix:** emit one event-level δ pulse per violating window, or recalibrate per-read to a target per-event
survival and remove the terminal-tail asymmetry.

### F4 — `_NEG_TERMS` penalty classification is one-shot/stale [P2, CONSENSUS]
`cat/hook.py:127,144-155`
Polarity is inferred once (lazily) from term membership. A future curriculum that ramps a weight negative
after the first hook call → that penalty enters `r_pos` and gets `(1−δ)`-discounted → penalty-evasion
reopens. Safe now (the known `vel_penalty` composition is config-blocked). Latent.
**Fix:** revalidate polarity whenever weights change, or attach immutable `positive/penalty` metadata per term.

### F5 — incomplete CaT parameter domain validation [P2, Codex]
`cat/hook.py:42,85` + `constraint_manager.py:49`
Guard checks only one `imp_max_p<min_p` case + the all-`0.1` placeholder. No `0≤min_p≤max_p≤1`, `0≤tau<1`,
finite positive `imp_seed`/limits. `imp_max_p=2` → δ>1; bad EMA/seed → unstable normalization. Breaks on an
invalid enforcement sweep override.
**Fix:** validate finiteness + full probability/EMA/limit domains at hook construction.

### F6 — reward normalizers accept 0/NaN via CLI override [P2, Codex]
`rewards.py:50,209,268,361`
`std`, `v_expected`, `i_ref`, `sigma` are divisors used without validation. A CLI sweep passing 0/NaN →
non-finite rewards or a silently-eliminated term after framework sanitization. Safe now (config positive).
**Fix:** reject non-positive/non-finite normalizers at term construction.

### F7 — impact_progress first-contact gate is control-boundary fragile [P3, Fable]
`rewards.py:208` + mjlab `contact_sensor.py`
`compute_first_contact` misses (a) a strike that lands+rebounds within one 20 ms interval and (b) contact
straddling the step boundary → the 8.0-weight strike bonus silently drops. Underpayment only, but it's
phase-random noise on the largest per-strike term and **selects for press-through over clean elastic
strikes** — the wrong direction given the press-vs-ballistic finding. Same hole was fixed for
`ImitationPriorTerm`'s latch, left open here.
**Fix:** mirror the `last_contact_time` latch (OR a per-event once-latch), or latch v_axial at fc.

### F8 — Λ is blind to every non-nail impact [P3, Fable]
`impulse_bound.py:171-186`
Contact-mask = hammer↔nail sensor only; arm/handle-vs-block/floor impacts never enter Λ. Under enforcement,
δ-pressure could route impact energy through **unmonitored** contacts (brace handle/arm on block). Scopes
the thesis claim to "per-joint impulse bounded **at the nail contact**." Defense point; possible 2nd
constraint term later.

### F9 — cat_vel+cat_impulse falsely fires the kill-the-run alarm [P3, Fable]
`impulse_bound.py:346-353`
`impossible_success` treats `reset_terminated` as success; a CaT velocity termination pre-contact →
`terminated ∧ Λ=0` → alarm=1.0 ("dead instrument, kill run"). No guard on the composition.
**Fix:** key `succeeded` on the depth predicate alone (drop `terminated |`), or raise on the composition.

### F10 — hook-with-stock-PPO silently enforces nothing [P3, Fable]
`env_cfgs.py:281-304` + `rl/cat_ppo.py:52-61`
Pairing check is one-directional: CatPPO raises if the hook is missing, but stock-PPO **with** the hook
computes δ, logs plausible `cat_delta` curves, and enforces nothing. At C2 = a wrong-results run that looks
instrumented.
**Fix:** env-side one-shot assert that a consumer flagged `extras`, or one launcher flag that sets both.

### F11 — i_ref is an unversioned calibration literal [P3, Codex]
`env_cfgs.py:337,348`
`0.6094` hard-coded; comment already records a prior 7.5× shift. Any future geometry/reference/solver/
action-scale/VIC change silently shifts the delivered-reward share.
**Fix:** load from a versioned calibration artifact keyed to scene/reference/physics; fail on hash mismatch.

## Cleared by BOTH reviewers (don't re-audit)
Reset/per-env-state correctness across the success auto-reset boundary (`_credited`, `_prev_depth`, `_pulse`,
`_episode_peak`, ring `_buf`, `_off_streak`, `_event_age` — all reset/re-arm together, env_ids slices align,
global ring index partial-reset-safe); substep freshness (sensor cache invalidated every 500 Hz substep);
debounce/window math (flicker-chop pays exactly one window; monotone `torch.maximum` defenses sound); the
`Z1_JOINT_IMPULSE_LIMIT=0.1` placeholder guard; live numerics (no active div-by-zero; broadcasts safe);
`reduce="last"` dilution correctly routed via full-step readers.

## Bottom line
The reset/state/numerics core is genuinely clean. Residual risk clusters in **(a) the delivered_impulse farm
surface** (F1 — the code I touched; the gate delays not prevents) and **(b) enforcement-day flips guarded by
comments, not code** (F2–F5). All dormant today. Highest-value fixes if/when we resume: F1 (discard
non-progress impulse + guard/delete `depth_gate=False`) and the F2/F3 enforcement-calibration pair, which
gate ever turning the constraint on.

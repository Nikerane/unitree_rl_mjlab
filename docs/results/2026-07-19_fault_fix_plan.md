# Reward/constraint fault-fix plan (2026-07-19)

Fixes for the Fable5+Codex audit (`2026-07-19_reward_code_fault_audit.md`). Two tracks:
**Track A = CPU fixes now** (no Vega — reward code is fully CPU-testable). **Track B = GPU A/B when Vega
returns** — measure whether the flagged things actually *change training behavior*, per the user's ask.

Guiding split: most faults are either (i) **guards** that prevent a future footgun — no behavioral test, just
an assertion + unit test; or (ii) **enforcement-day** issues dormant at `imp_max_p=0` — untestable on GPU
until Khadiv greenlights enforcement. Only **F1** and **F7** change *current* fixed-impedance behavior, so
those are the two worth a GPU A/B.

---

## Track A — CPU fixes (do now, no Vega)

Each batch = one commit, each fix carries a unit test that fails on the old code. Gate: full pytest +
`validate_rewards` A–M green before commit.

### A1 — F1 delivered_impulse farm surface  ★ highest value (active-code risk)
- **Escrow → discard** (`rewards.py:262-279`): in the `depth_gate=True` path, advance the credit baseline
  **every** step so non-progress impulse is *dropped*, not banked. Pay only the increment causally paired
  with this step's depth advance. (Concretely: on a non-advancing step, set `_credited = max(_credited, cur)`
  too — so a later nudge can't collect a press backlog.)
- **Delete `depth_gate=False`** — Phase 2 proved it a no-op (a farm surface with zero demonstrated benefit).
  Removes the ungated path, the `depth_gate` param, the sbatch `DEPTH_GATE` passthrough, and the
  `no_terminate`+ungated combination entirely. (Lazy-correct: less code, closes F1's worst two exploits at
  once.) *If you'd rather keep the toggle for a future seated-nail context, the fallback is `raise ValueError`
  on `no_terminate and not depth_gate` + the escrow fix — say which.*
- **Test:** `test_no_press_backlog_farm` — accrue delivered while depth static, then advance once → pays only
  the *this-step* increment, not the banked total. Rewrite/retire the `depth_gate=False` tests.
- Fix my wrong docstring ("bounded either way").

### A2 — F7 impact_progress first-contact latch
- Mirror the `ImitationPriorTerm` fix (`rewards.py:208`): OR in `last_contact_time>0` with a per-event
  once-latch so a within-interval rebound or a boundary-straddling contact still pays the strike bonus.
- **Test:** `test_impact_progress_pays_on_boundary_contact` — contact that starts late in step k / advances in
  k+1 still fires fc exactly once.
- Note: this one plausibly *shifts behavior* (it currently selects for press-through over clean strikes) → B2.

### A3 — F5+F6 config-domain guards (footgun prevention on CLI sweeps)
- **F6** (`rewards.py` term `__init__`s): reject non-positive / non-finite `std`, `v_expected`, `i_ref`,
  `sigma` at construction.
- **F5** (`hook.py` / `constraint_manager.py`): validate full domains — `0≤min_p≤max_p≤1`, `0≤tau<1`, finite
  positive `imp_seed`, finite positive limits. Catches `imp_max_p=2 → δ>1` etc.
- **Test:** parametrized `pytest.raises` on each bad value. Cheap, high-value — you *do* run CLI sweeps.

### A4 — F9+F10 composition guards
- **F9** (`impulse_bound.py:346-353`): key `impossible_success` `succeeded` on the depth predicate alone (drop
  `terminated |`) so a CaT-vel termination pre-contact can't false-fire the kill-alarm. Test: terminated∧Λ=0∧
  depth<thresh → alarm 0.
- **F10** (`env_cfgs.py`/`cat_ppo.py`): env-side one-shot assert that a δ-consumer flagged `extras` — so
  hook-with-stock-PPO fails loud instead of silently enforcing nothing. Test: hook present + non-CatPPO → raises.

### A5 — F4 penalty-polarity revalidation (light) + F11 i_ref provenance (light)
- **F4** (`hook.py:127,144-155`): revalidate `_NEG_TERMS` polarity whenever weights change (cheapest: assert on
  each call that no `_pos` term has gone negative; ponytail — skip the full metadata-per-term refactor unless a
  curriculum actually needs it).
- **F11** (`env_cfgs.py:337,348`): **not** a versioned-artifact system (YAGNI for a thesis repo) — just make
  `derive_impulse_thresholds.py` the single source and add a provenance assert/comment tying `i_ref=0.6094` to
  the reference+physics it was measured under, so a geometry change trips a check.

### A-defer — F2 + F3 (enforcement-calibration; heaviest, needs Khadiv anyway)
- **F2** cap time-basis: re-derive the six `IMP_J_LIMIT` at the shipped 25-substep window and bind
  `(window_substeps, physics_dt)` metadata to the caps; assert match when `imp_max_p>0`.
- **F3** δ multi-read: the real work — event-level single δ pulse per violating window (or per-read recal to a
  target per-event survival, removing the terminal-tail asymmetry).
- **Why deferred:** both are dormant at `imp_max_p=0` and only matter *at the moment enforcement turns on*,
  which is **Khadiv decision (e)**. Do them **as part of the enforcement bring-up**, not now — but add a
  hard guard today (part of F2's metadata assert) so `imp_max_p>0` **cannot** be flipped until they're done.
  That converts "guarded by a comment" into "guarded by an assert" cheaply.

### A-note — F8 (no code): add the scoping sentence to the thesis/defense notes — Λ bounds impulse *at the
nail contact*; a future second constraint term would cover arm/world reaction. Not a bug to fix now.

**Track A order:** A1 → A2 → A3 → A4 → A5, each its own commit; F2-guard-only from A-defer folded into A3.
All CPU, all testable now.

---

## Track B — GPU A/B when Vega returns (does the fix *change behavior*?)

Only F1 and F7 touch current behavior; everything else is a guard or enforcement-gated. So two experiments,
both small (3 seeds, 500 iters, `imp_max_p=0`), reusing the fixed launchers:

### B1 — F1: was the farm truly latent, and does closing it change anything?
- **B1a (control that it's harmless):** train **escrow-fixed** vs **current** `depth_gate=True`. Expect
  *identical* delivered ~0.87× i_ref, success, ep_len → confirms the fix is behavior-neutral on the healthy
  policy (the farm was never used).
- **B1b (adversarial — was it exploitable at all?):** on the **OLD** code, train a high-entropy / longer run
  (or `no_terminate` + old `depth_gate=False`) explicitly trying to *elicit* the press-farm. If delivered
  blows past ~1× i_ref with ep_len pinned at max and nail parked <30 mm → the surface was real and we closed
  it just in time; if not → confirms PPO can't find it (stronger "latent" claim for the thesis).
- Metrics: delivered/i_ref, ep_len, nail_depth, contact-window length (the press-farm watchdog).

### B2 — F7: does the first-contact latch improve strike quality?
- Train **latched** vs **current** `impact_progress`. Hypothesis: the latch pays the strike bonus more
  consistently and *stops selecting for press-through*, so the fixed policy should strike slightly cleaner
  (shorter contact window, peak/mean force closer to a strike than a press) at equal success.
- Metrics: delivered/i_ref, contact-window length, peak/mean force, success. This is the one fix that could
  genuinely *shift* the fixed-impedance result — worth measuring before it's baked as canonical.

### B-defer — F2/F3 validation is **enforcement-only**: the with-vs-without-enforcement A/B belongs to the
Khadiv-gated `imp_max_p>0` campaign (needs the decision-(e) window/cap choice first). Not part of this batch.

**Track B needs:** Vega access restored + the checkpoints/launchers redeployed (the eval-fix deploy already
queued). Batch B1+B2 = 4 arms × 3 seeds = 12 runs, ~20 min each.

---

## Recommendation / sequencing
1. **Now (CPU):** Track A, A1 first (the real risk, in code I wrote), through A5. Commit each. Push.
2. **When Vega's back:** deploy the pending eval-fix, then run **B1 + B2** to confirm A1 is behavior-neutral
   and measure whether A2/F7 shifts the fixed-impedance result.
3. **Enforcement track (separate, Khadiv-gated):** F2 re-derivation + F3 event-level δ, then the
   with/without-enforcement A/B — only after decision (e).

Nothing here is applied yet. Suggest starting with **A1** (highest value, closes the farm I introduced).

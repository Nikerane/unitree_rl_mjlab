# Impulse-CaT C2→C3 two-track enablement — design

**Date:** 2026-07-06 · **Branch:** `soft-cat` · **Decided by:** user (chunk shape = **two-track**,
after weighing rigorous-first; Prof. Khadiv gave a verbal "give CaT a try" — treated as a
pragmatic discharge of the mechanism-confirm gate, with the C3 results as the formal confirmation
artifact at the next meeting).

## Goal

Unblock the fixture-era tree and produce the thesis's first **enforced** impulse-constraint GPU
results: enable `imp_max_p > 0` on the **baseline-subtracted** per-joint impulse Λ_j (Track 1,
the critical path), while building the **rigorous contact-row isolation** (`JᵀF` over only the
hammer↔nail efc rows) in parallel as a *logged diagnostic and validator* — never blocking the GPU
runs (Track 2).

## Why two-track (decision record)

- The baseline-subtracted accumulator is already shipped (`SubstepImpulseAccumulator`,
  `src/tasks/hammer/mdp/impulse_bound.py`, toggle at
  `src/tasks/hammer/config/z1/env_cfgs.py:226`); enforcing it costs zero new plumbing.
- The rigorous quantity is feasible (verified 2026-07-06: `mujoco_warp` exposes
  `Contact.efc_address`/`geom`/`dim` and `Data.efc` with sparse `J` (`J_rownnz`, `J_rowadr`,
  `J_colind`), `force`, `type`) but is ~1–2 days of new GPU-side code whose main risk is a silent
  indexing bug — exactly what the approximate column + the object-side ∫F·dt cross-check catch.
- Role split: **enforced quantity = baseline-subtracted Λ_j** (known ~20% friction residue,
  defended as a conservative over-count); **rigorous contact-row Λ_j = logged metric** that (a)
  produces the methods-chapter contamination figure (raw vs subtracted vs contact-rows), (b)
  three-way-validates against the object-side ∫F·dt, and (c) can be promoted to the enforced
  quantity in a single re-run arm later if the residue proves material.

## Preconditions being discharged (authority: `env_cfgs.py:237-238` a/b/c gate)

- (a) C0 quantity gate **on the current tree**: re-run `derive_impulse_thresholds.py` after
  adopting the NEAR_NAIL re-solve (gripper-era numbers are void; the L6 grasp re-staled the pose
  2026-07-06: hammer body `pos -0.0506 0.001 0`, `quat` −90°Z in both sibling XMLs).
  **AMENDMENT (2026-07-06 review):** the re-solve was already done by the user in the working tree
  (`z1_constants.py` "RE-SOLVED 2026-07-06" block): with the new grasp a vertical strike AT the
  floor nail is **kinematically impossible** (in-plane face-down reachable only above z≈0.20), so
  the adopted reset poises the face at (0.5, 0, **0.25**) — the policy drives down and discovers
  the (necessarily oblique near the floor) contact itself. The plan ADOPTS and verifies this pose;
  it does not redo the solve.
- (b) Mechanism confirm: verbal go-ahead received (see header); C3 results = formal confirmation.
- (c) Local gate green: blocked today only by the NEAR_NAIL re-solve; re-green sequence below.
- Pinocchio side-gate: **record the standing decision** in `IMPULSE_CAT_IMPL_PLAN.md` — the
  MuJoCo-native object-side ∫F·dt is the accepted ground truth; Pinocchio `impulseDynamics` stays
  scaffolded/deferred (matches the 2026-06 decision in visual plan plan-c00a0e0144274fe4).

## Track 1 — critical path (strictly ordered)

1. **Commit the standing tree** (approved by user 2026-07-06). Grouped, unpushed commits in BOTH
   repos: this repo (impulse machinery / docs consolidation / tests, excluding `.env`, `.DS_Store`,
   `graphify-out/`) and the sibling `safe_impact_manipulation` (fixture + grasp + mocap-mirror
   asset changes on branch `hammer-z1`).
2. **Adopt + verify the 2026-07-06 windup re-solve** (already in the working tree, user-authored —
   see the amendment above). No IK re-run. Verify on the adopted pose: `playback_reference.py`
   (gate = **best**-strike ≥ the 0.027 m threshold; the press probe is a recorded watchdog number,
   not a pass criterion), `test_single_strike.py`, `render_reference.py` visual. If the reference
   no longer clears the threshold from the z=0.25 reset, STOP — the Q1 threshold/pose trade-off is
   the user's decision.
3. **Threshold re-derivation + config flip.** Run `derive_impulse_thresholds.py` (gate must PASS;
   exits 1 on FAIL). Consume its output: per-joint `J_limit` tensor replaces the scalar
   placeholder (`imp_limit` at `env_cfgs.py:248`; `Z1_JOINT_IMPULSE_LIMIT=0.1` at
   `impulse_bound.py:54` stays as the rejected sentinel); `i_ref` for `delivered_impulse`
   (`env_cfgs.py:265`) set from the script; `imp_seed` **stays `1e-3`** (decay floor — the "paste
   p95" comments at `env_cfgs.py:250` and the derive-script docstring are stale, clean them);
   `subtract_baseline: False → True` (`env_cfgs.py:226`). Record the Pinocchio decision line.
4. **Gate re-green.** `validate_rewards.py` phases A–M, `verify_contact_sensor.py`,
   `verify_reward_setup.py`, `test_single_strike.py` (Q1 re-confirm — CLAUDE.md mandate after any
   EE/pose change), the 5-file pytest set, then the full suite.
5. **C2 — enforcement smoke + mini-sweep (local/CPU).** `imp_max_p ∈ {0.25, 0.5}`.
   **AMENDMENT (2026-07-06 review):** reference strikes measure ~5% of J_limit (C0: the cap binds
   at ~21× more violent), so on real caps the δ machinery would never fire during the gate — the
   machinery hard-gates therefore run against a **scaled-down probe limit** (J_limit × 0.05) that
   forces over-limit events: c_max bounded / not pinned at seed / not single-event-spiked (>100×
   worst excess); δ-vs-excess **graded, not saturated**; strike physically unchanged by δ. At the
   REAL caps the gate *reports* the binding statistics (fraction of strikes in
   `[0.8·J_limit, J_limit]`, max Λ/J_limit ratio) without failing on them. The **δ-attributable
   peak-impulse reduction vacuous-check moves to C3 analysis** (same-seed `c3_imp` vs `c3_track`):
   ≈0 reduction with δ≈0 there ⇒ the vacuous verdict blocks *conclusions/thesis claims*, not the
   already-spent run. This is a deliberate re-scoping, not a downgrade: the reduction is a property
   of *learned* policies and cannot be measured on a 5-iteration smoke.
6. **C3 — Vega results run.** Arms: `Unitree-Z1-Hammer-CaT-Impulse` (chosen max_p/τ),
   `Unitree-Z1-Hammer-Track` (r_imit-only baseline), `Unitree-Z1-Hammer-CaT-Soft` (velocity CaT,
   **re-run on the new tree** for a same-tree comparison). 3 seeds each via
   `sbatch --array=0-2 scripts/slurm/train_array.sbatch` — **one GPU per run, never multi-GPU**.
   Submission happens from the Vega login node (user-side or via existing SSH setup — flagged as a
   possible manual step). Rsync logs back; write a dated record under `docs/results/`.

## Track 2 — rigorous contact-row isolation (parallel; starts any time after step 1, must not gate step 6)

7. **`ContactRowImpulseAccumulator`** (new mode/class beside the shipped accumulator): per substep,
   filter contacts to the hammer↔nail geom pair, gather their efc rows via `efc_address` (+ row
   count from `dim`/cone type), compute `(JᵀF)_j` on the six arm dofs from the sparse `J`,
   accumulate `|·|·dt` with the same per-event pulse semantics as the shipped accumulator.
8. **Exact-reconstruction invariant test:** Σ over ALL efc rows of `JᵀF` must equal
   `qfrc_constraint` to numerical precision (per env, per substep, on a real strike) — proves the
   row split before the contact-only subset is trusted. Plus unit tests for geom filtering and
   pulse semantics.
9. **Three-way gate extension + contamination figure.** Extend `derive_impulse_thresholds.py` (or
   a sibling section) to print raw vs baseline-subtracted vs contact-rows vs object-side ∫F·dt on
   the reference strikes; export the methods-chapter figure. Wire the new column into training
   logs (metric only, `imp_max_p` untouched by it).

## Error handling

- Re-solve fails to converge → retry with seed = current constants (known-good basin); if still
  failing, report joint-limit diagnostics to the user (do not hand-tune the pose).
- Gate FAIL at step 3/4 → stop the pipeline, surface the failing section verbatim; no config flip
  past a red gate.
- δ saturated or vacuous at C2 → do not submit C3; report the δ-vs-excess plot and the binding-ness
  stats; tuning τ/robust statistic is in scope, changing J_limit is NOT (it is a measurement).
- Track 2 disagreement (contact-rows vs object-side beyond the friction residue) → treated as a
  Track-2 bug until proven otherwise; never blocks Track 1, but blocks any promotion decision.
- Track 2 multi-env attribution: `mujoco_warp`'s efc struct has **no `worldid` field** (verified
  2026-07-06) — the contact-row metric may be wired into the *training* config only after a
  `num_envs=2` attribution test passes (contact→world via `contact.worldid`); until then it ships
  gate/eval-only. Silently-garbage batched logging is worse than no logging.

## Testing

Every new code path gets pytest coverage (accumulator modes, invariant, hook interaction);
`validate_rewards.py` Phase M extended to certify whichever accumulator modes ship; the standing
5-file pre-train pytest set stays the gate. Full suite green before every commit.

## Success criteria

1. Gate a/b/c all discharged and documented; `imp_max_p > 0` running with measured per-joint caps.
2. C2 hard gates pass with a graded δ and non-vacuous binding-ness.
3. C3: three arms × 3 seeds complete on Vega; dated results record exists comparing delivered
   impulse, per-joint Λ tails, success rate, and strike formation across arms.
4. Track 2: invariant test green; three-way comparison logged; contamination figure exported.
5. All docs/memory that tracked the blockers updated (IMPULSE_CAT_IMPL_PLAN §6b + Status,
   OPEN_QUESTIONS L168/179, `l6-hammer-fixture-ee` memory, freshness guard green).

## Amendments — round 2 (2026-07-06, three-auditor review: max-impact methodology, observability, executability)

1. **Fourth arm `c3_imp0`.** `c3_track` optimizes a different objective (no `delivered_impulse`,
   plus `r_imit`, stock PPO), so nothing "δ-attributable" is identifiable against it. The proper
   unconstrained baseline is `-CaT-Impulse` with the repo-default `imp_max_p=0` (CatPPO at δ≡0
   reduces exactly to stock PPO — verified in `cat_ppo.py`). The vacuous-check compares
   `c3_imp` vs `c3_imp0`; `c3_track`/`c3_catsoft` are secondary comparators only.
2. **Observability track (pre-submission).** Per-joint episode-peak Λ training metrics
   (`Episode_Metrics/imp_peak_joint1..6` — full-step readers of the accumulator, which also fixes
   the substep-mean dilution of the existing worst-joint scalar), a `cat_delta_peak` metric
   (episode-mean δ dilutes strike-time δ by ~episode length), `scripts/diag_impulse_trace.py`
   (per-strike substep-resolution force-propagation figure, checkpoint AND reference modes), and
   `scripts/compare_runs.py` (cross-run curves, per-arm mean±std across seeds, missing-key-safe).
   These MUST land before the Vega submission or the runs won't carry the per-joint keys.
3. **Checkpoint-eval task (post-queue, the impl plan's C4 analogue the two-track plan had
   dropped).** All checkpoints (4 arms × 3 seeds) rolled in the SAME instrumented env
   (`-CaT-Impulse` play cfg, `imp_max_p=0` log-only). Pinned protocol: final checkpoint
   `model_{ITERS-1}.pt`, 256 envs, ≥512 episodes per policy, `play=True`, mean-action rollout
   primary (sampled as robustness check), fixed eval seed, eval job id recorded.
4. **Maximize-side evidence mandated in the record**: (a) delivered-impulse training curve,
   `c3_imp0` overlaid on `c3_imp` — the converged gap IS the delivered-impulse cost of the cap;
   (b) eval frontier: per-episode delivered impulse vs worst-joint Λ/Λ̄ across arms.
5. **C2 weight-share step** (fulfils the config's "C2 tunes the weight" promise): per-term
   weighted episode-return table on the driven reference strikes incl. `delivered_impulse`'s share
   of positive return; hard-gate sanity only (fires, positive, finite); the weight decision is the
   user's, recorded with rationale.
6. **Probe-pass corrections**: heights run DESCENDING (the hook seeds its normalizer from the
   first over-limit sample — ascending order saturates δ on the very first strike and the graded-δ
   gate fails spuriously); graded-δ gate = "≥1 over-limit strike with 0 < δ < max_p"; the
   physics-identical check is a regression tripwire only (δ has no physics pathway in playback) and
   is complemented by the real check: probe δ>0 on contact steps, log-only baseline δ≡0.
7. **Provenance gap (STOP/ask)**: the windup-sweep solver was not preserved in either repo — the
   user either supplies it for commit or the `z1_constants.py` provenance comment is amended to
   note the solver was not preserved.

## Out of scope

Variable impedance (comes after fixed-impedance results, per CLAUDE.md); the N-P3O baseline arm
(optional, separate decision); promoting the rigorous quantity to enforcement (explicit later
decision, one re-run arm); G1 replication.

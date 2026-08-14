# Wave-2 waypoint-guidance replication — result (2026-08-03)

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer cap”
> wording below refers only to the historical registered-task boundary, not a validated
> Z1 reaction-impulse or damage limit. The result body remains frozen; see
> `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

Preregistration: `docs/results/2026-08-03_wave2_waypoint_replication_prereg.md`
(registered, and the training revision pinned, **before** Wave 2 trained).
Pooled population: **18 policies, 6 seeds per arm** — Wave-1 seeds 2–3 (frozen) plus
Wave-2 seeds 4–7. No policy was replaced, filtered or excluded.

Every endpoint and threshold below was fixed in advance. Nothing here was chosen after
seeing the data, and no threshold was moved to fit it.

## Headline

**Both guided arms complete all six gates on a majority of seeds; the control does so on
none.** Under the stricter three-clause straight label both guided arms come out at
exactly half their seeds. Guidance is not reliable in either form: 3 of 12 guided seeds
ignore the guideline entirely while still driving the nail.

The two questions this wave was run to answer:

- *Does progress-only's 2/2 replicate across six seeds?* **Largely — 5/6 complete all
  six gates.**
- *Does gate-only remain unreliable?* **It is 4/6, against 1/2 in Wave 1.** The pooled
  counts differ by one policy out of six. This design does not support ranking the two
  treatments against each other; it supports separating both from the control.

**Backend disclosure — read before any number below.** All 18 policies were TRAINED on
CUDA (A100, `device=cuda:0`) and every measurement reported here was produced on CPU at
a fixed reset. CPU/CUDA substep differences on this task are known and unmeasured for
the impulse channels. No metric here mixes the two backends: each row's trajectory and
impulse halves are both CPU runs of the same checkpoint at the same realized reset.

## Primary endpoint — all six gates crossed no later than first contact onset

| Arm | Completed | Per-seed gates@onset (2, 3 \| 4, 5, 6, 7) |
| --- | --- | --- |
| C0 | **0/6** | 0, 0 \| 1, 0, 0, 0 |
| G | **4/6** | 0, 6 \| 6, 6, 0, 6 |
| P | **5/6** | 6, 6 \| 6, 6, 0, 6 |

The control crossed a single gate once (C0/4) and never more. Neither treatment reaches
6/6: **G/2, G/6 and P/6 are guidance failures** — all three strike the nail and drive it
to the 32 mm stop while crossing zero gates. That is **3 of the 12 guided seeds** across
both waves: a successful *task* policy that ignores the *guidance*.

**Onset vs full-rollout gate counts disagree on 0 of 18 policies.** The frozen Wave-1
column and the new preregistered one coincide across the whole pooled set.

## Predeclared straight label — all three clauses

Six gates by onset **and** pre-contact max segment error ≤ 15 mm **and** pre-contact
path ratio ≤ 1.15:

| Arm | Straight | Seeds |
| --- | --- | --- |
| C0 | **0/6** | none |
| G | **3/6** | 3, 5, 7 |
| P | **3/6** | 2, 3, 7 |

**Three gate-completers fail the 15 mm clause and are honestly counted as not straight:**

| Policy | pre-contact max | verdict |
| --- | --- | --- |
| G/4 | 15.86 mm | gates, but wide |
| P/4 | 16.46 mm | gates, but wide |
| P/5 | 16.17 mm | gates, but wide |

This is the most interesting thing in the wave, and it is a direct consequence of having
predeclared the threshold. The 15 mm bound was set from Wave 1, where the three
completers ran **10.88–11.88 mm** and nothing sat near the boundary. Wave-2 completers
run **13.78–16.46 mm** — systematically wider, straddling the line. Had the threshold
been chosen now, it would have been drawn elsewhere and the result would have been
unfalsifiable. **The threshold is not moved.**

The honest reading: on the *primary* endpoint the guided and control arms do not overlap
at all (9 guided completions, 0 control); the *straight label* is a stricter,
calibration-sensitive refinement whose count depends on a bound fixed from three Wave-1
policies, and Wave 1's completer tightness now looks unrepresentative.

## Secondary endpoints — median [min, max] over 6 seeds

Aggregates are taken over the published CSV's rounded per-policy values, so a re-derivation
at full float precision moves four cells by 0.01 (C0 RMS 31.09→31.08, G max 15.36→15.37,
P max 14.98→14.97, P RMS 6.55→6.56). Immaterial to every claim here, but stated so a
re-derivation does not read as a discrepancy.

| Arm | pre-contact max (mm) | pre-contact RMS (mm) | path ratio | nail depth (mm) | peak qvel (rad/s) | delivered (N·s) | max Λ/cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 51.94 [36.98, 381.40] | 31.09 [21.46, 270.41] | 1.43 [1.23, 16.78] | 32.00 [0.00, 32.00] | 5.01 [4.98, 5.26] | 0.32 [0.00, 0.36] | 0.07 [0.00, 0.10] |
| G | 15.36 [10.88, 54.16] | 6.36 [5.92, 37.60] | 1.04 [1.04, 1.52] | 32.00 [32.00, 32.00] | 4.82 [4.53, 5.00] | 0.34 [0.32, 0.38] | 0.09 [0.05, 0.13] |
| P | 14.98 [11.50, 50.41] | 6.55 [5.48, 39.11] | 1.04 [1.03, 1.64] | 32.00 [32.00, 32.00] | 4.95 [4.38, 4.98] | 0.34 [0.31, 0.38] | 0.08 [0.05, 0.10] |

The medians separate treatments from control by roughly 3.5× on max error and ~5× on
RMS, and the ranges overlap **only** through each arm's own guidance failures.

Within the two guided arms the outcome is bimodal rather than continuous: a guided policy
either tracks the guideline (10.88–16.46 mm, ratio 1.03–1.10) or ignores it entirely
(50.41–54.16 mm, ratio 1.41–1.64), with no guided policy between those bands. **This does
not hold across all 18**: the control C0/4 sits in between at 36.98 mm, ratio 1.2288, with
one gate crossed, and C0/2 lies far outside either band at 381.40 mm, ratio 16.78.

**Full-rollout segment error, reported alongside as preregistered.** Over the whole
rollout the gate-completers span **18.65–32.20 mm** against their 10.88–16.46 mm
pre-contact; the non-completers are unchanged (36.98–381.40 mm) because their windows are
the whole rollout already. The completers' figure roughly doubles because the hammer
follows through past the nail endpoint, where finite-segment projection pins the closest
point at the nail. Full-rollout error is **not** the path-quality result and is not used
for any endpoint or threshold — it is reported because the preregistration requires it,
and because computing straightness this way is precisely the defect a reviewer caught in
Wave 1.

**No population-level or significance claim is made, and none is available here.** At n=6
per arm these are exact counts and descriptive ranges. No test was run, no variance
estimated, and nothing below should be read as evidence that one treatment beats the
other.

## Task outcome and safety — all 18

| | |
| --- | --- |
| Contact | 17/18 |
| Terminated success | 17/18 |
| Nail driven to the 32 mm stop | 17/18 |
| **qvel legal** | **0/18** — observed 4.3751–5.2556 vs the 3.1415 rad/s hardware limit |
| Max Λ/cap over all 18 | **0.127732** (G/4, joint 1); range 0.000000–0.127732 |

The single non-striker is **C0 seed 2**, the Wave-1 canonical negative, which never
contacts the nail at all (Λ, delivered impulse, axial force and contact all exactly
zero). It is reported, not omitted.

**Velocity legality fails universally and this is measured, not enforced.** Velocity CaT
and deterministic velocity termination were disabled for this replication by design.
Every one of the 18 policies would violate the Z1's joint-velocity limit on hardware.
That is an observation this wave records; it is not a result about enforcement, and it
is a separate experiment.

**The impulse constraint remains far from binding.** `imp_max_p = 0.0` throughout, so Λ
is logged and never acted on. The worst joint across 18 policies reaches 12.8% of its
manufacturer cap. Nothing here is near the constraint at this strike energy.

## Per-policy table (18 rows)

`evaluation/results/2026-08-03_wave2_waypoint/tables/wave1_wave2_eighteen_policy_comparison.csv`
— sha256 `57ff1ba7419b99cc13b55b23af94b87371d4790e7397b64ab4ae9f5f2f731862`

| arm/seed | wave | gates@onset | pre max (mm) | pre RMS (mm) | ratio | depth (mm) | qvel | delivered | max Λ/cap | payout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0/2 | 1 | 0 | 381.40 | 270.41 | 16.7777 | 0.0 | 5.2556 | 0.000000 | 0.000000 | 0.0 |
| C0/3 | 1 | 0 | 52.50 | 31.16 | 1.4204 | 32.0 | 4.9811 | 0.346613 | 0.097105 | 0.0 |
| G/2 | 1 | 0 | 54.00 | 32.59 | 1.4115 | 32.0 | 5.0016 | 0.364718 | 0.102874 | 0.0 |
| G/3 | 1 | **6** | 10.88 | 6.00 | 1.0392 | 32.0 | 4.9590 | 0.345315 | 0.051076 | 0.16 |
| P/2 | 1 | **6** | 11.88 | 7.92 | 1.0971 | 32.0 | 4.9661 | 0.308886 | 0.094214 | 0.16 |
| P/3 | 1 | **6** | 11.50 | 5.48 | 1.0347 | 32.0 | 4.9749 | 0.341845 | 0.050454 | 0.16 |
| C0/4 | 2 | 1 | 36.98 | 21.46 | 1.2288 | 32.0 | 5.1091 | 0.357834 | 0.083444 | 0.0 |
| C0/5 | 2 | 0 | 51.38 | 31.01 | 1.4365 | 32.0 | 4.9834 | 0.321521 | 0.047044 | 0.0 |
| C0/6 | 2 | 0 | 54.53 | 33.68 | 1.5052 | 32.0 | 5.0327 | 0.321008 | 0.091939 | 0.0 |
| C0/7 | 2 | 0 | 51.06 | 30.03 | 1.4321 | 32.0 | 4.9820 | 0.309353 | 0.046201 | 0.0 |
| G/4 | 2 | **6** | 15.86 | 6.30 | 1.0371 | 32.0 | 4.5699 | 0.377030 | **0.127732** | 0.16 |
| G/5 | 2 | **6** | 13.83 | 5.92 | 1.0398 | 32.0 | 4.5309 | 0.318441 | 0.114516 | 0.16 |
| G/6 | 2 | 0 | 54.16 | 37.60 | 1.5194 | 32.0 | 4.9777 | 0.321789 | 0.059157 | 0.0 |
| G/7 | 2 | **6** | 14.87 | 6.42 | 1.0356 | 32.0 | 4.6855 | 0.326811 | 0.079262 | 0.16 |
| P/4 | 2 | **6** | 16.46 | 6.41 | 1.0344 | 32.0 | 4.9239 | 0.381055 | 0.068371 | 0.16 |
| P/5 | 2 | **6** | 16.17 | 6.70 | 1.0413 | 32.0 | 4.3751 | 0.337911 | 0.076535 | 0.16 |
| P/6 | 2 | 0 | 50.41 | 39.11 | 1.6372 | 32.0 | 4.9779 | 0.308072 | 0.084569 | 0.001071 |
| P/7 | 2 | **6** | 13.78 | 5.85 | 1.0298 | 32.0 | 4.6735 | 0.377170 | 0.096834 | 0.16 |

`payout` is the summed treatment-term payout. Gate-completers earn the full 0.16;
G/2 and G/6 earn **exactly zero**; P/6 earns 0.001071 — `r_waypoint_progress` pays
new-best ordered approach credit toward the next gate
(`src/tasks/hammer/mdp/guideline.py`, `window_new_credit`), so P/6 made a trace of
forward progress toward gate 1 and never completed it. There is no corridor or tube in
this study — Wave 1 and Wave 2 both trained a **waypoint guideline**. C0 has no
treatment term by construction.

`ratio` is `precontact_path_ratio`, added in Wave 2 because the frozen
`traj_path_len_ratio` emits a blank for a non-contacting policy (it hardcodes a zero
chord), which would have left C0/2 unlabelable. The two agree wherever both are defined.

## Measurement conventions (as preregistered, unchanged)

- Perpendicular error is the recorded `substep_perpendicular_error_m` — distance to the
  **finite** entry→nail segment. It is never re-derived against an infinite line.
- Pre-contact = substeps from reset **through the first contact-onset sample, inclusive**.
  A non-contacting policy's window is the entire rollout.
- Gate completion, path ratio and the perpendicular metrics all share that one window.
- Full-rollout segment error is reported alongside (above) but is **not** the
  path-quality result: it inflates with post-strike follow-through past the nail.
- Nail depth is the clamped value at the 32 mm stop, not raw elastic overshoot.

## Verification

This result was produced under the preregistration's requirement that the 6→18 row
extension be independently reviewed **before** any Wave-2 number was reported.

- **Analysis extension (before any Wave-2 number existed).** Two independent reviewers,
  both read-only, both returned REQUEST-CHANGES. Neither found an arithmetic error;
  both independently rebuilt the frozen Wave-1 projection from scratch and reproduced
  `4ab58b67…`. Findings resolved before Phase C: the predeclared straight label's third
  clause was uncomputable for a non-contacting policy (fixed by appending
  `precontact_path_ratio`, leaving the frozen column untouched); the trajectory/impulse
  join was asserted but never checked (now verified per row, including the reset digest);
  the "one shared" pre-contact window was derived three times and the unit-tested helper
  was not the shipped path (now unified); the artifact validator had no production caller
  (now `validate_wave2_artifacts.py`, run against all 12 leaves). Committed at
  `bf498ff` with a full suite of 2088 passed, 1 skipped.
- **Result note.** Independently cross-checked against this preregistration. Ten issues
  were raised and all were corrected here, including: a guided-failure undercount
  (2/12 → **3/12**, G/2 had been dropped); significance-flavoured wording at n=6; a claim
  attributed to the frozen Wave-1 note that it had explicitly refused to make; "most
  seeds" where the predeclared label gives exactly half; a false "nothing in between
  anywhere in the 18" (C0/4 sits between the bands); the missing full-rollout companion
  metric; the missing CUDA-training/CPU-measurement disclosure; an overstated
  startup-check field count and uncommitted Slurm exit states; and a "brushed the
  corridor" phrase that named both the wrong mechanism and an out-of-scope treatment.
- **Raw-data verification.** An independent reviewer rebuilt the 18-row table directly
  from the raw `trace.npz` pairs, writing the arithmetic from the preregistered
  definitions rather than importing the generator.

## Provenance

| Role | Revision |
| --- | --- |
| Wave-2 training | `e1a0282c9dc7b0283ae632a46a78debfd80bdf8c` |
| Wave-1 training | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |
| Analysis / rendering | `bf498ff3af992ed0f0c041484ecdbb4bf9d38a71` |
| Asset | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` |

Wave-2 training-relevant source is byte-identical to Wave 1 (`src/` tree `bf6596c7`,
`vega_train.sbatch` `e2b8abb6`, `train.py` `f727e288`). All 12 Wave-2 runs recorded
`train_done_rc = 0` and clean provenance in `wave2_training_inventory.json`, and all 12
passed the preregistered startup check (10 fields: task, seed, iterations, environments,
code revision, asset revision, clean state, absent reward overrides, `imp_max_p`, caps).
No job was cancelled, retried or re-seeded, so the retry rule was never invoked.

The Slurm exit states (`COMPLETED`, ExitCode `0:0`, job arrays `40606838/39/40`) were
read from `sacct` at the time and are **not** captured in a committed artifact; what is
committed is the per-run `train_done_rc`, the job IDs, and the launcher logs under
`provenance/`. The startup check itself likewise left no artifact file — a gap worth
closing for the next campaign.

**Execution device is now recorded directly**, not inferred: all 12 Wave-2 leaves record
`requested=cpu`, `actual_env_device=cpu`, `actual_tensor_device=cpu`,
`platform=macOS-26.5.1-arm64-arm-64bit`, and 12/12 passed the artifact contract. Wave-1
artifacts are unchanged and retain their documented inferred-CPU bridge.

The impulse half of every row carries no device field, and does not need one:
`scripts/diag_impulse_trace.py:1191` aborts on any non-CPU device, so its CPU provenance
is enforced by construction rather than recorded and trusted.

**Frozen Wave-1 projection.** The 18-row generator reproduces the six Wave-1 rows
byte-for-byte in every pre-existing column against the frozen CSV
`4ab58b67658ef6c899a21ffbd200d4c568327352c72da0dd1c386437dda86268`. This is re-proven on
every run of the generator, not only in the test suite. New columns were appended, never
substituted; the Wave-1 CSV and its hash are not reissued.

## Figures and videos

- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_eighteen_panel_xz.png`
- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_eighteen_panel_xy.png`
- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_per_arm_summary.png`

All 18 videos, 50 fps slow-motion at the 500 Hz substep rate:

| Wave | Path |
| --- | --- |
| 1 | `evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_{c0,g,p}_seed{2,3}/policy.mp4` |
| 2 | `evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_{c0,g,p}_seed{4,5,6,7}/policy.mp4` |

Each leaf also carries `montage.png`, `trajectory.png`, `trace.npz` and `metadata.json`.

## What this does NOT establish

- **No population-level or causal claim.** Six seeds per arm supports exact counts and
  ranges only. G 4/6 vs P 5/6 is one policy; do not read a treatment ranking into it.
- **A single fixed reset is not a generalization test.** All 18 are one-initial-condition
  replays chosen for comparability.
- **200 iterations only.** Whether C0 seed 2's collapse is undertraining is untested and
  was deliberately excluded from this wave.
- **No constraint enforcement, no velocity enforcement, no VIC.** Λ and qvel are logged.
- **Guidance does not guarantee straightness**, and straightness is not the task: G/2,
  G/6 and P/6 drove the nail fully while ignoring the guideline entirely.
- **Training was CUDA, every measurement is CPU.** Known CPU/CUDA substep differences on
  this task are unmeasured for the impulse channels, and CUDA fixed-reset impulse
  validation remains unavailable because `diag_impulse_trace.py` is CPU-only by design.
  The two backends are never combined within a row.
- **No ranking of G against P is supported.** 4/6 vs 5/6 on the primary endpoint, and
  3/6 vs 3/6 under the straight label, is one policy of difference at n=6.

## Recommended next step

The straightness effect is established well enough that more seeds at this configuration
would buy little. The two open questions worth separating are now:

1. **Why do 3/12 guided seeds ignore the guidance entirely** while still succeeding at
   the task? G/2, G/6 and P/6 all reach the 32 mm stop with zero or near-zero treatment
   payout — the task reward is evidently sufficient on its own, and the guidance term is
   not binding for them. This is a reward-balance question, not a seed-count question.
2. **Velocity legality**, which fails 18/18 and is the actual blocker for hardware.

Neither requires another replication wave.

# Z1 impulse soft-CaT Step 1 — binding, attribution, and dose calibration

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-15
- Verification Status: ANALYZED
- Version Label: z1_impulse_cat_step1_v1
- Source: frozen VIC-TT seed-2 checkpoint plus one explicitly diagnostic two-arm canary
- Overall Confidence: SOLID for telemetry, binding, and attribution; CAUTION for event-dose calibration, threshold meaning, and training effects

## Verdict

**Verdict 2: the provisional no-`2x` project boundary binds in representative
training-like stochastic VIC behavior and its per-read pressure is independently active and
graded, but complete physical-event dose is badly calibrated by task-terminal censoring. Revise
the event calibration and stop before scientific training.** It does not bind in the 64-world
deterministic mean-policy population, and it is not masked by velocity CaT in the sampled tail.

The boundary is not a Unitree damage or hardware limit. It remains a provisional project
threshold derived from the old arithmetic after retiring the unsupported global `2x`; the retained
`27.333 ms` factor is itself a censored historical contact prefix. This experiment qualifies the
constraint mechanism and identifies where it activates. It does not qualify physical actuator
safety or select `imp_max_p`.

## What changed

The existing `CatSoftHook` gained pull-only constraint telemetry; the PPO algorithm, environment
physics, actions, rewards, and rollout data path were not replaced. The telemetry exposes finite,
detached batched values for velocity and impulse deltas, their exact max-combination, the winning
constraint and responsible joint, raw per-joint Lambda, active caps, utilization, and signed/positive
margins. The authoritative accumulator Lambda is reported directly rather than reconstructed from a
subtraction.

The no-learning survey reuses the frozen evaluator seams and captures telemetry before auto-reset.
It records 500 Hz contact state separately from 50 Hz controller reads, reports completed contacts
separately from right-censored prefixes, preserves ambiguous physical-event associations, and
replays candidate `imp_max_p` values offline from one recorded trace. No learning occurs in the
survey.

## Frozen protocol

- Code: `8e61caf9618aa27ff403f94ffef091ab962c6dd7`
- Assets: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Checkpoint: VIC-TT seed 2 `model_499.pt`, SHA-256
  `c1544b779e78e7323bf02ce7b0f165745ee63b5aad9f930f9eea64eb6ea4ea77`
- Quantity: baseline-subtracted, contact-masked absolute joint-reaction impulse at 500 Hz in a
  25-substep/50 ms sliding window
- Velocity CaT: substep detection, `limit=3.1415 rad/s`, `max_p=0.5`
- Live impulse arm: `imp_max_p=0`; provisional project caps
  `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N·m·s`
- Fixed population: 64 environments, deterministic mean policy, complete first episode, no
  auto-reset, seed `2026081202`
- Training-like population: 4,096 environments x 24 controller steps, sampled policy, auto-reset,
  seed `2`; 98,304 controller reads and 12,292 observed episode segments
- Survey job `41272370`: `COMPLETED/0:0` in 2m29s on one A100; stderr empty

## Result

In the table, Lambda values are observed episode/segment peaks shown as `median / p95 / max`.
Utilization is relative to the provisional no-`2x` cap. `P_event` summaries are
`median / p95 / max`. The first triple uses all 580 unambiguous one-read **observed physical-event
prefixes**; only 9 completed before task termination, so their completed-only triple is shown in
parentheses. These values describe exposure recorded by the task pipeline, not a complete-contact
dose distribution.

| Scope | Boundary or dose | Fixed mean population | Training-like sampled population | Separate CaT attribution / event dose |
|---|---:|---|---|---|
| J1 | `0.820 N·m·s` | `0.467 / 0.472 / 0.482`; max `58.8%` | `0.463 / 0.483 / 1.226`; max `149.5%`; 6 violating reads, 4 associated events | J1 was responsible on 3 reads across 2 event IDs; per-joint counts can overlap |
| J2 | `1.640 N·m·s` | `0.392 / 0.397 / 0.401`; max `24.5%` | `0.388 / 0.401 / 1.089`; max `66.4%`; 0 violating reads | Nonbinding |
| J3 | `0.820 N·m·s` | `0.756 / 0.779 / 0.808`; max `98.6%` | `0.755 / 0.819 / 1.406`; max `171.5%`; present on all 593 violating reads | J3 was responsible on 590 reads; its source windows associated with 594 event IDs, including 11 ambiguous reads |
| J4 | `0.820 N·m·s` | `0.424 / 0.436 / 0.448`; max `54.6%` | `0.421 / 0.451 / 0.886`; max `108.1%`; 3 violating reads, 2 associated events | Never the max-margin responsible joint |
| J5 | `0.820 N·m·s` | `0.302 / 0.305 / 0.310`; max `37.8%` | `0.300 / 0.311 / 0.728`; max `88.8%`; 0 violating reads | Nonbinding |
| J6 | `0.820 N·m·s` | `0.012 / 0.015 / 0.016`; max `2.0%` | `0.017 / 0.029 / 0.053`; max `6.4%`; 0 violating reads | Nonbinding |
| All joints | provisional project-boundary binding | 0 violating reads/windows/events; J3 came within `1.44%` of cap | 593/98,304 violating reads (`0.603%`), 587 unique violating segments (`4.775%`), 594/12,464 candidate-associated physical-event IDs (`4.77%`) | 585/593 violating reads were hard-terminal; impulse wins at least `591/593` for every positive candidate |
| Contact duration | 500 Hz physical contact | 64/64 contacts were right-censored 22 ms prefixes | Completed contacts `22 / 24 / 60 ms`; all associated observed prefixes `22 / 30 / 68 ms` | 10,983/12,464 contacts were censored; associated: 17 completed, 577 censored; unambiguous one-read: 9 completed, 571 censored |
| Candidate `imp_max_p` | `0.05` | No provisional-cap activation | active-read delta_impulse mean/max `0.00755 / 0.05`; combined max `0.5` | wins `591/593`, velocity masks 2; observed-prefix `P_event=0.00308 / 0.0273 / 0.05` (completed `0.01865 / 0.05 / 0.05`, N=9) |
| Candidate `imp_max_p` | `0.10` | No provisional-cap activation | active-read delta_impulse mean/max `0.01511 / 0.10`; combined max `0.5` | wins `592/593`, velocity masks 1; observed-prefix `P_event=0.00616 / 0.0546 / 0.10` (completed `0.03730 / 0.10 / 0.10`, N=9) |
| Candidate `imp_max_p` | `0.20` | No provisional-cap activation | active-read delta_impulse mean/max `0.03021 / 0.20`; combined max `0.5` | wins `592/593`, velocity masks 1; observed-prefix `P_event=0.01232 / 0.1092 / 0.20` (completed `0.07459 / 0.20 / 0.20`, N=9) |
| Candidate `imp_max_p` | `0.30` | No provisional-cap activation | active-read delta_impulse mean/max `0.04532 / 0.30`; combined max `0.5` | wins `592/593`, velocity masks 1; observed-prefix `P_event=0.01848 / 0.1639 / 0.30` (completed `0.11189 / 0.30 / 0.30`, N=9) |
| Candidate `imp_max_p` | `0.50` | No provisional-cap activation | active-read delta_impulse mean/max `0.07553 / 0.50`; combined max `0.5` | wins `592/593`, ties 1; observed-prefix `P_event=0.03080 / 0.2731 / 0.50` (completed `0.18648 / 0.50 / 0.50`, N=9) |

The per-joint violating-read counts sum to more than 593 because one controller read can violate
multiple joints. Likewise, a 50 ms Lambda source window can overlap multiple short physical contact
events; the 594 event IDs are therefore candidate associations, not 594 unambiguous causal
assignments. The combined maximum remains `0.5` for every candidate because the velocity arm reaches
its own `max_p=0.5` elsewhere. The winner counts show why the separate telemetry matters: on the
impulse-violating reads, impulse wins almost every soft-OR comparison even though the aggregate
maximum alone cannot reveal that.

Every violating read included J3: 587 were J3-only, three were J1+J3, and three were
J1+J3+J4. The latter co-violations were confined to the two ambiguous four-read windows. Maximum
positive margins were `0.4058`, `0`, `0.5859`, `0.0662`, `0`, and `0 N·m·s` for J1--J6. On the 593
provisional-cap violating reads, counterfactual delta_velocity was zero on 591, `0.09190` on one,
and `0.5` on one (mean `0.000998`), which directly explains the impulse winner share.

The observed trace contained one 50 Hz activation read in 585/587 activation windows and four reads
in two windows. That prevalence is **conditioned by task success termination**, not evidence that a
full physical contact intrinsically produces one read: 585/593 violating reads occurred on terminal
steps, and 571/580 unambiguous one-read event associations were right-censored. The two four-read
windows each crossed two physical contacts, so their compounded maxima—for example,
`1-(1-0.2)^4=0.5904` and `1-(1-0.5)^4=0.9375`—are activation-window exposure, not clean single-contact
dose. Impulse did win all 580 unambiguous observed-prefix associations at every positive candidate.

## Diagnostic-only `0.9` boundary

The uniform `0.9` vector `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738]` was used only to prove
plumbing. In the fixed population it produced 60/64 J3 violations, each associated with exactly one
50 Hz read and one censored 22 ms contact prefix. At counterfactual `imp_max_p=0.5`, impulse won all
60 soft-OR comparisons and gave `P_event` median/p95/max `0.1396/0.2942/0.5`.

The separately authorized diagnostic training pair also completed without retry:

- control `41272362_0`, `imp_max_p=0`, 3m55s;
- target `41272362_1`, `imp_max_p=0.5`, 3m54s;
- both used seed 2, 4,096 x 24, 50 PPO iterations, the same diagnostic vector, and differed only in
  impulse dose;
- these were matched scratch runs (4,915,200 samples per arm), not resumes from frozen
  `model_499.pt`;
- both produced finite `model_0.pt` and `model_49.pt`; both stderr logs were empty.

This pair proves that the active impulse configuration runs through the existing CatPPO pipeline. It
does not make the diagnostic vector a hardware/scientific boundary, and 50 iterations from one seed
cannot establish performance or learned safety. The final-ten aggregate `cat_delta_peak` was
descriptively `0.3454` for control and `0.4119` for target, but those policies had already diverged;
that difference is not treated as an impulse-only causal estimate.

Soft CaT does not physically clamp Lambda at the boundary. Once Lambda exceeds the chosen threshold,
it applies a graded soft-termination probability through the existing reward/discount and GAE path;
whether subsequent learning reduces Lambda is a separate matched-training question.

## Dose analysis and calibration revision

**No `imp_max_p` value is selected.** `0.20` remains an analytical starting point only: it is plainly
visible per read and impulse wins `592/593` active soft-OR comparisons, but the pooled
observed-prefix p95 (`10.92%`) is dominated by censored task episodes. The completed and unambiguous
subset contains only nine contacts; at `0.20` its median/p95/max pressure is
`7.46%/20%/20%`. That sample is too small and termination-selected to calibrate a scientific dose.

The smallest calibration revision is another **no-learning** frozen-policy measurement in which
success termination is deferred only long enough to observe physical release and flush the existing
50 ms Lambda window. Physics, actions, rewards, reference, caps, VIC gains, and CaT equations remain
unchanged, and the continuation is diagnostic rather than a training treatment. Recompute
per-contact read counts and `P_event` from completed contacts, then choose the target dose. Until
that measurement exists, no provisional-boundary training canary is ready to propose or launch.

The activation question itself is answered: the provisional cap binds, the impulse arm is not
masked, and the response is graded. The unresolved question is the strength of the complete-event
response, which is deliberately kept separate.

## Validation and claim boundary

- Telemetry/survey/launcher task suite before launch: 302 passed. Fresh post-result changed-area
  verification covered 1,016 unique tests in split commands: 215 + 532 + 235 + 34 passed.
- Mandatory gates passed: reward validation phases A-M, contact-sensor verification, and random-policy
  reward-setup verification.
- A live 2-environment x 8-step frozen-policy smoke exercised both survey summaries and confirmed
  shape/finiteness, `imp_max_p=0` exact zero, exact max-combination, and diagnostic activation.
- Survey and both diagnostic canary jobs completed `0:0`; all stderr files are empty; raw and compact
  evidence are SHA-256 bound.
- No p-values or independent-seed effect-size tests are claimed. The fixed worlds, controller reads,
  contact events, and rollout segments are not treated as independent training replicates.

Statistical-fallacy coverage is 11/11. Simpson/ecological risks are controlled by reporting fixed and
sampled populations separately and retaining the correct unit of analysis. Censoring/survivorship is
explicit: pooled observed-prefix exposure is labelled as such and is not used to select a
complete-event dose. Look-elsewhere/forking-path risk is bounded by reporting all five offline doses
from the same trace.
No causal training claim is made from the one-seed diagnostic pair; Berkson, collider, base-rate,
regression-to-mean, correlation/causation, and reverse-causality patterns are not applicable to the
reported mechanism invariants.

## Artifacts

- Compact repository summary:
  `docs/results/assets/2026-08-15_z1_impulse_cat_step1/summary_compact.json`
- Compact diagnostic-canary summary:
  `docs/results/assets/2026-08-15_z1_impulse_cat_step1/diagnostic_canary_compact.json`
- Evidence manifest:
  `docs/results/assets/2026-08-15_z1_impulse_cat_step1/EVIDENCE_SHA256SUMS`
- Vega raw survey root:
  `/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-diag90-canary/surveys/8e61caf9618aa27ff403f94ffef091ab962c6dd7/frozen-seed2-v1`
- Vega diagnostic run root:
  `/ceph/hpc/home/eunikhilr/campaigns/z1-vic-impulse-diag90-canary/runs/8e61caf9618aa27ff403f94ffef091ab962c6dd7/b58ccd2f81fd246f27c1e8d88cf86484cd888703`

The 51 MB raw `summary.json` and NPZ traces stay on Vega; the repository stores their hashes and a
48 KB array-free summary rather than duplicating trace-level data.

## Reproduce

Inside the exact clean Vega worktree and pinned Python environment, the survey payload is:

```bash
$PY scripts/impulse_cat_activation_survey.py \
  --checkpoint docs/results/assets/2026-08-14_z1_vic_seed2_canary/model_499.pt \
  --output-dir "$SURVEY_OUTPUT" \
  --device cuda:0
```

The guarded diagnostic pair is the committed zero-argument array launcher:

```bash
sbatch \
  --export=ALL,RUN_ROOT="$RUN_ROOT",ASSET_REPO="$ASSET_REPO",EXPECTED_CODE_REVISION=8e61caf9618aa27ff403f94ffef091ab962c6dd7,EXPECTED_ASSET_REVISION=b58ccd2f81fd246f27c1e8d88cf86484cd888703 \
  scripts/slurm/vega_vic_impulse_diag90_canary.sbatch
```

Both routes refuse output-directory reuse; do not use these commands to retry the banked jobs.

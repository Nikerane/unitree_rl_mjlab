# First-strike quality-conditioned reward experiment

**Date:** 2026-07-26
**Status:** approved design; implementation and GPU launch remain gated
**Scope:** fixed impedance only; four-arm matched-seed reward experiment plus a CPU reference-recipe probe

## 1. Decision this experiment must support

The previous 4×8 campaign showed that the physics-rate first-event package
produces reliable, fast strikes, but it also selects a repeatable fore-aft
onset offset. The existing data contains fast, well-aligned strikes, so the
plant can produce both properties; the policy does not select them frequently.

This experiment must distinguish:

1. whether the raw first-event speed payout has a large marginal effect on
   contact quality;
2. whether the delivered-impulse payout creates useful nail work or mainly
   selects longer dwell/recontact;
3. whether conditioning impact utility on physical contact quality can preserve
   speed, task success, and delivered impulse while selecting better contact;
4. whether the observed effect remains only a simulator result because the
   learned trajectories exceed the Z1 joint-velocity rail.

It does **not** test impulse enforcement, variable impedance, a new
reference-guided training architecture, or hardware transfer. Reference
recipes are characterized on CPU only unless the explicit promotion gates pass.

## 2. Evidence that constrains the design

- F and E contact approximately 12.5 mm behind the nail-axis center in the
  signed fore-aft direction; C is approximately centered.
- C is globally curved but substantially better centered than F and E.
  Therefore path length, global straightness, and lateral excursion are not
  valid primary reward targets.
- Existing F/E episodes already contain centered strikes near 1.8 m/s.
  Accurate, fast fixed-impedance behavior is therefore feasible in the current
  simulator.
- In F, the weighted discounted speed payout is about 87% of the explicit
  first-event maximize stream. Both the speed and delivered-impulse readers are
  center-blind.
- The previous arms are matched by training-seed identity and evaluator random
  stream. The required seed-level Mann–Whitney analysis remains part of the
  evaluation protocol, but paired exact inference must also be reported.
- Nearly every previous episode exceeded the manufacturer joint-velocity rail.
  Reward conclusions remain simulator-only until independently reproduced
  under a qualified controller.

The design follows task-level precedents that condition fast performance on
accuracy or valid first contact:

- Büchler et al. couple landing accuracy and outgoing speed in a table-tennis
  smash reward: <https://arxiv.org/pdf/2006.05935>.
- GIFT evaluates hammer task performance at first tool contact:
  <https://roboticsproceedings.org/rss17/p060.pdf>.
- Potential-based path shaping can preserve an existing optimum but cannot
  repair a base objective that prefers the wrong terminal behavior:
  <https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>.

## 3. Overall experiment

Run four concurrent arms with eight fresh matched training seeds per arm:

| Arm | First-event speed | Delivered impulse | Contact-quality conditioning |
|---|---:|---:|---|
| **F8** | current linear reader, weight 8 | current linear reader, weight 2 | none |
| **F0** | disabled, weight 0 | current linear reader, weight 2 | none |
| **D0** | current linear reader, weight 8 | disabled, weight 0 | none |
| **FQ** | bounded reader, nominal weight 8 | bounded reader, nominal weight 2 | shared bounded quality factor |

Use training seeds 8–15 in every arm. All arms use the same training budget,
fixed gains, action space, task rewards, tracker timing, termination semantics,
checkpoint-selection rule, and evaluation random streams.
Freeze campaign ID `fq4x8` and run shorts `f8`, `f0`, `d0`, and `fq`.

`IMP_J_LIMIT` remains unchanged. `imp_max_p` remains zero. The impulse
constraint is log-only. There is no `set_gains`, commanded stiffness, variable
impedance, superlinear excess reward, path imitation, clock-time velocity
tracking, or global straightness penalty.

## 4. Physics-rate contact-quality snapshot

The shared first-strike tracker will maintain one immutable event record for the
first accepted event, populated at two explicitly tested physics phases.

At **onset**, before any future event-window quantity is available, it latches:

- hammer-face pose;
- nail-head pose and nail axis;
- signed fore-aft and lateral center offsets in the nail plane;
- actual contact point and contact normal when available;
- precontact axial speed;
- first-contact time.

At **finalization**, after success or the fixed event-window boundary, it
latches:

- productivity and finalization reason;
- axial and transverse object-side event impulse;
- delivered nail impulse;
- per-joint event impulse;
- peak joint velocity over the event.

Onset fields never change during finalization, and finalized fields never
change afterward. Reward payout remains one-shot at the control boundary after
finalization. The tracker may observe these quantities but must not change
physics.

Contact geometry comes from a dedicated diagnostic `ContactSensor`, leaving the
existing task sensor shape unchanged. It requests `found`, contact-frame
`force`, world-frame `pos`, and world-frame `normal` with multiple retained
slots. The slot count is qualified on the CPU scripted reference at
`8 -> 16 -> 64`: freeze the smallest count with no overflow whose centroid
agrees with the next larger count to `1e-6 m`. If 64 slots overflow or adjacent
qualified counts disagree, the experiment stops rather than approximating.

Contact-quality instrumentation is orthogonal to reward treatment. A frozen
`quality_instrumentation` mode adds only the passive sensor and diagnostic
tracker fields; it must not alter observations, actions, rewards, terminations,
or physics. It is enabled for all F8/F0/D0/FQ action-tape identity audits and
strict sampled evaluations, even if F8/F0/D0 omit it during training; FQ
requires it during training because its reward reads the score. Training and
evaluation configuration hashes are banked separately, and the action-tape
identity gate must prove that adding the instrumentation leaves every physical
channel unchanged.

The nail plane is compiled once from the runtime nail-head geometry: its
world-frame pose defines the plane origin and axis, and the compiled geom size
defines the radius. No site-center proxy or hard-coded world axis substitutes
for this frame.

## 5. Contact-quality score

The current 12 mm hammer-site-center distance remains a diagnostic only. It
will not become the training gate merely because it separates previous arms.

The primary quality signal is the **actual first-contact point** expressed in
the frozen nail-head plane. For the accepted hammer–nail contacts in the onset
physics substep, compute the normal-impulse-weighted contact centroid
\(p_c\). Let \(p_n\) be the nail axis at the nail-head plane, \(a\) the unit
nail axis, and \(r_n\) the compiled nail-head radius:

\[
e_c = \left\|(I-aa^\top)(p_c-p_n)\right\|,
\qquad
q_{\mathrm{contact}} =
\operatorname{clip}\left(1-\left(e_c/r_n\right)^2,0,1\right).
\]

This directly measures where the contact load enters the nail. It does not
assume the hammer-face site center is the contact point, and it avoids an
expensive online mesh-overlap calculation.

If multiple relevant contacts are present, weights come from their nonnegative
normal impulses. If their total normal impulse is zero, the quality snapshot is
invalid and pays zero; implementation must never depend on arbitrary MuJoCo
contact-array order.

Before training, the score must pass all of the following offline gates:

1. reproduce exactly from locally banked traces, or from a deterministic
   provenance-bound replay if the old payload omitted contact positions;
2. use the compiled nail-head geometry as the sole source of \(r_n\);
3. be invariant to world-frame translation and rotation about the nail axis;
4. remain finite and within [0, 1] for contact, no-contact, multiple-contact,
   and degenerate cases;
5. improve monotonically as a synthetic contact point moves from the nail edge
   toward the axis;
6. show stable seed-level behavior under the existing reset distribution;
7. not be selected or parameterized to maximize separation among C/D′/F/E;
8. be inspected alongside hammer-site offset, contact-normal axiality, and
   transverse impulse, so central but glancing contact is not mislabeled as
   proven physically clean contact.

If the contact point or its impulse weight cannot be reconstructed reliably,
the experiment pauses. There is no fallback to the 12 mm site-center indicator.
No physical contact earns zero quality, but instrumentation overflow or
nonfinite contact geometry invalidates the evaluation row; it is not silently
converted into a legitimate zero-quality episode. During training, the same
condition fails closed to zero payout and increments an explicit sentinel.

The primary endpoint is each seed's all-episode mean
`first_contact_quality_sampled`, with no-contact and zero-positive-normal-force
physical episodes scored zero. Instrumentation-invalid episodes invalidate
their evaluation row as defined above. Report central-half-radius rate
(`e_c <= 0.5 r_n`), full-head rate (`e_c <= r_n`), contact-conditional mean
quality, and the continuous \(e_c\) distribution as supporting diagnostics.
These are task-quality measures, not physical harm thresholds.

The provisional `I_REF_FIRST_STRIKE_SUCCESS=0.3088 N·s` must also pass its
existing production-tracker provenance/reproducibility gate before launch. If
it is recalibrated, the frozen replacement is used identically in F8, F0, D0,
and FQ; the campaign must not mix normalizers across arms.

## 6. FQ reward

For a productive first event, define bounded components

\[
\phi_v = \operatorname{clip}(v_{\mathrm{pre}}/V_{\mathrm{FQ}},0,1),
\qquad
\phi_I = \operatorname{clip}(I_{\mathrm{delivered}}/I_{\mathrm{ref}},0,1).
\]

Here `V_FQ=1.4598331451416016 m/s` is an FQ-only empirical scale/knee, not a
retuned success target. It is derived from the raw bank SHA-256
`ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8`, resolved
manifest SHA-256
`69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f`, and
qualification SHA-256
`641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b`. Compute
the q90 with NumPy `method="higher"`; reproduce 26/256 calibration saturation
and 33/128 group-held-out validation saturation exactly. Missing provenance,
an incompatible quantile method, or either failed reproduction fails closed.
F8/D0 retain their existing `1.0 m/s` scale, and `I_REF=0.3088 N·s` is not
modified.

FQ pays once:

\[
r_{\mathrm{FQ}} =
q_{\mathrm{contact}}
\left(8\phi_v + 2\phi_I\right).
\]

Both components are bounded so extreme speed or impulse cannot compensate for
poor contact. No reward is paid for no-contact, nonproductive contact, or an
event rejected by the existing first-strike semantics.

The unit-test suite must explicitly reject an additive implementation such as
`q_contact + 8*phi_v + 2*phi_I`, because it would still pay a large maximize
reward for poor contact.

F8 and F0 keep the current linear delivered reader so the F8/F0 comparison
isolates the marginal raw speed payout. D0 keeps the current linear speed
reader so F8/D0 isolates the marginal delivered-impulse payout. FQ is a
remedy-package test, not isolated quality multiplication or a one-variable
ablation. Only after this campaign, preregister the bounded center-blind `FB`
follow-up (the bounded reader package without the quality factor); do not add
that arm to this 4×8 matrix.

## 7. Offline evidence and visual audit before GPU

Bank locally, with checksums:

- all raw trace payloads;
- the archived C/D′/F/E resolved accepted-attempt manifest;
- seed summaries and evaluation configuration;
- the exact asset revision needed for contact-point reconstruction.

Generate fixed-scale static figures in the style already preferred for the
trajectory grid:

1. x–z medoid trajectories for every arm and seed;
2. signed x–y onset distributions with nail and hammer footprints;
3. quality-versus-speed and quality-versus-delivered-impulse frontiers;
4. contact-aligned axial/transverse force and cumulative impulse traces;
5. first-contact-time and recontact distributions.

Trajectory examples must be predeclared seed-level medoids, not
lexicographically first episodes. Interactive HTML may supplement, but never
replace, the static evidence.

The medoid distance uses standardized Euclidean distance over
`(quality, useful_speed, contact_time, delivered_impulse, signed_x, signed_y)`.
Within a seed, each scale is its finite standard deviation, replaced by `1.0`
when zero. Exact ties break by `(env_id, episode_ordinal)`.

Counterfactually rescore the archived C/D′/F/E episodes under `q_contact`. Reject
the proposed FQ score if either preregistered check fails:

1. within each archived arm, the top quartile by counterfactual FQ has lower
   mean physical contact quality than the bottom quartile; or
2. the absolute seed-level Spearman correlation between counterfactual FQ and
   physical contact quality is below `0.50` while its absolute correlation with
   useful speed exceeds the quality correlation.

These are falsification checks, not thresholds tuned to separate the archived
arms.

### 7.1 Human-review video library

For every accepted arm/seed, render the exact medoid episode used by the static
trajectory grid. Each entry binds the MP4 and state trace by episode identity,
trace digest, explicit frame-to-substep timing, and SHA-256. Display the video
beside its x-z/x-y path, contact close-up, contact-aligned force/impulse trace,
and frozen metrics. Also render one deterministic mean-policy rollout per
checkpoint as a clearly labelled secondary diagnostic.

The gallery provides separate human annotations (`good`, `questionable`,
`bad`, plus free-text reasons). These annotations are qualitative diagnostics:
they cannot change the preregistered replacement rule or select a different
episode after results are visible. Large MP4 files remain outside Git; the
manifest, checksums, montages, state overlays, and annotation export are
banked.

Simulator state remains quantitative ground truth. Pixel-space computer vision
may check video/overlay consistency and extract keyframes, but it does not
replace the exact simulator trajectory or contact measurements.

### 7.2 Paired CPU reference-recipe probe

Reference design is evaluated independently before adding another training
arm. Replay three nail-frame-relative recipes over the same 32 randomized reset
states:

1. **R0 — shipped centered polyline:** the current `SingleStrikeReference`;
2. **R1 — learned impact-only arc:** the transverse template from frozen
   `traj_dc`, with R0's axial schedule;
3. **R2 — terminally repaired arc:** identical to R1 until 90 mm axial
   standoff, then smoothly blend transverse offset to zero by 20 mm standoff.

This is 96 paired CPU rollouts. It asks whether the reward-induced arc provides
any physical benefit and whether terminal alignment can be repaired without
deleting the early wind-up. The template must be anchored to the live reset
head and live nail frame; world-coordinate replay is invalid.

The R1 source set is exactly `traj_dc`, frozen at SHA-256
`b22dabb94a10a1e7f68f3fe6a4a2f9412e14916a40a3dbafc81e2f3c3cc7f89f`; no other
trace may substitute. These source traces lack pre-apex motion, so the
pre-apex transverse residual is identically zero and the apex is repeated at
the wind-up/descent boundary rather than inferred. Resample the observed
post-apex descent on 51 uniformly spaced points and store the nail-frame
transverse residual between the realized trace and the R0 centered path built
from that source trace's own live head and nail poses. Select the residual
template minimizing summed pairwise transverse L2 distance; exact ties break by
lexicographic trace digest. At replay, rotate that residual through the same
nail-frame basis used by the signed-offset diagnostic and add it to the live
R0 target at the corresponding wind-up/descent progress. This anchors the
target to both the live reset head and live nail while changing only the
transverse template; R0's axial schedule and pacing remain unchanged. For R2,
let \(d\) be axial standoff and
\(u=\operatorname{clip}((0.09-d)/0.07,0,1)\). Its transverse offset is R1's
offset multiplied by the quintic minimum-jerk factor
\(1-(10u^3-15u^4+6u^5)\), making it identical to R1 through 90 mm and zero with
continuous first and second derivatives by 20 mm.

Before any recipe outcome is inspected, generate a fresh 32-row digested reset
manifest using reset seeds `0..31`; do not reuse a prior manifest. Each row
records the realized initial robot/nail state, live head and nail poses,
nail-frame basis, code/config/asset revisions, and a canonical state digest.
All three recipes must reproduce each row's digest.
After the nominal script, issue no more than two final-target control steps,
stopping earlier on success; there is no further endpoint hold. A rollout that
has not produced both a productive first event and task success by the
`n_script + 2` boundary is a press-through failure.

R0 first serves as the calibration baseline: it must reproduce the Phase M
in-script productive-contact and success contract and its 32-reset physical
baseline is banked. Apply the following absolute gates to any R1 or R2 recipe
proposed for promotion, not automatically to the diagnostic comparator:
at least 30/32 productive in-script first events, at least 29/32 accepted
onsets inside the provisional 12 mm center proxy, median precontact speed at
least 95% of R0, every accepted event above 0.5 m/s, median raw delivered
impulse at least 90% of R0, no prefix joint speed above 3.1415 rad/s, no rollout
above `Lambda/cap=1`, and no numerical/sentinel failure. The R2 repair contrast
additionally requires at least four more within-12-mm onsets than R1
(`>=4/32`, i.e. 12.5 percentage points), whether or not R1 passes an absolute
promotion gate. Replacing R0 additionally requires a positive physical reason:
at least 5% higher precontact speed or at least 10% lower p95 prefix joint
speed. Promotion also requires the preregistered terminal axial-speed,
terminal lateral-speed, and approach-angle gates, measured in the live
nail-frame from the same terminal window; freeze their exact thresholds in the
fresh manifest before any R1/R2 outcome is inspected. Missing, nonfinite, or
failed values fail closed. Delivered impulse alone cannot justify replacement
because it is dwell/force dominated. Global straightness is never a promotion
target.

Before any GPU reference arm, counterfactually score each replay under all
three references. The generating recipe must have the largest cumulative raw
imitation score in at least 26/32 resets and a median self/next-best score ratio
of at least 1.25. Otherwise the weak prior cannot identify the recipes. A
passing candidate is then tested under `solref_scale=2` and rejected if speed,
delivered impulse, or worst-joint impulse changes by 20% or more, or if
success/centering fails.

A CPU pass certifies scripted feasibility and separability under the existing
weak prior only; it is not evidence that PPO will learn the recipe.

No global-straightness reward is introduced, and global straightness is never
a CPU-recipe promotion target. A GPU
`FQ` versus `FQ+terminal-funnel` comparison is launched only if the CPU probe
passes and the first 4x8 result shows a remaining terminal credit-assignment
failure.

## 8. Verification before training

Add unit tests for:

- exact onset phase and immutable snapshot;
- one event, one payout, no rearming;
- reset and partial-reset behavior;
- no-contact and nonproductive-event zero;
- coordinate-frame invariance;
- multiple-contact centroid and zero-normal-impulse edge cases;
- finite bounded `q_contact`, `phi_v`, and `phi_I`;
- monotonic quality behavior;
- no off-quality speed or impulse compensation;
- F8/F0/D0/FQ action-tape physical identity;
- exact allowed configuration differences among arms.

Then run:

```text
pytest tests/test_impact_progress_reward.py \
       tests/test_impulse_bound.py \
       tests/test_impulse_constraint.py \
       tests/test_delivered_impulse_reward.py \
       tests/test_cat_soft_hook.py \
       tests/test_first_strike_quality_reward.py
validate_rewards.py                 # phases A–M
verify_contact_sensor.py
verify_reward_setup.py
playback_reference.py
```

Every gate must pass in the `unitree_mjlab` conda environment before any Vega
training.

## 9. Evaluation and inference

Evaluate every seed with the strict sampled protocol: 256 environments × 2
episodes, using only `*_sampled` columns. Require:

- `impossible_success_n == 0`;
- `lambda_dead_n == 0`;
- exact clean git provenance;
- complete quotas and no numerical/sentinel failures.

### Primary contrasts

1. **F0 versus F8:** causal effect of removing raw first-event speed payout.
2. **D0 versus F8:** causal effect of removing raw delivered-impulse payout.
3. **FQ versus F8:** effect of replacing center-blind maximize payout with
   bounded quality-conditioned utility.

FQ versus F0 and FQ versus D0 are secondary.

For the D0-versus-F8 mechanism question, preregister three secondary
first-event endpoints:

- event-window nail-depth gain,
  `max(peak_depth - depth_at_contact, 0)`;
- contact dwell in milliseconds, defined as raw-contact substeps inside the
  accepted event window times `physics_dt`;
- recontact count, defined as off-to-on transitions after the accepted onset
  and before finalization.

Report their eight paired seed differences and paired-bootstrap intervals,
alongside raw delivered impulse and first-window/overall success. These
mechanism endpoints do not enlarge either three-test Holm family. Describe the
delivered reader as mainly selecting dwell/recontact only if the one-sided 95%
paired-bootstrap upper bound for the D0-minus-F8 delivered-impulse difference
is below zero, the corresponding upper bound is below zero for either dwell or
recontact count, the lower bound for the event-window-depth-gain ratio is
greater than `0.90`, and the existing success guardrails pass. A nonpositive or
nonfinite F8 depth-gain denominator makes that mechanism claim not
distinguishable.

For each primary contrast:

- report the required two-sided seed-level Mann–Whitney test;
- report an exact paired sign-flip test on matched-seed differences;
- use Holm correction across the three primary Mann–Whitney tests and,
  separately, across the three paired sign-flip tests;
- report paired seed-bootstrap confidence intervals and all eight differences;
- do not describe the tests as independent confirmations.

Bootstrap matched seed blocks with replacement for 100,000 resamples using
NumPy `PCG64` seed `20260726`. Ratio bounds use the resampled arm means; a
nonpositive or nonfinite denominator is an automatic gate failure. Every
decision boundary is strict.

### Decision rules

An arm replaces F8 only if all conditions pass:

1. mean `first_contact_quality_sampled` improves by at least 0.10, with
   corrected Mann–Whitney `p < 0.05` and corrected exact paired sign-flip
   `p < 0.05`;
2. the one-sided 95% paired-bootstrap lower bound for useful-speed ratio is
   greater than 0.95;
3. first-window and overall success remain at least 90% and no more than five
   percentage points below F8;
4. the one-sided 95% lower bound for success-weighted delivered first-event
   impulse ratio is greater than 0.90;
5. all provenance, quota, liveness, numerical, and CaT-log gates pass.

A null F0 result means only that the experiment found no evidence for the
predeclared large marginal effect. It does not prove equivalence or falsify
all reward-level explanations.

A null D0 result has the same limited interpretation for the delivered-impulse
reader. A D0 change in dwell or per-joint impulse without additional nail work
is evidence about contact-regime selection, not proof that delivered impulse
is intrinsically unsafe.

### Hardware interpretation

Joint-velocity legality is always reported. This campaign is explicitly
simulator-only if the manufacturer rail is violated. No arm is called
hardware-ready, and no impulse-enforcement or VIC decision is made, until the
selected reward is revalidated under a separately qualified speed-aware
controller.

## 10. Provenance and execution

Use named-file commits only. Deploy through commit, push, and a clean Vega
checkout/pull; never copy tracked files with `scp`. Any `-dirty` evaluation row
is invalid. Use one GPU per training run.

Freeze a 32-row accepted-training/checkpoint manifest before strict evaluation.
It binds arm/seed, retained checkpoint, training attempt and retry history,
code/config/asset revisions, and artifact hash. After infrastructure-only
evaluation retries are resolved, freeze a separate 32-row accepted-evaluation
manifest binding each checkpoint to its evaluation attempt, RNG stream,
payload hash, provenance, and dirty/sentinel state. Analysis starts only after
the second manifest is complete.

GPU launch is authorized only after the offline quality-score audit,
counterfactual rescoring, unit tests, standard gates, preregistration, and code
review all pass.

## 11. Interpretation matrix

| Outcome | Interpretation | Next action |
|---|---|---|
| F0 improves quality and passes guardrails | Raw speed payout has a large harmful marginal effect | Adopt F0 provisionally; compare with FQ |
| F0 improves quality but loses speed/impulse | Speed proxy is useful but mis-specified | Prefer FQ if FQ passes |
| D0 shortens dwell/lowers Lambda without losing task work | Delivered-impulse payout was selecting press/recontact | Keep it disabled or bounded/quality-conditioned |
| D0 loses useful depth/success | Delivered-impulse payout contributes task work | Retain it only in bounded quality-conditioned form |
| FQ improves quality and passes guardrails | Quality-conditioned impact is a viable fixed-impedance objective | Revalidate with speed-aware controller |
| Neither improves quality | Center-blind speed alone is not the dominant cause | Factor one-shot credit, reader semantics, and normalizer separately |
| Quality improves but qvel remains illegal | Reward diagnosis succeeded only in simulation | Do not infer hardware readiness |
| `q_contact` fails offline validation | The proposed terminal proxy is not trustworthy | Stop; improve contact instrumentation before training |

## 12. Explicitly deferred

- global trajectory reference or straight-line tracking;
- GPU late ante-impact funnel shaping unless the CPU recipe and FQ
  credit-assignment gates both pass;
- contact-force reward;
- accuracy as a new constrained-RL mechanism;
- changes to impulse caps or accumulation-window semantics;
- `imp_max_p > 0`;
- variable impedance or `set_gains`;
- hardware conclusions.

These require separate evidence and decisions.

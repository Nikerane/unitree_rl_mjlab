# First-strike quality-conditioned reward experiment

**Date:** 2026-07-26
**Status:** approved design; implementation and GPU launch remain gated
**Scope:** fixed impedance only; three-arm matched-seed reward experiment

## 1. Decision this experiment must support

The previous 4×8 campaign showed that the physics-rate first-event package
produces reliable, fast strikes, but it also selects a repeatable fore-aft
onset offset. The existing data contains fast, well-aligned strikes, so the
plant can produce both properties; the policy does not select them frequently.

This experiment must distinguish:

1. whether the raw first-event speed payout has a large marginal effect on
   contact quality;
2. whether conditioning impact utility on physical contact quality can preserve
   speed, task success, and delivered impulse while selecting better contact;
3. whether the observed effect remains only a simulator result because the
   learned trajectories exceed the Z1 joint-velocity rail.

It does **not** test impulse enforcement, variable impedance, a new trajectory
architecture, or hardware transfer.

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

Run three concurrent arms with eight fresh matched training seeds per arm:

| Arm | First-event speed | Delivered impulse | Contact-quality conditioning |
|---|---:|---:|---|
| **F8** | current linear reader, weight 8 | current linear reader, weight 2 | none |
| **F0** | disabled, weight 0 | current linear reader, weight 2 | none |
| **FQ** | bounded reader, nominal weight 8 | bounded reader, nominal weight 2 | shared bounded quality factor |

Use training seeds 8–15 in every arm. All arms use the same training budget,
fixed gains, action space, task rewards, tracker timing, termination semantics,
checkpoint-selection rule, and evaluation random streams.

`IMP_J_LIMIT` remains unchanged. `imp_max_p` remains zero. The impulse
constraint is log-only. There is no `set_gains`, commanded stiffness, variable
impedance, superlinear excess reward, path imitation, clock-time velocity
tracking, or global straightness penalty.

## 4. Physics-rate contact-quality snapshot

The shared first-strike tracker will latch one immutable pre-contact/onset
snapshot for the first accepted productive event:

- hammer-face pose;
- nail-head pose and nail axis;
- signed fore-aft and lateral center offsets in the nail plane;
- actual contact point and contact normal when available;
- precontact axial speed;
- axial and transverse object-side event impulse;
- first-contact time;
- delivered nail impulse;
- per-joint event impulse;
- peak joint velocity.

Every field must be captured at an explicitly tested physics phase. Reward
payout remains one-shot at the control boundary. The tracker may observe these
quantities but must not change physics.

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

The primary endpoint is each seed's all-episode mean
`first_contact_quality_sampled`, with no-contact and invalid-contact episodes
scored zero. Report central-half-radius rate (`e_c <= 0.5 r_n`), full-head rate
(`e_c <= r_n`), contact-conditional mean quality, and the continuous \(e_c\)
distribution as supporting diagnostics. These are task-quality measures, not
physical harm thresholds.

The provisional `I_REF_FIRST_STRIKE_SUCCESS=0.3088 N·s` must also pass its
existing production-tracker provenance/reproducibility gate before launch. If
it is recalibrated, the frozen replacement is used identically in F8, F0, and
FQ; the campaign must not mix normalizers across arms.

## 6. FQ reward

For a productive first event, define bounded components

\[
\phi_v = \operatorname{clip}(v_{\mathrm{pre}}/1.0\ {\rm m\,s^{-1}},0,1),
\qquad
\phi_I = \operatorname{clip}(I_{\mathrm{delivered}}/I_{\mathrm{ref}},0,1).
\]

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
isolates the marginal raw speed payout. FQ is a remedy arm, not a one-variable
ablation.

## 7. Offline evidence and visual audit before GPU

Bank locally, with checksums:

- all raw trace payloads;
- the resolved accepted-attempt manifest;
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

Counterfactually rescore the archived C/D′/F/E episodes under `q_contact`. Reject
the proposed FQ score if higher counterfactual reward still systematically
selects worse contact-point quality or if the score is effectively only a
speed proxy.

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
- F8/F0/FQ action-tape physical identity;
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
2. **FQ versus F8:** effect of replacing center-blind maximize payout with
   bounded quality-conditioned utility.

FQ versus F0 is secondary.

For each primary contrast:

- report the required two-sided seed-level Mann–Whitney test;
- report an exact paired sign-flip test on matched-seed differences;
- use Holm correction across the two primary Mann–Whitney tests and,
  separately, across the two paired sign-flip tests;
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

GPU launch is authorized only after the offline quality-score audit,
counterfactual rescoring, unit tests, standard gates, preregistration, and code
review all pass.

## 11. Interpretation matrix

| Outcome | Interpretation | Next action |
|---|---|---|
| F0 improves quality and passes guardrails | Raw speed payout has a large harmful marginal effect | Adopt F0 provisionally; compare with FQ |
| F0 improves quality but loses speed/impulse | Speed proxy is useful but mis-specified | Prefer FQ if FQ passes |
| FQ improves quality and passes guardrails | Quality-conditioned impact is a viable fixed-impedance objective | Revalidate with speed-aware controller |
| Neither improves quality | Center-blind speed alone is not the dominant cause | Factor one-shot credit, reader semantics, and normalizer separately |
| Quality improves but qvel remains illegal | Reward diagnosis succeeded only in simulation | Do not infer hardware readiness |
| `q_contact` fails offline validation | The proposed terminal proxy is not trustworthy | Stop; improve contact instrumentation before training |

## 12. Explicitly deferred

- global trajectory reference or straight-line tracking;
- late ante-impact funnel shaping;
- contact-force reward;
- accuracy as a new constrained-RL mechanism;
- changes to impulse caps or accumulation-window semantics;
- `imp_max_p > 0`;
- variable impedance or `set_gains`;
- hardware conclusions.

These require separate evidence and decisions.

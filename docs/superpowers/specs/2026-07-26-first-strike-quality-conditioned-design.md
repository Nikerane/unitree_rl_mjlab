# Lean first-strike quality-conditioned reward experiment

**Date:** 2026-07-26
**Status:** approved lean design; Tasks 1–4 are complete, the FQ-min wiring
amendment and all launch gates remain open
**Scope:** fixed impedance only; one matched four-arm reward experiment

## 1. Decision this experiment must support

The previous fixed-impedance 4×8 campaign produced reliable, fast first
strikes, but the fastest event-reward arms selected a repeatable fore-aft
contact offset. Existing episodes contain strikes that are both fast and
centered, so the simulator can produce both properties; the current objective
does not select them reliably.

This lean experiment asks three causal questions:

1. What is the marginal effect of the raw first-event speed payout?
2. What is the marginal effect of the raw delivered-impulse payout?
3. With delivered impulse disabled, does multiplying a bounded speed payout by
   physical first-contact quality select better contact than paying no explicit
   first-event maximize reward?

It then asks one practical question: can that minimal quality-gated speed
treatment replace F8 without materially losing useful speed, first-window or
overall success, event-window nail work, or the existing safety/provenance
guardrails?

This is not an impulse-enforcement, variable-impedance, reference-guided,
hardware-transfer, or trajectory-shaping experiment. `imp_max_p` remains zero,
so any positive result is a reward result in the fixed-impedance simulator.

## 2. Evidence that constrains the design

- F and E contacted approximately 12.5 mm behind the nail-axis center in the
  signed fore-aft direction; C was approximately centered.
- C was globally curved while being better centered than F and E. Global path
  length, global straightness, and lateral excursion are therefore not valid
  primary reward targets.
- Existing F/E episodes include centered strikes near 1.8 m/s. Accurate, fast
  behavior is feasible in the current simulator.
- In F, the discounted speed payout was about 87% of the explicit first-event
  maximize stream. Both raw speed and raw delivered-impulse readers are
  center-blind.
- Delivered impulse can increase through dwell and recontact without a
  proportional increase in useful event-window nail motion. Its causal role
  must be separated before it is retained in a remedy.
- Nearly every prior episode exceeded the manufacturer joint-velocity rail.
  Finite exceedances remain part of the simulator estimand, but they prevent a
  hardware-speed-qualified interpretation.

The reward form follows task-level precedents that condition fast performance
on accuracy or valid first contact:

- Büchler et al. couple landing accuracy and outgoing speed in a table-tennis
  smash reward: <https://arxiv.org/pdf/2006.05935>.
- GIFT evaluates hammer performance at first tool contact:
  <https://roboticsproceedings.org/rss17/p060.pdf>.
- Potential-based path shaping cannot repair a base objective that prefers the
  wrong terminal behavior:
  <https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>.

## 3. Frozen 4×8 treatment matrix

Run four concurrent arms with training seeds 8–15 in every arm:

| Treatment label | Registered task | Short | First-event speed | Delivered impulse |
|---|---|---|---|---|
| **F8** | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` | `f8` | raw linear, weight 8 | raw linear, weight 2 |
| **F0** | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-F0` | `f0` | disabled, weight 0 | raw linear, weight 2 |
| **D0** | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-D0` | `d0` | raw linear, weight 8 | disabled, weight 0 |
| **FQ-min** | `Unitree-Z1-Hammer-CaT-Impulse-Event-Quality` | `fq` | bounded and quality-gated, weight 8 | disabled, weight 0 |

The compatibility mapping is frozen: manifests and launchers retain campaign
ID `fq4x8`, short `fq`, and the existing `...Event-Quality` task ID, while all
human-facing analysis and results call that treatment **FQ-min**. These names
must not be interpreted as the superseded two-reader FQ package.

For productive first events, the raw readers remain exactly:

\[
r_{\mathrm{speed,raw}} = 8\,v_{\mathrm{pre}}/(1.0\ {\rm m\,s^{-1}}),
\qquad
r_{\mathrm{delivered,raw}} = 2\,I_{\mathrm{delivered}}/(0.3088\ {\rm N\,s}).
\]

They are one-shot, linear, and unbounded above. A zero weight disables a
reader; it does not change the tracker or event definition.

All arms use:

- 500 PPO iterations, 4096 training environments, and `model_499.pt`;
- the same training-seed identity and matched strict-evaluation RNG streams;
- fixed arm gains `Kp/Kd=1000/100`, except joint 2 at `1500/150`, and
  gripper gains `100/20`;
- the same DiffIK action space, task rewards, reset distribution, tracker
  timing, success termination, checkpoint rule, and training budget;
- `IMP_J_LIMIT=[1.640, 3.280, 1.640, 1.640, 1.640, 1.640] N·m·s`;
- `imp_max_p=0.0`, with the impulse constraint log-only;
- no `set_gains`, commanded stiffness, superlinear excess reward, path
  imitation, clock-time velocity tracking, or global straightness penalty.

### 3.1 Preregistered contrasts

The three primary causal contrasts are:

1. **F8 versus F0:** removing the raw speed payout.
2. **F8 versus D0:** removing the raw delivered-impulse payout.
3. **D0 versus FQ-min:** adding only bounded quality-gated speed to the
   delivered-off treatment.

For effect tables, orient differences as `F0-F8`, `D0-F8`, and
`FQ-min-D0`, so a positive contact-quality difference always means the
second treatment listed above improved quality.

**FQ-min versus F8 is a supporting practical comparison.** It determines
whether the minimal remedy can replace the current objective under frozen
practical margins, but it is not a fourth causal test in the Holm family.
FQ-min versus F0 is descriptive only.

## 4. Physics-rate contact-quality record

The shared first-strike tracker keeps one immutable record for the first
accepted event, populated at two tested physics phases.

At onset the tracker latches:

- actual contact point, radial contact error, and contact quality;
- quality validity and slot-overflow state;
- precontact axial speed and first-contact time;
- contact-normal axiality.

At finalization, after success or the fixed event-window boundary, it latches:

- productivity and finalization reason;
- axial and transverse object-side event impulse;
- raw delivered nail impulse;
- depth at contact and peak event-window nail depth.

Onset fields never change at finalization, and finalized fields never change
afterward. Reward payout remains one-shot at the control boundary after
finalization. The tracker observes physics but never changes it.

The schema-v3 artifact separately records raw head/contact trajectories, nail
geometry, qvel, and Lambda channels. Offline analysis expresses the contact
point in the frozen nail-plane basis and derives onset axial/lateral speed,
approach angle, event-window depth gain, dwell, recontact, and qvel/Lambda
summaries from those physics-rate channels. It must not pretend those derived
fields were tracker latches.

Contact geometry comes from the dedicated diagnostic `ContactSensor`. It
requests `found`, contact-frame `force`, world-frame `pos`, and world-frame
`normal` in multiple retained slots. Slot count is qualified on the CPU
scripted reference at `8 -> 16 -> 64`: freeze the smallest count with no
overflow whose centroid agrees with the next larger count to `1e-6 m`. If 64
slots overflow or adjacent qualified counts disagree, stop rather than
approximate.

The passive `quality_instrumentation` mode adds only that sensor and diagnostic
tracker fields. It must not alter policy observations, actions, rewards,
terminations, or physics. FQ-min requires it in training; strict sampled
evaluation enables it for all four arms. Training and evaluation configuration
hashes are banked separately. Fixed-action-tape replay must prove physical
identity with and without the passive instrumentation and across all four
treatment configs.

## 5. Contact-quality score

The 12 mm hammer-site-center distance remains a diagnostic. It is not the
training gate.

For accepted hammer–nail contacts in the onset physics substep, let \(p_c\) be
the nonnegative-normal-force-weighted contact centroid, \(p_n\) the nail-axis
point at the nail-head plane, \(a\) the unit nail axis, and \(r_n\) the compiled
nail-head radius:

\[
e_c = \left\|(I-aa^\top)(p_c-p_n)\right\|,
\qquad
q_{\mathrm{contact}} =
\operatorname{clip}\left(1-\left(e_c/r_n\right)^2,0,1\right).
\]

The runtime nail-head site supplies \(p_n\), the provenance-bound
`nail_slide` axis supplies \(a\), and the compiled `nail_head` geom supplies
\(r_n\). The tracker axis/config, frozen asset geometry, and schema-v3
`nail_geometry` record must agree. No hammer-site-center proxy or
mesh-overlap approximation substitutes for this frame.

If multiple relevant contacts are present, weights are their nonnegative
normal forces. If their total is zero, quality is invalid and pays zero.
Implementation must not depend on MuJoCo contact-array order.

The score must:

1. be invariant to world translation and nail-axis rotation;
2. remain finite and bounded in [0, 1] for contact, no-contact,
   multiple-contact, and degenerate inputs;
3. improve monotonically as a synthetic contact point moves from the nail edge
   toward the axis;
4. use the compiled nail-head radius;
5. fail closed online and increment an explicit sentinel on overflow or
   nonfinite geometry;
6. invalidate the complete strict-evaluation row on instrumentation failure;
7. never be tuned to separate archived C/D-prime/F/E outcomes.

No physical contact, or a physical contact with zero positive normal force,
earns all-episode quality zero. Instrumentation overflow, nonfinite geometry,
or a missing required snapshot is not a legitimate zero; it invalidates the
row.

The primary endpoint is each training seed's all-episode mean
`first_contact_quality_sampled`. Supporting contact endpoints are radial error,
central-half-radius rate, full-head rate, and contact-conditional mean quality.
Contact-normal axiality, lateral onset speed, approach angle, and transverse
impulse prevent a centered but glancing strike from being mislabeled as proven
clean contact.

## 6. FQ-min reward and frozen speed scale

For a productive first event, FQ-min pays exactly:

\[
r_{\mathrm{FQ\text{-}min}} =
8\,q_{\mathrm{contact}}\,
\operatorname{clip}\left(
\frac{v_{\mathrm{pre}}}{1.4598331451416016\ {\rm m\,s^{-1}}},
0,1\right).
\]

No contact, an invalid quality snapshot, a nonproductive event, or a rejected
event earns zero. Extreme speed cannot compensate for poor contact because the
speed factor is capped before multiplication by quality.

`V_FQ=1.4598331451416016 m/s` is an FQ-min-only empirical scale/knee, not a
retuned success threshold. It is derived from:

- raw bank SHA-256
  `ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8`;
- resolved manifest SHA-256
  `69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f`;
- qualification SHA-256
  `641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b`.

Qualification recomputes the q90 with NumPy `method="higher"` and must
reproduce both 26/256 calibration saturations and 33/128 group-held-out
validation saturations exactly. Missing bytes, hash mismatch, another quantile
method, a different scale, or either saturation mismatch fails closed.
F8/F0/D0 retain their raw `1.0 m/s` speed normalizer.

F8 and F0 retain `I_REF_FIRST_STRIKE_SUCCESS=0.3088 N·s` for their raw
delivered reader. Qualification verifies the existing production-tracker
normalizer provenance and exact cross-arm config identity; it does not
recalibrate or mix delivered normalizers during this campaign.

FQ-min has `delivered_impulse.weight=0`. Raw delivered impulse remains recorded
for mechanism description, but it is neither part of the FQ-min reward nor an
FQ-min acceptance gate. The previously implemented
quality-conditioned-delivered reader is outside active scope: it must not be
registered by FQ-min, launched, analyzed as a treatment, or silently restored.
It may remain as unregistered dead code until a separate cleanup, or be removed
if exact config and reward tests prove there is no consumer.

## 7. Lean CPU qualification before GPU

GPU work is authorized only after all of the following pass on the exact launch
commit:

- the FQ-min config/reward amendment and exact allowed-difference tests;
- the frozen `V_FQ` provenance, q90 method, and saturation reproduction;
- contact kernel, onset/finalization, reset, overflow, and one-shot payout
  tests;
- schema-v3 recorder, digest sensitivity, immutable snapshot, and legacy-reader
  compatibility tests;
- fixed-action-tape physical identity across F8/F0/D0/FQ-min and
  instrumented/uninstrumented F8;
- the CPU contact-slot probe with no overflow;
- `validate_rewards.py` phases A–M;
- `verify_contact_sensor.py`;
- `verify_reward_setup.py`;
- `playback_reference.py`;
- one independent code/statistics review.

The archived C/D-prime/F/E policy zoo is not re-evaluated on a GPU for this
qualification. The quality score is validated from pure geometry tests,
schema-v3 fixtures, the provenance-bound local bank used for `V_FQ`, the CPU
reference probe, and treatment-identity replay. This removes a costly
non-causal gate without weakening launch provenance.

## 8. Strict sampled evaluation

Evaluate every accepted checkpoint with 256 environments and exactly the first
two completed stochastic episodes per environment: 512 episodes per seed and
16,384 episodes for the complete campaign. Use only `*_sampled` decision
fields.

The matched evaluator RNG identities remain
`reset=2036072919`, `observation=2046072933`, and `action=2056072941`.

Every row must have:

- schema version 3 and a recomputable content-addressed trace digest;
- the passive quality instrumentation enabled;
- exact code, asset, checkpoint, nail-asset, campaign-config,
  treatment-config, and accepted-manifest hashes;
- complete quotas and matched action/reset/observation RNG identities;
- `impossible_success_n==0`;
- `lambda_dead_n==0`;
- zero quality overflow/nonfinite/missing-snapshot sentinels;
- finite qvel and Lambda channels;
- a clean code and asset worktree.

A failed row invalidates the complete 32-row analysis. Retry only documented
infrastructure failures with identical arm, seed, task, and config. Retain
every attempt. Never replace a completed weak, unstable, or inconvenient seed.

### 8.1 Offline analysis surface

The active analysis reads schema-v3 artifacts and reports:

- all-episode contact quality and radial error;
- nail-frame contact coordinates;
- onset axial speed, lateral speed, contact-normal axiality, and approach
  angle;
- useful first-window speed;
- first-window and overall success;
- event-window nail-depth gain;
- contact dwell and recontact count;
- raw delivered impulse;
- peak qvel, qvel-rail exceedance, per-joint Lambda, and worst
  `Lambda/cap`.

For F8 versus D0, report depth gain, dwell, recontact, raw delivered impulse,
and both success endpoints together. That is the mechanism audit for whether
the delivered reader selected useful nail work or mostly selected contact
duration/recontact.

Generate exactly two static figures:

1. `paired_seed_effects.png` — all eight paired seed effects for the three
   causal contrasts and the supporting FQ-min-versus-F8 practical margins;
2. `aggregate_nail_plane_contact_map.png` — every valid accepted contact in the
   common nail-plane frame, with identical axes and the compiled nail outline.

There is no active HTML report, trajectory grid, per-seed medoid grid,
quality/speed frontier, or force/impulse grid.

## 9. Inference protocol

The training seed is the inferential unit. Episode-level observations are
aggregated within each seed before any test.

For each of the three primary causal contrasts:

- run the repository's exact two-sided Mann–Whitney test across the two sets of
  eight seed summaries;
- Holm-adjust those three Mann–Whitney p-values as the sole confirmatory
  decision family;
- report `U`, tie-aware `A12`, arm means, absolute effects, and all eight
  matched-seed differences;
- report the exact paired sign-flip p-value as a matched-seed sensitivity
  analysis only.

The paired sign-flip result is not a second Holm family, an independent
confirmation, or a duplicate co-gate. A contrast is statistically
distinguishable only through its Holm-adjusted Mann–Whitney result. A null
result means not distinguishable at eight seeds; it does not establish
equivalence.

Bootstrap matched seed blocks with replacement for 100,000 resamples using
NumPy `PCG64` seed `20260726`. Report paired intervals for absolute differences
and one-sided 95% bounds for preregistered ratios. Ratio bounds use resampled
arm means; a nonpositive or nonfinite denominator fails the relevant gate.
Every decision boundary is strict.

Secondary mechanism endpoints do not enlarge the three-test Holm family.
Describe the raw delivered reader as mainly selecting dwell/recontact only if:

1. the one-sided 95% upper bound for `D0-F8` raw delivered impulse is below
   zero;
2. the corresponding upper bound is below zero for dwell or recontact count;
3. the one-sided 95% lower bound for the D0/F8 event-window nail-depth-gain
   ratio is greater than 0.90; and
4. D0 passes the frozen first-window and overall success guardrails.

A nonpositive or nonfinite F8 depth-gain denominator makes that mechanism claim
not distinguishable.

## 10. FQ-min acceptance and interpretation

The confirmatory causal tests and the practical replacement rule answer
different questions. FQ-min may replace F8 as the fixed-impedance **simulation
reward objective** only if every practical condition passes:

1. the all-episode mean contact-quality gain `FQ-min-F8` is at least 0.10;
2. the one-sided 95% paired-bootstrap lower bound for the useful-speed ratio
   `FQ-min/F8` is greater than 0.95;
3. FQ-min first-window and overall success are each at least 0.90 and no more
   than 0.05 below F8;
4. the one-sided 95% paired-bootstrap lower bound for the event-window
   nail-depth-gain ratio `FQ-min/F8` is greater than 0.90;
5. qvel and Lambda channels are finite, liveness is proven, treatment-shared
   safety diagnostics are complete, and no provenance, quota, instrumentation,
   numerical, or sentinel gate fails.

Raw or success-weighted delivered impulse is descriptive and cannot pass or
fail the remedy. The exact paired sign-flip sensitivity cannot pass or fail it
either. The supporting FQ-min-versus-F8 comparison is not added to the
three-test Holm family.

Finite joint-speed exceedances do not remove episodes from the simulator
estimand. They must be reported by phase, magnitude, duration, and offending
joint. FQ-min can be called sampled hardware-speed-qualified only if both
FQ-min and F8 have zero finite 500 Hz exceedances. Otherwise any accepted
replacement is explicitly simulator-only. Because `imp_max_p=0`, no outcome
establishes hard impulse enforcement, hardware safety, or deployability.

## 11. Provenance and execution

Use named-file commits only. Deploy tracked files through commit, push, and a
clean Vega checkout/pull; never copy tracked files with `scp`. Use one GPU per
training run.

Before strict evaluation, freeze a 32-row
`accepted_training_checkpoints.tsv`. Each row binds treatment label, registered
task, short, seed, retained `model_499.pt`, all training attempts and
infrastructure-only retry history, code/config/asset revisions, checkpoint
hash, and clean state.

After infrastructure-only evaluation retries are resolved, freeze a separate
32-row `accepted_evaluations.tsv`. Each row binds the accepted checkpoint to
its evaluation attempt, RNG identities, schema-v3 payload digest and byte hash,
training and evaluation config identities, code/asset revisions, quotas, dirty
state, and all sentinel predicates.

Analysis starts only after all 32 accepted-evaluation identities and all 16,384
sampled episodes verify. There is no partial-matrix result.

## 12. Interpretation matrix

| Outcome | Supported interpretation | Next action |
|---|---|---|
| F0 improves quality without practical loss | Raw speed payout had a harmful marginal effect | Prefer the simpler arm unless FQ-min is practically stronger |
| D0 reduces dwell/recontact without losing depth/success | Raw delivered payout selected contact regime more than useful work | Keep delivered payout disabled |
| D0 loses depth or success | Raw delivered payout contributed useful work | Revisit its form only in a separately designed follow-up |
| FQ-min improves over D0 and passes the F8 practical rule | Minimal quality-gated speed is a viable simulation objective | Adopt provisionally and qualify a speed-aware controller |
| FQ-min improves over D0 but fails an F8 margin | The mechanism is promising but not an acceptable replacement | Keep F8 or D0; identify the failed margin |
| No causal contrast is distinguishable | No tested reader has a large detectable marginal effect at eight seeds | Do not claim equivalence; inspect credit timing and power |
| Quality improves but qvel remains illegal | Reward diagnosis succeeded only in simulation | Do not infer hardware readiness |
| Quality instrumentation fails | The terminal proxy is not trustworthy | Stop before training or invalidate the campaign |

## 13. Explicitly deferred

- any quality-conditioned delivered-impulse reward;
- a bounded center-blind reader package;
- archived-policy-zoo GPU re-evaluation;
- global trajectory reference or straight-line tracking;
- R0/R1/R2 reference-recipe implementation or qualification;
- late ante-impact funnel shaping;
- a full video gallery, annotation UI, or HTML report;
- contact-force reward;
- accuracy as a constrained-RL mechanism;
- changes to impulse caps or accumulation-window semantics;
- `imp_max_p>0`;
- variable impedance or `set_gains`;
- hardware conclusions.

These items require evidence from the lean reward campaign and a separate
decision.

## Appendix A. Deferred reference-recipe notes

This appendix preserves the calibrated recipe design so it is not lost. It is
not an active task, launch gate, file-map entry, or authorization to run CPU or
GPU reference experiments. Revisit it only if the completed reward campaign
identifies a residual terminal credit-assignment problem that a reward-only
treatment did not resolve.

The deferred matched CPU probe compares 32 reset seeds (`0..31`) under:

1. **R0:** the shipped `SingleStrikeReference`;
2. **R1:** the learned impact-only transverse arc from frozen `traj_dc`, with
   R0's axial schedule and pacing;
3. **R2:** R1 through 90 mm axial standoff, then a minimum-jerk transverse
   repair to zero by 20 mm.

The R1 source set is bound to SHA-256
`b22dabb94a10a1e7f68f3fe6a4a2f9412e14916a40a3dbafc81e2f3c3cc7f89f`; no other
trace substitutes. Because those traces lack pre-apex motion, the pre-apex
transverse residual is exactly zero and the apex is repeated at the
wind-up/descent boundary. Resample the observed post-apex descent on 51 uniform
points. Build each residual in the source rollout's live nail frame against
the R0 path constructed from that rollout's live head and nail poses. Select
the residual minimizing summed pairwise transverse L2 distance; ties break by
lexicographic trace digest.

For R2, with axial standoff \(d\),

\[
u=\operatorname{clip}((0.09-d)/0.07,0,1),
\qquad
r_\perp^{R2}=r_\perp^{R1}
\left[1-(10u^3-15u^4+6u^5)\right].
\]

Here the bracketed factor multiplies the R1 transverse residual; the notation
above means “R1 residual scaled by the factor,” not an additive offset. R2 is
identical to R1 through 90 mm and reaches zero residual with continuous first
and second derivatives by 20 mm.

Before outcomes, generate a fresh digested 32-row reset manifest binding
realized qpos/qvel, live head/nail poses, nail-frame basis, code/config/asset
revisions, and canonical state digest. Every recipe must reproduce every row.

The corrected deadline contract is:

- an accepted productive first event must start no later than the final nominal
  script step;
- after the nominal script, issue at most two final-target control steps and
  stop earlier on success;
- normal task success must occur no later than the end of step
  `n_script+2`;
- there is no endpoint hold, and a late onset, late success, press-only
  contact, or missing productive first event is a deadline failure.

R0 first has to reproduce the Phase M in-script productive-contact and success
contract. A candidate R1/R2 promotion then requires at least 30/32 productive
in-script first events, at least 29/32 accepted onsets inside the provisional
12 mm center proxy, median precontact speed at least 95% of R0, every accepted
event above 0.5 m/s, median raw delivered impulse at least 90% of R0, no prefix
joint speed above 3.1415 rad/s, no `Lambda/cap>1`, and no numerical or sentinel
failure.

Terminal axial-speed, terminal lateral-speed, and approach-angle thresholds
must be preregistered in the fresh manifest before any R1/R2 outcome is
inspected. Measure all three in the same live-nail-frame terminal window ending
immediately before the accepted onset. Missing, nonfinite, wrong-window, or
failed values fail promotion. These terminal gates apply to every candidate
promotion; they are not replaced by the 12 mm proxy.

R2 additionally needs at least four more within-12-mm onsets than R1
(`>=4/32`). Replacing R0 additionally needs at least 5% higher precontact speed
or at least 10% lower p95 prefix joint speed. Delivered impulse alone and global
straightness can never justify replacement.

Before any separately authorized GPU reference arm, the generating recipe must
have the largest cumulative raw imitation score in at least 26/32 resets and a
median self/next-best ratio of at least 1.25. Re-run R0 and any candidate at
`solref_scale=2`; reject a candidate if speed, raw delivered impulse, or
worst-joint impulse changes by 20% or more, or if success, centering, deadline,
or terminal gates fail. A CPU pass would show scripted feasibility and weak
prior separability only; it would not show that PPO can learn the recipe.

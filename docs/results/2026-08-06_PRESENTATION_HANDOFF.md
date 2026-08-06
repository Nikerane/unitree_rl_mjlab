# Presentation handoff — Z1 impact-safe hammering

## Start here in the new chat

Work only in:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal`

The presentation must be designed collaboratively with the owner. **Do not edit the
PowerPoint until the narrative, slide sequence, plots, and videos have been discussed
and explicitly approved.** The existing deck is:

`/Users/nikerane/Desktop/_Munich_hammering/Thesis_updates_ppts/Julty_updates.pptx`

Its untouched SHA-256 at handoff is:

`619557de0714b3c512b48cee299b52a1d88962f0f56b66f1397611681db54754`

Use the `presentations` skill when slide work begins. First inspect and render the
existing deck, then propose a narrative and slide-by-slide plan. Do not silently
rewrite the deck.

## Current repository state

- Branch: `overnight-impulse-minimal`
- HEAD and remote before the post-meeting planning edits:
  `52d895e7e5abeaeade198290394b348bcf476b97`
- The branch was created on 2026-08-05 from `origin/cartesian-guideline-fic` to
  isolate the minimal overnight experiment from the broader campaign. It was not
  created for the final audit.
- The result generator and tests are committed and pushed.
- The two result directories below remain untracked intentionally, pending
  collaborative presentation design and a final freeze.
- No Critical or Important engineering defect remains in the accepted numerical
  evidence.

## Owner's immediate objective

Create a clear progress presentation that tells this experimental story:

1. Baseline hammering can solve the nail-driving task but discovers curved or
   inconsistent hammer-head approaches.
2. Ordered waypoint/reference guidance makes the pre-contact path substantially
   straighter for policies that learn the guidance.
3. The resulting guided policies violate the joint-velocity limit unless velocity
   CaT is enabled.
4. At delivered-reward dose D4, velocity CaT produces a complete observed legality
   contrast while retaining similar descriptive first-event impulse values.
5. Within the velocity-CaT arms, the sampled CUDA evaluation shows a descriptive
   D0→D2→D4 first-event impulse ordering, but the one-fixed-reset CPU comparison does
   not reproduce D4>D2.
6. The unchanged project impulse caps are not crossed, so active impulse CaT is not
   scientifically identifiable in this operating regime. No active impulse-CaT arm
   was trained.

The owner cares especially about understandable trajectories and videos. Keep the
presentation visual and concise; do not turn it into a provenance lecture. Put the
necessary scientific limitations in short captions or speaker notes.

## Decisions reached during the 2026-08-06 presentation working session

The owner is manually editing a separate copy of the presentation. The original
deck path and hash above therefore do **not** describe the latest manually edited
slides. Continue with discussion, accuracy checks, and asset selection only. Do not
edit any PowerPoint until the owner identifies the exact current copy and explicitly
approves the narrative, slide sequence, figures, and videos.

### Correct guidance terminology

- C0 is the task-only control: no guidance reward, no active velocity or impulse
  CaT, and delivered-impulse reward weight 0. The policy still observed guideline
  state, so say **“no guidance reward,” not “no reference observation.”**
- G is **one-shot ordered checkpoint/gate guidance**. It is not a continuously
  rewarded distance-to-line treatment.
- P is **dense ordered waypoint-progress guidance**.
- No continuously rewarded “reference-line closeness” policy was trained.
- P was selected for the later campaign because it was simpler and marginally more
  reliable in this small sample, not because it was proven superior to G.

Useful guidance summaries:

| Treatment | All six points by contact | Strict straight | Median pre-contact max error | Median pre-contact RMS error | Median path ratio | Median delivered impulse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G: ordered gates | 4/6 | 3/6 | `15.36 mm` | `6.36 mm` | `1.04` | `0.34 N·s` |
| P: waypoint progress | 5/6 | 3/6 | `14.98 mm` | `6.55 mm` | `1.04` | `0.34 N·s` |

Illustrative guided policies selected for trajectory figures:

| Policy | Points reached | Pre-contact max error | Pre-contact RMS error | Path ratio |
| --- | ---: | ---: | ---: | ---: |
| G seed 3 | 6/6 | `10.88 mm` | `6.00 mm` | `1.0392` |
| P seed 3 | 6/6 | `11.50 mm` | `5.48 mm` | `1.0347` |

Use approach-focused x-z views for this slide. The old full trajectories contain
post-contact follow-through and can make the guidance look worse or conceptually
confusing. The actual experiment used **six** reference points, not three.

### Selected task-only control trajectory

C0 seed 5 is the preferred illustrative control because its delivered impulse is
close to the C0 median and its path visibly shows the curved approach discovered
without a guidance reward.

| Property | C0 seed 5 fixed-CPU value |
| --- | ---: |
| Guidance reward | none |
| Velocity CaT | inactive |
| Impulse CaT | inactive |
| Delivered-impulse reward | D0, weight `0.0` |
| Delivered nail-axis impulse | `0.321521 N·s` |
| Peak joint speed | `4.983424 rad/s`, joint 2 |
| Speed limit | `3.1415 rad/s` — observed rollout is illegal |
| Maximum joint impulse | `0.154304 N·ms`, joint 2 |
| Joint-2 impulse cap | `3.28 N·ms` |
| Maximum `Lambda/cap` | `0.047044` |
| Nail-driving outcome | success |

The legacy `imp_delivered_n_s` field is episode-cumulative. For this particular
rollout it equals the reconstructed first finalized event, but the slide must label
the endpoint carefully. A clean C0 path must not show a guideline or waypoint
circles because C0 had no guidance reward.

### Training configuration for Experiment A

- PPO through RSL-RL; fixed impedance; Cartesian position action passed through
  DiffIK.
- `4,096` parallel environments and `24` steps/environment/iteration.
- Batch size: `98,304` transitions per policy iteration.
- Four minibatches: `24,576` samples/minibatch.
- Five learning epochs, therefore 20 gradient-minibatch updates/iteration.
- `200` policy iterations: `19,660,800` transitions per trained policy.
- Initial learning rate `0.001`, adaptive schedule; `gamma=0.99`, `lambda=0.95`,
  PPO clip `0.2`, entropy coefficient `0.02`, desired KL `0.01`, maximum gradient
  norm `1.0`.
- Actor and critic: `256-128-64` ELU; observation normalization enabled; initial
  policy standard deviation `1.0`.
- Physics `500 Hz`, policy control `50 Hz`, maximum episode duration `20 s`.
- Seeds 2–7: six trained checkpoints/treatment; training used CUDA on an A100.

Exact configuration sources:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-03_wave2_waypoint/provenance/wave2_p_seed5/agent.yaml`

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-03_wave2_waypoint/provenance/wave2_p_seed5/env.yaml`

### Impulse reward and metric explanation

The current linear delivered-impulse reward is project-specific:

```text
r_t = 1[first event finalized, productive, and unpaid] * I_first / I_ref
I_first = sum_k max(F_downward,k, 0) * Delta t
```

- The sum is accumulated at the `500 Hz` physics-substep rate over the first contact
  event.
- The window is capped at 25 substeps, or `50 ms`; this is a maximum measurement
  window, **not an assumption that contact lasts 50 ms**.
- The reward is one-shot, success-censored, and requires more than `0.5 mm` useful
  nail-depth progress.
- `I_ref = 0.3088 N·s` is a provisional scale and is not yet reproducibly
  calibrated.
- Nail impulse in `N·s` and per-joint reaction impulse `Lambda_j` in `N·ms` are
  different measurements and must not be mixed.
- Do not state that this exact reward was copied from one paper. It is a
  project-specific synthesis grounded in impulse-momentum mechanics and sparse
  event-reward design.

### Constraints-as-terminations slide

Use the title **“Constraints as Terminations (CaT)”**. Explain that the current
implementation is a soft modification of the learning return, not necessarily a
physical episode stop:

```text
measure constraint -> compare with limit -> delta = 0 if safe, delta > 0 if violated
future return is scaled by gamma * (1 - delta)
```

The implementation also scales the positive task reward by `(1-delta)`. Velocity
CaT measures the peak joint velocity at the `500 Hz` substep rate against the
`3.1415 rad/s` limit. In the Presentation3 experiment, impulse CaT was log-only
because `imp_max_p=0`.

The clean matched velocity comparison holds guidance and impulse reward fixed:

| Configuration | Reference treatment | Velocity CaT | Delivered reward | Median peak joint speed | Sampled legality | First-event nail impulse |
| --- | --- | --- | --- | ---: | ---: | ---: |
| P+D4 | waypoint progress | no | enabled, weight `4.0` | `5.20 rad/s` | `0/3,072` | `0.340551 N·s` |
| P+V+D4 | waypoint progress | yes | enabled, weight `4.0` | `2.96 rad/s` | `3,072/3,072` | `0.338672 N·s` |

Slide-safe wording: **At D4, velocity CaT restored observed joint-speed legality
while retaining similar first-event nail impulse.** Do not describe this as “no
reference versus reference”; the reference treatment and delivered reward were held
fixed and only velocity CaT changed.

There is no active impulse-CaT comparison to tabulate. For the same D4 arms, the
log-only impulse monitoring result was:

| Configuration | Median maximum `Lambda/cap` | Largest sampled `Lambda/cap` | Cap crossings |
| --- | ---: | ---: | ---: |
| P+D4, no velocity CaT | `0.141188` | `0.192924` | `0/3,072` |
| P+V+D4 | `0.523345` | `0.760081` | `0/3,072` |

The `0.760081` maximum came from P+V+D4 seed 2, sampled CUDA episode
`V+M-env213-episode0`, at joint 1: `Lambda=1.2465335 N·ms` against a
`1.64 N·ms` cap. Its first contact was at `0.112 s`, its finalized first-event nail
impulse was `0.3295908 N·s`, and it successfully drove the nail. Across all four
arms there were `0/12,288` sampled crossings and `0/24` fixed-rollout crossings.
Say **“remained below the project-defined cap,”** not “not even close to the cap.”

The sampled velocity-CaT dose medians were D0 `0.317080 N·s`, D2
`0.326769 N·s`, and D4 `0.338672 N·s`. The fixed-CPU medians were D0
`0.324832 N·s`, D2 `0.361221 N·s`, and D4 `0.349387 N·s`; therefore D4 was
not above D2 in the fixed comparison. Present this as descriptive evidence only.

## Experiment A — waypoint/reference guidance

Authoritative result note:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/docs/results/2026-08-03_wave2_waypoint_replication_result.md`

Failure atlas:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/docs/results/2026-08-03_wave2_guided_failure_atlas.md`

Frozen Wave-1 evidence/videos:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-02_wave1_waypoint/`

Pooled endpoints across six policies per arm:

| Arm | Meaning | All six gates by contact | Predeclared straight |
| --- | --- | ---: | ---: |
| C0 | task reward, no guidance treatment | 0/6 | 0/6 |
| G | one-shot ordered gate reward | 4/6 | 3/6 |
| P | dense unseen waypoint-progress reward | 5/6 | 3/6 |

Interpretation:

- Guided arms produced nine gate-completing policies versus zero controls.
- Six of twelve guided policies met the strict straight label.
- Three guided policies ignored guidance entirely while still driving the nail.
- Progress-only was selected pragmatically for later experiments because it was
  simpler and marginally ahead, not because P was proven superior to G.
- All 18 policies were qvel-illegal at the fixed reset. Maximum observed
  `Lambda/cap` was only about `0.128`.

Visual rule: only show waypoint/reference geometry for policies trained with that
guidance. For the later Presentation3 arms this geometry is appropriate because all
four retain waypoint-progress weight 8. Do not draw the old gate disks; use the
straight dashed guideline and ordered waypoint markers.

## Experiment B — velocity CaT and delivered-impulse reward

### Exact arms

All four arms retain waypoint-progress guidance, fixed impedance, and the same
DiffIK action interface. Impulse CaT is log-only in every arm (`imp_max_p=0`).

| Slide label | Exact treatment | Delivered reward | 500 Hz velocity CaT |
| --- | --- | ---: | --- |
| M | P+D4 | D4 (`4.0`) | no |
| V+D0 | P+V+D0 | D0 (`0.0`) | yes |
| V | P+V | D2 (`2.0`) | yes |
| V+M | P+V+D4 | D4 (`4.0`) | yes |

Prefer plain-language audience labels such as “D4 · no velocity CaT” rather than
unexplained M/V codes.

### Evidence population

- 24 trained checkpoints: four arms × seeds 2–7.
- Strict sampled evaluation: 512 CUDA episodes/checkpoint, 12,288 total episodes.
- Fixed comparison: one identical digest-bound CPU rollout/checkpoint.
- Scientific replicate is the trained checkpoint: `n=6` per arm. Do not describe
  12,288 episodes as 12,288 independent policies.

### Headline results

Velocity legality at D4:

- M, without velocity CaT: `0/3,072` sampled episodes legal.
- V+M, with velocity CaT: `3,072/3,072` sampled episodes legal.
- Sampled first-event checkpoint medians: M `0.340551 N·s`, V+M `0.338672 N·s`.
- Matched V+M−M median: `+0.005132 N·s`, 4/6 positive.

Across all velocity-CaT arms, legality was `9,213/9,216`; the three violations all
occurred in V/D2 seed 2. Therefore velocity CaT shows a strong observed contrast but
is not a hard guarantee and does not establish hardware safety.

Delivered-reward dose within velocity-CaT arms:

| Dose | Sampled CUDA first-event median | Fixed CPU first-event median |
| ---: | ---: | ---: |
| D0 | `0.317080 N·s` | `0.324832 N·s` |
| D2 | `0.326769 N·s` | `0.361221 N·s` |
| D4 | `0.338672 N·s` | `0.349387 N·s` |

- Sampled matched D2−D0 median: `+0.011350 N·s`, 6/6 positive.
- Sampled matched D4−D2 median: `+0.014680 N·s`, 5/6 positive.
- Fixed matched D4−D2 median: `−0.008917 N·s`, only 2/6 positive.

Permitted conclusion: the sampled CUDA evaluation shows a descriptive checkpoint-
mean ordering. Do not claim a proven, significant, causal, universal, or backend-
independent monotonic dose response.

Impulse-cap result:

- Strict crossings: `0/12,288` sampled episodes and `0/24` fixed rollouts.
- Maximum sampled `Lambda/cap = 0.760081`.
- Every arm used `imp_max_p=0`; impulse CaT was not active.
- Correct conclusion: active impulse CaT was **not identifiable and was not trained**.
- Caps are project-defined engineering thresholds, not manufacturer-certified
  safety limits.

Retained exception:

- M seed 6 completed all six waypoints in `0/512` sampled episodes, while still
  succeeding at the task.
- The bank contains 12,287 success-finalized first events and one productive
  window-finalized event, also from M seed 6.
- Keep that checkpoint visible; never filter or replace it.

### Metric language that must remain exact

- Primary impact endpoint: **finalized productive first-event nail-axis impulse**.
- Episode cumulative impulse = first event + post-event tail. It is not the same as
  first-strike impulse and can move in the opposite direction.
- The legacy `recontact` implementation means any contact after finalization; release
  is not required. Audience label: **post-finalization contact**.
- Sampled CUDA and fixed CPU results must remain visibly separate and must not be
  pooled.

## Supervisor-meeting update — future work, not current evidence

Meeting transcript:

`/Users/nikerane/.codex/attachments/d0f750db-d84e-4a35-8f68-d11771d3e03d/pasted-text.txt`

The transcript is imperfect, so distinguish firm directions from tentative or
garbled suggestions. The consolidated directions are:

1. Prefer explicit trajectory/reference guidance over adding more ad-hoc reward
   terms. Retain the current simple reference for now; investigate a more principled
   way of producing the trajectory later. The relevant phrase may have been
   “trajectory optimization,” but it is ambiguous. Do not infer approval for a
   two-level generate-then-track architecture without reconfirming it.
2. For future training, move from three-dimensional Cartesian relative-displacement
   actions plus DiffIK toward direct joint-position targets. A task-space
   hammer-trajectory tracking reward remains acceptable.
3. Demonstrate the active impulse-CaT mechanism quickly by deliberately lowering
   the user-defined impulse caps until the current behavior reaches them, then
   compare CaT off versus on. Call this a **diagnostic threshold**, not a physical
   safety limit.
4. Calibrate `I_ref` independently: drop a known object of approximately hammer mass
   onto the nail from controlled height(s) and measure useful axial impulse with the
   production first-event tracker. The pinned Presentation3 asset assigns **0.2 kg
   to the entire rigid hammer body** (not to a separately modeled head) and
   **0.045 kg to the printed fixture**. A 0.2 kg dropper matches the detached
   simulated impactor; a 0.245 kg dropper is a distinct fixture-plus-hammer
   sensitivity condition and changes ideal same-height momentum/energy by 22.5%.
   Do not silently pool the two, and do not train a policy merely to choose the
   reward normalization scale.
5. A jitter penalty was suggested, but the environment already has an action-rate
   penalty. Measure the failure first and add another term only if the existing term
   does not address it.
6. Add curriculum and domain randomization only after a stable baseline exists.
7. Variable impedance remains the final stage, after the joint-space fixed-gain
   baseline.
8. A fixed trajectory may be useful when isolating the effect of impedance, but this
   part of the transcript is especially tentative and requires confirmation.

### Controlled-drop mass decision (confirmed 2026-08-06)

- This is a **simulation-only calibration**. "Independent" means independent of
  the learned policy, robot kinematics, and robot controller—not independent of
  the simulator. A physical hardware drop is outside the scope of this step.
- Use the production nail/contact physics and the production first-event delivered-
  impulse tracker so the resulting scale matches the quantity used by training.
- Geometry: a **flat-faced cylindrical dropper**, centered on the nail and
  constrained to translate along the nail axis without rotation. This removes
  rectangular-block orientation as an uncontrolled contact variable.
- Minimum experiment: define `h0` as the live reset clearance from the hammer's
  striking face to the nail contact surface, then run **one centered 0.200 kg drop
  from rest at 1.0 h0**. Log measured pre-contact velocity rather than relying only
  on the ideal `sqrt(2 g h)` value.
- Primary calibration object: **0.200 kg flat-faced detached-hammer surrogate**.
- Sensitivity condition: **0.245 kg flat-faced surrogate**, representing the
  0.200 kg hammer plus the 0.045 kg printed fixture.
- Optional follow-ups only if useful after inspecting the primary result: repeat at
  `0.5 h0` and `1.5 h0`, and/or repeat with the 0.245 kg sensitivity mass. Keep the
  mass conditions separate; 0.245 kg changes ideal same-height momentum and kinetic
  energy by **22.5%** and must not be pooled with the primary calibration.
- Neither dropper is claimed to reproduce the arm's configuration- and
  controller-dependent effective mass; the purpose is to provide an independent,
  reproducible scale for the delivered-impulse normalizer.

The presentation task requires no new training. The table below records the
supervisor-derived roadmap. The owner reconfirmed on 2026-08-06 that this ordering
is canonical:

| Priority | Proposed engineering step |
| --- | --- |
| P0 | Preserve and present the current evidence without retroactively changing its meaning. |
| P1 | Move policy actions to direct joint-position targets under fixed impedance while retaining task-space trajectory tracking. |
| P2 | Calibrate `I_ref` independently with the simulation-only controlled drop. |
| P3 | Temporarily lower the impulse threshold and validate active impulse CaT off versus on. |
| P4 | Add curriculum and domain randomization after the fixed-gain baseline and CaT validation are stable. |
| P5 | Add variable impedance last. |

Jitter remains a measure-first diagnostic within these stages: inspect it before
adding any new penalty, but do not promote it into a stage that reorders the
supervisor's roadmap.

### Canonical post-meeting execution sequence (reconfirmed 2026-08-06)

1. **Move policy actions from Cartesian DiffIK to joint space under fixed
   impedance.** Preserve the Cartesian implementation as a separate registered task
   and branch. The new policy commands absolute default-offset joint targets with
   per-joint scaling and physical-limit clipping; it does not command deltas from
   the current joint state.
2. **Retain task-space trajectory tracking.** Keep the current P
   waypoint-progress/reference guidance and task-space measurements while changing
   only the policy action interface.
3. **Calibrate `I_ref` with the simulation-only controlled drop.** The minimum
   experiment is one centered 0.200 kg cylindrical drop from rest at `h0`; optional
   mass/height sensitivity checks follow only if useful.
4. **Temporarily lower the impulse threshold to validate active impulse CaT.** Call
   it a diagnostic threshold, not a physical safety limit, and compare CaT off
   versus on under a matched fixed-impedance setup.
5. **Add curriculum and domain randomization** only after the preceding fixed-
   impedance baseline and active-CaT validation are stable.
6. **Introduce variable impedance last.** Branch VIC from the verified joint-space
   baseline. The policy commands the same desired joint positions plus per-joint
   proportional stiffness; stiffness may move below or above nominal, and damping
   is coupled to stiffness following Bogdanovic--Khadiv--Righetti.

Joint-space fixed-baseline gates before stages 3--5:

1. Verify action scaling, physical-limit clipping, reset behavior,
   observation/action dimensions, and compatibility with existing rewards,
   reference guidance, and CaT hooks.
2. Confirm the task can execute a scripted/reference strike under nominal fixed
   gains before spending GPU time on learning.
3. Run the repository reward/contact/unit gates and short CPU/CUDA smokes before
   fixed-gain training.

The executable Stage-1 plan is:

`docs/superpowers/plans/2026-08-06-z1-joint-position-fixed-stage1.md`

It freezes the P+V+D4 parent treatment, causal DiffIK-to-joint qualification,
six-joint default-offset action contract, paper-aligned joint command-trackability
term, one-seed-then-three-seed gates, and the boundary that keeps the controlled
drop, active impulse-CaT diagnostic, curriculum/randomization, and VIC out of the
Stage-1 branch.

The agreed command-trackability representation is a nonnegative calibrated cost,
`k_tt * ||q_des-q_next||^2`, with reward weight `-1.0`. Its effective contribution
is therefore negative. Register `r_tt` explicitly as a negative term in the
soft-CaT reward split so a velocity violation cannot discount this penalty. Fixed
impedance and later VIC use the same frozen `k_tt`.

Future VIC-stage integration gates:

1. Implement per-joint `Kp`, coupled `Kd`, gain reset, bounded gain mapping, and
   gain telemetry.
2. Verify that nominal constant gains reproduce the fixed-gain controller path
   closely; then test lower/higher gain commands for stability and correct actuator
   updates.
3. Run a **one-seed matched learnability pilot**: joint fixed impedance versus joint
   VIC, initially with impulse CaT log-only. If both learn a genuine strike, expand
   to three seeds before any final campaign. This pilot validates controller
   integration; the later active-constraint comparison evaluates safety.

Promotion gate from one VIC-stage seed to three seeds:

- At least 90% task success over 64 evaluation episodes for each controller.
- Genuine impact behavior rather than a slow press.
- No NaNs, controller instability, or persistent joint-limit saturation.
- VIC gains vary meaningfully and are not persistently pinned to either bound.
- Reduced reaction impulse is logged but is not a required promotion criterion in
  this log-only integration pilot.

### Current research-question wording

A concise thesis-level version is:

> Can a learned variable-impedance policy deliver strong hammer impacts while
> keeping per-joint impact loads within prescribed limits?

The current evidence is fixed-impedance evidence. The slide or narration must make
clear that variable impedance is the planned final stage, not an achieved result.

## Temporary presentation aids created during the working session

These are discussion aids, not frozen result assets. Verify that they still exist
and regenerate them from the authoritative raw traces if necessary:

`/Users/nikerane/.codex/visualizations/2026/08/06/019fd652-d625-7f51-bb6b-38d3c8cce143/c0-seed5-clean-trajectory.png`

`/Users/nikerane/.codex/visualizations/2026/08/06/019fd652-d625-7f51-bb6b-38d3c8cce143/checkpoint-guidance-seed3-trajectory.png`

`/Users/nikerane/.codex/visualizations/2026/08/06/019fd652-d625-7f51-bb6b-38d3c8cce143/waypoint-progress-seed3-trajectory.png`

`/Users/nikerane/.codex/visualizations/2026/08/06/019fd652-d625-7f51-bb6b-38d3c8cce143/checkpoint-guidance-g3-approach.png`

`/Users/nikerane/.codex/visualizations/2026/08/06/019fd652-d625-7f51-bb6b-38d3c8cce143/waypoint-guidance-p3-approach.png`

Raw trajectory traces:

- C0 seed 5:
  `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_c0_seed5/trace.npz`
- G seed 3:
  `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_g_seed3/trace.npz`
- P seed 3:
  `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_p_seed3/trace.npz`

## Current Presentation3 assets — inspect, do not freeze yet

Current directory:

`evaluation/results/2026-08-06_presentation3_final/`

It contains a table, four figures, one 2×2 video, and a poster. These assets were
intentionally left untouched for owner collaboration. Their current hashes are:

- table: `fe9ddd100103d3f6a5fcf05c2837b945917c2026d5a052162a3457a75e93d9a7`
- dose figure: `9470e3db5e318e8bb9f765ac50c848dcb45b9832c4d1d7d4d878decef8f71baf`
- velocity figure: `223874955688e15be9f01d4c9e7bc238e0ce965605d1325de144c50df2cdba9a`
- cap figure: `3509a612d5c1c168ff65ce1847e8c5d4f4f4b3df315a0f1e9865f98a55cf1d67`
- trajectory grid: `c52c8175ddcf6f5c2696473292bcd58b074c2006fb4f3f8181eb2b6bccd5c154`
- composite MP4: `1817ecae4ecc28c164a249b2869fbe806b621f5348d1af91307ccfefecc3010f`
- poster: `8b128e678d219293d0ae61932bc9cde80063b5420c3676958654958fb499af1d`

Important: these are pre-presentation-review assets, not final deliverables. The
current cap title sounds like an active impulse-CaT result and must be replaced. The
committed hardened generator also renames the legacy exported contact column; a
fresh hardened table has SHA-256
`4df0ccaaf0afb7bda055c0c3e569130cf2aecfab8c899065d45ca6bfaf3d9094`.
All numerical cells are unchanged by that rename, and all four regenerated figures
were byte-identical before presentation-specific redesign.

### Required visual corrections to discuss with the owner

1. Retitle the cap panel as log-only measurement, explicitly state 0/12,288 and
   “project-defined caps; not manufacturer-certified safety limits.”
2. Put protocol labels on every plot: sampled CUDA or identical fixed CPU reset.
3. State `n=6 checkpoints/arm`; explain thin seed traces versus thick median.
4. Use plain-language treatment labels rather than M/V shorthand alone.
5. Give the two dose panels one shared y scale if they invite direct comparison.
6. Keep first-event and cumulative impulse visibly distinct.
7. Define the nail target marker in trajectory plots.
8. Review the seed-2 composite with the owner. If retained, document that seed 2 is
   the numerically lowest frozen seed and record the terminal-frame padding rule.

## Raw input locations

Existing-arm sampled r4:

`evaluation/results/2026-08-06_presentation3_vd0/sampled_existing_r4/`

V+D0 sampled:

`evaluation/results/2026-08-06_presentation3_vd0/sampled/`

Existing-arm fixed evidence:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-05_presentation3/`

V+D0 fixed evidence/checkpoints:

`evaluation/results/2026-08-06_presentation3_vd0/`

Frozen manifests:

- `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/binding_first_18_r4.tsv`
- `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/pvd0_6.tsv`

## Audit sources

- Raw science and permitted claims:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/fresh-raw-science-audit.md`
- Final scientific wording review:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/final-scientific-review.md`
- Visual/reproducibility findings:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/final-visual-repro-review.md`
- Final generator review — APPROVED:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/fresh-generator-contract-rereview.md`
- Independent integrity review — APPROVED:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/independent-integrity-fix-review.md`

## Next-chat workflow

1. Read this handoff completely. Start by summarizing the project in plain language,
   especially the seven-term task reward, G/P reference guidance, the one-shot
   first-event delivered-impulse reward, velocity CaT, and the log-only impulse
   measurement.
2. The owner has manually edited a separate presentation copy. Ask for the exact
   current PPTX or current slide screenshots/sequence before assuming that the deck
   path at the top contains the latest slides.
3. Continue agreeing on the narrative, slide sequence, figures, videos, and exact
   wording. A current design question is whether “Constraints as Terminations” should
   contain both the mechanism and result tables or be split into mechanism and
   results slides.
4. Use the `presentations` skill to inspect and render the owner-designated deck, but
   do not edit it until the owner explicitly approves both the content plan and the
   exact copy to modify.
5. Review all figures and videos together. Do not assume either the Presentation3
   assets or the temporary trajectory aids are final visual designs.
6. After explicit owner approval, edit only a copy of the PowerPoint, render every
   slide, and visually verify layout, labels, equations, figures, and video links.
7. Only after the owner approves the final deck: regenerate/finalize assets, create
   one hash inventory with video-selection/composition provenance, back it up, and
   mirror it read-only. Do not freeze intermediate presentation drafts.

No new training, evaluator work, reward machinery, or infrastructure is needed for
the presentation task. The supervisor's proposed engineering experiments belong in
a separate task.

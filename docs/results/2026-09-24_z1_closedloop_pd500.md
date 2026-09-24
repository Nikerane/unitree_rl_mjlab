# Closed-loop fixed-gain transfer — 2026-09-24

## Result and decision

The old actors can make contact when they receive fresh feedback under the
sampled500Hz controller, unlike replaying their old targets. This supports
moving to a **fresh-training pilot**, not insisting that a transferred old
policy become safe before learning.

At the finer0.125ms physics resolution, the dose.2/.3 actors contact the nail
in34/128 and102/128 cases;30 and101 have a complete50ms first-contact window
within the scored episode. However **J2/J3/J4 exceed3.1415rad/s in every
first episode** at both resolutions. J6 remains below1.561rad/s, with no
screened sustained saturated torque reversals on any joint.

This is a posttraining fixed-controller intervention, **not new training**,
not optimal-impedance evidence and not hardware safety qualification.

## Setup and integrity

- Four runs: two existing trained actor instances x physics.25/.125ms.
-128 worlds, seed2026092101, external speed goal2.2m/s, same task/dose/cost.
- Kp=[256,384,384,256,192,128], Kd6.4, gripper100/20 unchanged.
  Targets update50Hz from fresh actor inference; torque updates500Hz and
  holds between updates. Stiffness outputs are overridden by the fixed gains.
  Speed is externally supplied, matching these checkpoints' training phase.
- Each actor is first strictly loaded and contract-validated in its unchanged
  original2ms/decimation10 environment. That actor is retained for a separate,
  explicitly intervened environment; no old checkpoint metadata is rewritten.
- Native rewards, terminations and resets run unchanged. Only first episodes
  are scored, including terminal substeps; masks exclude later episodes.
- Maximum500ms observation per first episode. Still-active episodes are
  **censored**, not task failures. Native productive contact is not final
  task success.
- Initial state/physical-observation hashes match between paired resolutions.
  Actual applied-torque error0; all holds bitwise constant; exactly10 PD
  updates per20ms action interval.
- No production-source changes, training, cap increases or state clipping.

## Results

| Source dose | Physics ms | Contact /128 | Complete50ms windows | Native productive contact | Censored episodes | J6 peak rad/s |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.25 | 37 | 34 | 32 | 122 | 1.226 |
| 0.2 | 0.125 | 34 | 30 | 29 | 120 | 1.466 |
| 0.3 | 0.25 | 104 | 103 | 104 | 127 | 1.554 |
| 0.3 | 0.125 | 102 | 101 | 102 | 127 | 1.561 |

All runs have joint-speed breach counts[0,128,128,128,0,0].
Fine-resolution peaks:

- Dose.2: [2.709,7.651,11.119,11.159,2.497,1.466]rad/s.
- Dose.3: [2.396,5.729,6.290,6.902,2.431,1.561]rad/s.

Median delivered impulse on complete windows is0.1223Ns (dose.2) and0.1170Ns
(dose.3) at the finer resolution. These are **conditional on complete contacts**,
not all-world performance. Measured median onset speeds are1.873 and3.288m/s;
the latter exceeds the supplied2.2m/s target substantially.
Control-step geometric tracking RMS is about78.0/81.0mm respectively.

Unlike frozen-tape comparisons, these closed-loop trajectories can diverge:
feedback produces different subsequent actions after small physical changes.
For example, dose.2's maximum delivered impulse differs strongly across
resolutions (.5871 versus.2354Ns), with different contact/complete populations.
Do not call the impact measurements numerically converged from this comparison
or compare unpaired conditional extremes as a causal timestep effect.

Contact-row joint impulses are saved in the analysis, but have **no validated
hardware safety thresholds**. They are external generalized contact impulse,
not internal gearbox load. The historical enforced baseline-subtracted
measurement and its provisional caps remain unchanged in this diagnostic.

## Next step requested by user

An active goal now covers a bounded **fresh250-update pilot**, one seed,
selected fixed gains/500Hz controller, existing task/reward and
speed-conditioning phase, p=.2 and speed/effort limits unchanged. The training
route must first record the changed controller/timing honestly in new
checkpoints and pass required reward/contact/setup gates plus checkpoint
reload. Existing checkpoint metadata currently requires2ms/decimation10;
do not silently weaken that old contract.

Old-policy success is **not** a training prerequisite. Held-out evaluation
must report all-world contact utility, complete-window delivered impulse,
tracking and all-six speed violation rates. No automatic long run or
hardware safety claim. Exact pilot design confirmation requested in chat;
no training submitted by this diagnostic.

## Evidence and checks

- CPU job45857043 array0–3, all COMPLETED exit0; durations3:12,6:00,3:20,6:17.
- Campaign /cephhome/eunikhilr/campaigns/z1-closedloop-pd500-20260924.GeG463.
- Local raw traces .superpowers/sdd/2026-09-24-closedloop-pd500/results/.
- [Compact analysis](assets/2026-09-24_closedloop_pd500/analysis.json);
  scripts/protocol banked alongside.
- New real-C-MuJoCo step-wrapper test observed RED→GREEN; together with
  existing sampled-PD tests10 passed.
- Full project pytest -x:377 passed, existing
  tests/test_docs_current.py::test_index_points_to_current_fic_baseline_and_vic_prototype_routes
  failed on hard-coded index date;35 upstream Torch warnings.
- Independent review found no scoped prelaunch blocker and checked actor
  lifetime, strict original load, controller clocks and terminal-step masks.
  Its requested active-mask/complete-window reductions were applied.

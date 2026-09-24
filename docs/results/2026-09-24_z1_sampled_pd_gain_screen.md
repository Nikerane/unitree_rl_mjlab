# 500 Hz sampled-PD gain screen — 2026-09-24

## Decision

A bounded startup candidate exists: canonical arm stiffness
**[256,384,384,256,192,128] Nm/rad**, uniform damping **6.4 Nm·s/rad**.
It removes the screened sustained saturated reversals and agrees closely at
the two tested physics resolutions. This is **not a safe policy, impact result,
hardware recommendation, or qualified variable-impedance range**.

J6 peak falls from3.1685 to1.0459rad/s, with no J6 speed breaches in either
source tape at either resolution. However, **J4 exceeds3.1415rad/s in all256
fine-resolution replay worlds**, peaking4.3258rad/s. All seven settings breach
at least one arm-joint speed limit. Do not change the speed/effort limits.

Next bounded step: full-strike diagnostic at this candidate gain vector,
retaining500Hz held torque and50Hz targets. Check all-six speed maxima,
saturation/reversals, delivered impulse, and contact-only joint-load numerical
sensitivity at matched physical times and complete contact windows. Frozen
old command tapes test controller response, not an adapted closed-loop policy.
If commands miss or windows are incomplete, report that rather than interpreting
it as zero impact or failed learning. No training was launched by this screen.

## Experiment

- Seven settings: full reference stiffness[512,768,768,512,384,256] with
  damping25.6, then full/half stiffness crossed with damping12.8,6.4,3.2.
  These are simulation candidates, not certified manufacturer gain limits.
- Two existing policy command tapes,128 worlds each; first60ms only.
  Both tapes terminate after this prefix, so no post-termination hold was used.
- Targets50Hz, PD torque sampled/clipped/held at500Hz, physics0.25/0.125ms.
  Finer physics does not update control faster.
- Gripper gains100/20 unchanged; all seven actuators have exactly30 torque
  updates per replay. Every inter-update torque hold is bitwise constant.
  Actual actuator force equals held torque (maximum error0).
- All28 replays completed; **no hammer–nail contact** in any.
  Paired initial-state, target, runtime, gain, and effort-limit identities match.
- No learning, production-source changes, new reward weights, or limit changes.
  Ideal body gravity compensation and the remaining model assumptions persist.

## Results

Values below pool the two128-world source tapes at the finer physics step.
World counts are repeated simulated cases, not independent training seeds.

| Stiffness scale | Damping | J6 peak rad/s | Worst arm peak rad/s | J6 sustained-chatter worlds /256 | Largest paired peak-speed difference rad/s | Startup screen |
|---|---:|---:|---:|---:|---:|---|
| 1 | 25.6 | 3.169 | 4.174 | 256 | 0.0861 | Reject |
| 1 | 12.8 | 3.625 | 5.497 | 256 | 0.0453 | Reject |
| 1 | 6.4 | 2.246 | 6.108 | 0 | 0.0192 | Candidate |
| 1 | 3.2 | 2.768 | 6.549 | 0 | 0.0443 | Candidate |
| 0.5 | 12.8 | 2.949 | 3.303 | 256 | 0.5537 | Reject |
| 0.5 | 6.4 | 1.046 | 4.326 | 0 | 0.0076 | Candidate |
| 0.5 | 3.2 | 1.336 | 5.794 | 0 | 0.0116 | Candidate |

Screened chatter means at least3 consecutive saturated torque sign reversals
(four control ticks). This is a diagnostic heuristic, not causal proof of
oscillation. The candidate's J6 median total sign reversals also falls29→2,
and J6 saturation fraction falls96.67%→0%; the finding is not just crossing
the saturated-reversal threshold. The baseline also has sustained J5 chatter;
the selected candidate has none on any arm joint at either resolution.

Numerical screening budgets were preregistered as per-joint/per-source
p95 peak-speed difference≤0.05×3.1415 and maximum≤0.10×3.1415rad/s.
The selected candidate's largest difference is0.007585rad/s.
This is agreement at two resolutions over60ms, **not a general convergence proof**.

### Why select half stiffness / damping6.4?

Among the four no-sustained-chatter, numerically agreeing candidates, it has
the lowest worst-arm speed and breaches only J4, not J1/J5 as the full-stiffness
candidates do. It is not simply immobilizing the hammer:
median net head displacement is68.28mm versus63.43mm at the reference setting.
J6 joint-target RMS error falls0.05223→0.03616rad. Whole-task path tracking
and impact effectiveness remain untested by this short prefix.

Its canonical joint peaks are
[1.6760,2.5204,1.4151,4.3258,1.7045,1.0459]rad/s.
J2 remains torque-saturated throughout this prefix; J3 and J4 saturate for
about66.5% and76.6% of samples. Removing the wrist switching therefore does
not solve the large-command/speed-compliance problem for the rest of the arm.
These are gains applied to **old fixed command tapes**, not policies retrained
for the changed sampled controller.

## Provenance and verification

- Vega campaign:
  /cephhome/eunikhilr/campaigns/z1-pd500-gains-20260924.HNeBkc
- CPU-partition job45840024, array0–6, all COMPLETED exit0; elapsed85–126s
  per setting. Initial unused GPU allocation45840023 was cancelled while
  pending, then resubmitted with --partition=cpu --gres=none.
- Source commit cc46d9998a3c9cd5d8813e89b57e44248353e03e.
  Assets commit b58ccd2f81fd246f27c1e8d88cf86484cd888703.
  Runtime identity remains enforced against each source capture.
- Full raw traces and per-replay hashes are retained remotely and locally at
  .superpowers/sdd/2026-09-24-gain-screen/results/.
- Compact analysis, protocol, and scripts:
  [assets](assets/2026-09-24_sampled_pd_gain_screen/analysis.json).
  Run analyze.py against the complete raw-results directory to reproduce.
- Seven new helper tests observed failing before implementation, then passing.
  New helpers plus prior sampled-PD tests:16 passed.
- Project pytest -x:377 passed,1 pre-existing failure:
  tests/test_docs_current.py::test_index_points_to_current_fic_baseline_and_vic_prototype_routes
  (hard-coded September21 index date versus existing September22 heading).
  35 upstream Torch deprecation warnings. Full-suite pass is not claimed.
- Independent scoped review found no critical/important defect. Reviewer
  required excluding contact-positive replays (none occurred) and cautioned
  against treating the chatter heuristic as proof. Hardware, impact and
  learning performance were not assessed.

## Interpretation

The data support changing damping as well as stiffness when calibrating this
sampled500Hz simulation controller. They do not establish a real Z1 gain range
or show that the research objective is achieved. The remaining J4 overspeed is
real within these simulated replays and must stay visible in subsequent tests.

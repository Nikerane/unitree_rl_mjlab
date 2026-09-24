# Finer-step startup replay and separate wrist-gain probe

2026-09-22. No learning or production edits. Same60ms frozen-action,128world,1.85m/s-source diagnostic as the [previous torque study](2026-09-22_z1_torque_timestep_diagnostic.md). These are open-loop counterfactuals, not full policy qualification.

## 1. Original gains, finer physics

| dt (ms) | Dose .2 peak J6 rad/s | Dose .3 peak J6 rad/s | World-zero velocity sign changes, .2/.3 |
|---|---:|---:|---:|
| .5 | 1.308782 | 1.300222 | 48/30 |
| .25 | .669530 | .728876 | 0/0 |
| .125 | .473070 | .470994 | 0/0 |
| .0625 | .473358 | .471288 | 0/0 |

At .25ms and below, the sampled wrist force saturates only on the first substep in world zero. The last refinement changes global wrist peak by about .06%. At matched physical times, maximum world-zero wrist position differences are2.17e-5rad (.2) and3.15e-5rad (.3), but maximum velocity differences still reach .06175/.06098rad/s during the early transient. Thus peaks and positions are settling; complete time-history convergence has not been established to a predeclared tolerance. No need to conceal the remaining early-time difference or call .0625ms a mandated production timestep.

## 2. Separate fixed wrist-gain counterfactual

Replace only J6 gains with Kp=256N·m/rad,Kd=25.6N·m·s/rad, derived from documented [SDK defaults and conversion](../research/reward-design/2026-09-22_z1_gain_range_provenance.md). Hold them fixed over the60ms replay. This deliberately replaces the wrist's recorded stiffness action in the diagnostic; it is not a new variable-impedance mapping. All six position targets and other five joints' gains remain identical to the saved-command replay. Force limits remain unchanged. Both stiffness and damping change together, so this does not identify which gain alone resolves chattering.

| Policy | dt | Peak J6 rad/s | World-zero saturation | World-zero sign changes |
|---|---:|---:|---:|---:|
| .2 | 2ms | .477802 | 0/30 | 0 |
| .2 | .0625ms | .484275 | 0/960 | 0 |
| .3 | 2ms | .471253 | 0/30 | 0 |
| .3 | .0625ms | .477382 | 0/960 | 0 |

Initial torque demand and actual force both have magnitude14.122131N·m, versus44.131660 requested/30 applied under the original gains. Coarse/fine peak differences are about1.3%; full trajectory equivalence is not claimed. End wrist positions differ by .000182rad (.2) and .000877rad (.3), with other joints still retaining original gains and their own numerical dynamics.

## Interpretation

The original wrist gains, initial target and force clamp produce a strongly timestep-sensitive alternating response. Softer *paired* wrist gains remove observed J6 saturation/chattering at the existing2ms timestep in this precontact probe. This supports evaluating lower wrist gains, rather than blindly raising penalties or prescribing a32×finer physics step. It does not prove the SDK defaults are optimal, hardware-safe, or adequate at impact. It also does not validate every other joint or settle gain range calibration.

Next useful decision: a bounded full-strike, no-learning gain evaluation with explicit treatment/checkpoint semantics and tracking/delivered-impulse/load reporting, before selecting new policy gain bounds or retraining. Preserve the original checkpoint contract; do not silently reinterpret its stiffness outputs.

## Verification/provenance

Vega45732966: eight refinement replays, COMPLETED0:0,42s. Vega45733118: four wrist-gain replays, COMPLETED0:0,23s. Full initial qpos/qvel hashes agree across all conditions; source arm reset state exact; no accepted contact in60ms; correct sample counts; expected wrist gains verified; native torque reconstructed within .01N·m; every saved position target and non-wrist gain checked identical against the corresponding unmodified-gain run. No active jobs remained at completion.

Artifacts: [refinement](assets/2026-09-22_timestep_refinement/) and [wrist-gain probe](assets/2026-09-22_sdk_wrist_probe/). Only experiment-local `torque_timestep_probe.py` gained timestep options, post-step position recording and the explicit wrist-default intervention. Production source, rewards, thresholds and action mapping are unchanged.

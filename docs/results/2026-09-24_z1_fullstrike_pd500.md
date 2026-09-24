# Selected-gain full-strike replay — 2026-09-24

## Decision

**Wrist chatter remains absent, but this is not a usable or safe strike.**
In the280ms frozen-command replay, J6 stays below3.1415rad/s and no arm joint
shows the screened sustained saturated reversals. J2/J3/J4 exceed the speed
limit in all256 source cases at both physics resolutions, including within
the original command prefixes and before nail contact.

Only one case touches the nail, at276.75/277ms depending on resolution.
It is after the source command prefix. There are **zero complete50ms impact
windows**, so neither delivered impulse nor joint impact-load consistency
can be qualified here. Undefined impact results are not zero-load successes.

No training or production changes were made.

## Conditions

Selected startup gains: canonical arm
Kp=[256,384,384,256,192,128]Nm/rad, Kd=[6.4]*6Nm·s/rad.
Gripper100/20 and all effort/speed limits unchanged.
Target updates50Hz; torque sampled, clipped, and held at500Hz.
Physics0.25 and0.125ms. Exactly140 torque updates over280ms.

Two old128-world target tapes, with the last source command held after that
world's original termination. These are **frozen-command counterfactuals**,
not closed-loop evaluations of a policy acting on its new observations.
The original commands were learned with a different controller setup.

Eight replays: two sources × two physics steps × additional contact recording
off/on. The built-in contact-row metric remains on in both; the paired check
tests only the additional raw-row recording and all-row reconstruction.

## Measured motion

At the finer0.125ms physics resolution:

| Source tape | J1 peak | J2 peak | J3 peak | J4 peak | J5 peak | J6 peak |
|---|---:|---:|---:|---:|---:|---:|
| .2 policy | 2.628 | 8.493 | 8.857 | 10.748 | 2.458 | 1.046 |
| .3 policy | 2.200 | 9.012 | 8.341 | 9.903 | 2.239 | 1.204 |

All values are rad/s; limit3.1415 for each joint.
J2/J3/J4 each breach in128/128 worlds per source at each resolution.
J1/J5/J6 have no breaches. The same classification holds when excluding
the held post-source-termination tail. Therefore the overspeed is not
merely an artifact of extending the replay, nor caused by hammer–nail impact.

J6 peak over both source tapes is1.2044rad/s.
No sustained saturated-reversal run (at least3 reversals) appears on any
arm joint. This does not prove universal controller stability.

### Numerical agreement

Both source tapes pass the preregistered peak-speed numerical screen
(p95 difference≤.05×3.1415 and maximum≤.10×3.1415rad/s, each joint/source).
The largest paired peak difference is0.04213rad/s, much smaller than the
observed overspeeds. Largest arm-velocity RMS difference at matched poststep
times is0.01302rad/s; largest generalized-position RMS difference is0.000396.
This is agreement between two tested resolutions, not a convergence proof.

## Contact and load observations

- Source.2:1/128 contacts at each resolution; source.3:0/128.
- The one contact is276.75ms at.25ms physics and277ms at.125ms physics.
  Only3–3.25ms remain in the replay: the required50ms window is incomplete.
- No complete first-contact window exists in either the original-source
  prefix or the full counterfactual horizon. All complete-window impulse
  aggregate fields are null in the analysis.
- All-row constraint-force reconstruction maximum error:0.
- Sensor-off samples with nonzero isolated contact-row force:0.
- Every common saved trace and command array is bitwise identical between
  additional-recording off/on, for both sources and both resolutions.
- Actual actuator force equals held torque throughout (maximum error0);
  every2ms torque hold is bitwise constant.

These checks validate plumbing on these traces, **not loaded-contact
measurement accuracy**. The isolated quantity is generalized hammer–nail
contact impulse, not measured internal gearbox load. No old provisional
threshold is transferred to this measurement.

## What this means and the bounded next step

The selected gains remove the screened wrist behavior, but changing gains
does not make the old command sequence safe or effective. This experiment
cannot establish whether an adapting policy can succeed.

Next, perform a **closed-loop feasibility evaluation** under the same
500Hz sampled controller and fixed gains, so commands respond to the
actual changed state instead of replaying an old tape. Keep this clearly
separate from training. Report all-six speed violations, contact/success,
and complete impact windows. A short fresh-training pilot is a later
decision; do not launch long training or retune gains just to hide these
violations. Do not relax speed/effort limits.

## Provenance, tests, and review

- Successful Vega CPU campaign:
  /cephhome/eunikhilr/campaigns/z1-pd500-strike-20260924-v2.oL2EZN
- Job45856697 array0–3: all COMPLETED exit0 (3:38–5:31 per task).
- Initial job45856688 failed before stepping: new canonical-index assertion
  compared identical values with int64/int32 storage. The corrected guard
  normalizes integer storage while still rejecting wrong column order.
  Failed campaign retained at
  /cephhome/eunikhilr/campaigns/z1-pd500-strike-20260924.bABwBP.
- Source cc46d9998a3c9cd5d8813e89b57e44248353e03e;
  assets b58ccd2f81fd246f27c1e8d88cf86484cd888703;
  source/capture/runtime identity gates retained.
- Four new regression cases observed RED→GREEN: column validation and
  complete-window/source-prefix censoring. Final focused run40 passed;
  35 upstream Torch deprecation warnings.
- Project pytest -x:377 passed,1 existing unrelated failure:
  tests/test_docs_current.py::test_index_points_to_current_fic_baseline_and_vic_prototype_routes
  (hard-coded index date). Full-suite pass is not claimed.
- Independent scoped review identified the additional-observer labeling
  qualification, accepted and documented. No remaining scoped major
  implementation concern reported; numerical results computed locally.
- Raw data: .superpowers/sdd/2026-09-24-fullstrike-pd500/results/ and remote
  campaign above. [Compact analysis](assets/2026-09-24_fullstrike_pd500/analysis.json)
  and reproducer scripts are banked beside this result.

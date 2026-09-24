# Fixed-all-arm calibration — 2026-09-23

## Status

New diagnostic screen completed, **no new training**. Production code, old
checkpoint metadata, joint-speed/effort limits and rewards are unchanged.
The two experimental fixed-gain settings are not hardware-qualified values.

Vega job `45831955`, all four arms completed. Two existing seed-2 policies
(impulse-CaT doses .2/.3), each tested with Kp [512,768,768,512,384,256] or
the same vector with Kp6=128; Kd=25.6 on every arm joint. Three speed goals
1.5/1.85/2.2 m/s, 128 first episodes per goal, 1536 first episodes total.
Later episodes written by the evaluator are excluded from these statistics.

| Old policy dose | Fixed Kp6 | Hits / 384 | Timeouts / 384 | J6 speed breaches | Any joint speed breach |
|---|---:|---:|---:|---:|---:|
| .2 | 256 | 341 | 172 | 0 | 384 |
| .2 | 128 | 353 | 157 | 0 | 384 |
| .3 | 256 | 383 | 135 | 0 | 384 |
| .3 | 128 | 383 | 137 | 0 | 384 |

Hits and timeouts are not mutually exclusive: touching the nail does not
guarantee task success before the episode horizon. The evaluator's `complete`
field means an episode record is closed, not nail-driving success.

All 1536 first episodes exceeded J4's 3.1415 rad/s limit; J2/J3 also frequently
exceeded their limits. Worst observed J4 speed was 8.33224 rad/s. Worst J6
speed across the four arms was .70167 rad/s. Effective native gain/bias fields
were checked on every action application; all recorded first-episode gains
also independently matched the requested vectors.

Tracking and impact depend strongly on old policy and goal. Across conditions,
median tracking RMS ranged 19.90–93.77 mm and median delivered impulse
.1544–.4023 N s. These are **posttraining controller interventions**: the
policies were not trained for these gains. They show that changing gains alone
does not solve the whole-arm problem; they do not establish that fresh RL with
fixed gains cannot succeed, or that lower damping alone caused the breaches.

## Startup timestep check: completed

Job45833228 completed both arms, six replays per arm, 60ms each at
2/.25/.125ms. No contact occurred. The existing probe records world0 six-joint
trajectories and all-world J6 peaks: this is not whole-cohort all-joint or
full-strike convergence.

At 2ms, world0 J5 peak was 1.819–1.842rad/s; at .25/.125ms it was
.736–.798rad/s. The difference exceeds the previous startup numerical budget
(10% of3.1415rad/s worst case), so **neither candidate is qualified at2ms**.
Between .25/.125ms, maximum six-joint peak difference was below .0073rad/s
in these world0 probes. J6 remained below .485rad/s with Kp256 and .226rad/s
with Kp128. However J4 still reached3.806–4.044rad/s at .125ms: this part is
not removed by timestep refinement. Several proximal joints saturate their
unchanged effort bounds under the old commands.

Thus distinguish numerical error from aggressive commanded motion. A finer
physics step is a candidate remedy for the former, not a speed safety filter.

## Contact-source measurement check: completed

Job45833213 completed four128-world CPU replays, old **wrist-only** candidate,
two policies x .25/.125ms. Every original saved physical and command array
matched its banked same-dt reference exactly. All-row force reconstruction
error was0; no sensor-off samples had nonzero isolated contact-row force.
Isolated hammer-face/nail generalized force was summed across contact rows,
then absolute-valued and integrated in the same50ms sliding window. Reported
peaks exclude postterminal held-command tails.

Worst J6 difference between the two timesteps, in N m s:

| Source policy | Current baseline-subtracted measure | Contact-only measure |
|---|---:|---:|
| dose .2 | .002658 | .000366 |
| dose .3 | .009098 | .000823 |

All six contact-only quantities fall within the previously used numerical
budgets (p95<=.05, maximum<=.10, normalized by old provisional caps). J6
p95 differences are .018827/.013752 and maximum .022320/.050193. The old
caps here are **only fixed normalization scales**, not validated thresholds
for this different measurement. No safety classification is transferred.

This supports contact-row isolation as a substantially more numerically
consistent candidate measurement on these traces. It measures external
generalized contact impulse, **not internal gearbox loads**. Production
measurement has not been replaced; training-scale performance and fresh
controller/contact tests remain to be checked.

First attempt45833025 failed before stepping due to an obsolete remote
helper directory. Corrected dependency hashes match the inspected tools.

## Decision and next step

No new training at the existing2ms configuration. Do not raise speed caps or
reinterpret old checkpoints. Next bounded step: fixed-command **full-strike**
check for an all-arm candidate at .25/.125ms, keeping20ms control and50ms
windows. If physical motion and delivery agree, use that configuration for
a fresh fixed-gain, speed-constrained feasibility pilot. Keep the unresolved
impact-load constraint log-only until the measurement and its thresholds are
explicitly bound to the new definition. This is not a demand for zero speed
violations from an old-policy transfer before new learning is allowed.

Fourteen new focused tests and48 existing variable-impedance/contact-row tests
passed (36 upstream/deprecation/profiler warnings in the latter). A broad local
suite attempt ended without a final summary at21%; no full-suite pass claimed.
Independent Astra read-only review found no critical/important code issue in
these bounded wrappers; it did not independently recompute remote statistics.
Deferred minors: compare native gains to frozen requested constants as well
as live buffers; hash every executed observer dependency, not just entry files.
Retain prior scratch/raw evidence rather than deleting it: scientific
reproducibility takes priority over the workflow cleanup convention.

Compact evidence is banked in `assets/2026-09-23_full_arm_calibration/`.
Full raw evidence currently in `.superpowers/sdd/2026-09-23-full-arm-calibration/results/`
and remote `/cephhome/eunikhilr/campaigns/z1-full-arm-calibration-20260923.PIMeqR/`.
Source cc46d9998a3c9cd5d8813e89b57e44248353e03e; assets
b58ccd2f81fd246f27c1e8d88cf86484cd888703. Plan:
`docs/superpowers/plans/2026-09-23-full-arm-calibration.md`.

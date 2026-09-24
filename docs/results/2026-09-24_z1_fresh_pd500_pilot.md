# Fresh fixed-gain500Hz training pilot — training and evaluation complete

User approved a fresh250-iteration pilot after old-policy transfer retained
contacts but exceeded J2–J4 speed limits. Old-policy success is not a launch gate.

## Experiment

- One fresh training seed2,1024 environments,250 PPO iterations on Vega.
- Arm Kp canonical J1–J6:256,384,384,256,192,128; Kd6.4 for each.
- Policy50Hz; explicitly sampled/held torque500Hz; physics4kHz.
- Speed limit3.1415rad/s on all six joints, effort limits30/60/30/30/30/30Nm unchanged.
- Reward functions/weights unchanged; impulse-CaT dose0.2, previous provisional caps retained.
- External speed conditioning1.5–2.2m/s, event cost0.8493676. This is NOT yet autonomous speed selection.
- The13-output interface is retained. Six stiffness outputs are ignored physically;
  this pilot is fixed impedance, not learned variable impedance.
-50ms reward/impulse windows become200 physics substeps. Two-sample onset-speed
  validity guard remains two samples; onset finite difference now spans0.25ms.

## Verified before launch

- Vega reward phasesA–M, contact-sensor check and reward-setup check passed.
-207 targeted tests passed on Vega.
- Actual-profile GPU check verified torque equality,8-substep hold and reset timing.
- Separate two-iteration smoke completed and strict checkpoint reload passed.
- PT/ONNX identify the sampled fixed controller; mismatched native-controller
  checkpoints are rejected. Metadata checks live gains and physical windows.
- Independent OpenAI review found no remaining launch blocker after two provenance
  fixes. No GLM/Qwen/Gemini review obtained: OpenCode model catalog was empty.
- Full local regression was interrupted by the laptop shutdown, with no complete
  result recovered. The pre-existing docs-date assertion failed in its partial
  output. The new controller tests passed separately locally and in the207-test
  Vega gate. Full Vega regression45858771 finished FAILED exit1 after43min34s:
  3458 passed,287 failed,4 skipped. All9 sampled-controller tests passed.
  This is NOT a full-suite-pass claim.
  Follow-up reran286 failures (excluding the native variable-impedance test)
  with system Git and inherited `module`/`BASH_ENV` removed:285 passed,1 failed
  in407.02s. No source/test edits were made for this rerun. Module Git lost
  libiconv under sanitized test environments; shell startup could also override
  fixture Git. Remaining issues, outside this fixed-gain pilot:
  - Historical PNG renderer test assumes filesystem enumeration order. Both
    expected files exist, but the order is reversed.
  - Native variable-impedance test requires bitwise equality between two
    algebraically equivalent float32 damping calculations. A standalone check
    reproduces the exact1.52587890625e-5 difference. Implementation/test are
    unchanged; this pilot overwrites those gains with its selected fixed gains.
  These issues remain recorded rather than modifying unrelated tests to claim green.

## Provenance

Source:cc46d9998a3c9cd5d8813e89b57e44248353e03e plus opt-in sampled-controller patch.
Remote campaign:`/cephhome/eunikhilr/campaigns/z1-fresh-pd500-20260924.sbkVqV`.
Smoke job45857822 completed exit0; fresh250 job45857934 completed all250 learning
iterations, strict final-checkpoint reload and evaluation ongn19.
Final checkpoint SHA256:`0de912ae08faf988454c3cc561da62abc0df4bc744940383c870c97466377815`.
Smoke checkpoint SHA256:`8f8a693fec209358d050794341fd8da15f11b1b132ab8eb7595b918f7773eda7`.

The full run is fresh, not an extension of smoke. Completed evaluation uses two
held-out seeds2026092401/2026092402,128 first episodes per goal1.5/1.85/2.2.
Report delivered impulse, contacts/productivity, tracking and all-six true
substep speed peaks, retaining terminal steps and excluding reset-stream episodes.
Training/evaluation job45857934 completed exit0 in50min39s. Video job45858251
completed exit0 in24s; montages visually inspected and show the red reference
curve and hammer motion. Videos are illustrative single-world first episodes,
not a substitute for the128-world quantitative evaluations.

## Held-out results

One trained policy, deterministic inference, two held-out seeds, three paired
external speed goals,128 first episodes per seed/goal:768 evaluated episodes
over256 reset cases. All768 had native productive contact, zero measured joint
speed-limit violations, zero episode censoring, and zero provisional joint-impulse
threshold exceedances. Productive contact means positive nail progress; it is not
by itself a task-completion or hardware-safety certificate.

Pooled medians over256 episodes per goal:

| External goal (m/s) | Measured onset speed (m/s) | Tracker impulse (N·s) | Geometric tracking RMS (mm) |
|---|---|---|---|
|1.50|1.615|0.336|12.56|
|1.85|1.873|0.384|15.91|
|2.20|2.205|0.318|19.11|

Observed maximum joint speeds across every evaluation, canonicalJ1–J6:
`[1.01464,3.01409,2.97860,2.89236,1.60031,1.95007] rad/s`.
The closest approach to the3.1415rad/s limit was J2, with0.12741rad/s margin.
The original J6 overspeed/chatter finding did not recur in these measured
first episodes. This is evidence for the new trained setup, not proof that the
old controller was safe or that untested initial states cannot violate limits.

### Interpretation limits

- The tracker finalizes at success or its50ms window. Its recorded impulse is
  success/window-censored, not necessarily the complete physical collision
  impulse. Do not interpret the middle goal's larger median as an optimal speed.
- These are paired evaluation conditions from one training seed, not768
  independent training results. No generalization probability or guarantee is claimed.
- The tested policy is deterministic. Training exploration and other disturbances
  are not covered by the zero-violation evaluation count.
- Speed is externally conditioned and impedance is fixed; this is not yet
  policy-selected terminal speed or learned variable impedance.
- Tracking error is geometric reference RMS over sampled control steps, including
  contact/followthrough; it is not a clock-timed trajectory-tracking guarantee.

## Next decision

Proceed to two additional independent training seeds with this exact setup to
test reproducibility; do not alter gains/rewards merely to improve this result.
No additional policy run has been launched. Before hardware/load-safety claims,
validate the joint-load measurement and physical thresholds; before ranking
impact speeds by full delivered impulse, compare a common uncensored window.

## Banked evidence

- `assets/2026-09-24_fresh_pd500/train250/model_249.pt` and `completion.json`:
  final checkpoint/hash and strict reload contract.
- `assets/2026-09-24_fresh_pd500/train250.log`: exactly250 learning iterations.
- `assets/2026-09-24_fresh_pd500/evaluation/`: all768 raw episode rows and six-condition summary.
- `assets/2026-09-24_fresh_pd500/video/`: three goals, real-time and4×slow video, montages and provenance.
- Preflight logs, source hashes and campaign scripts under the same assets folder.
- Independent OpenAI reviewer recomputed all raw-row counts, maxima and medians;
  no discrepancy found. External GLM/Qwen/Gemini remained unavailable.

Joint-impact-load safety remains unqualified: the retained constraint measurement
and provisional thresholds are not manufacturer damage limits or validated
internal gearbox loads. A passing speed screen alone cannot establish load safety.

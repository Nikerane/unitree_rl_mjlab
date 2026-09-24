# Z1 bounded wrist-gain screen

Status: native screen complete; contact timestep validation pending. This is an exploratory frozen-policy intervention, not a hardware-safe envelope or learned-VIC improvement.

## Execution and validity

Vega array **45735252**, all ten arms COMPLETED 0:0 (19–45 seconds each). Two existing seed2 policies (impulse-CaT maximum probability 0.2/0.3), five fixed wrist gain pairs, three external speed goals 1.5/1.85/2.2 m/s, 128 first episodes per condition. No new learning, production changes or safety-cap changes. First five joints retain policy-controlled impedance. Source cc46d99 and asset b58ccd2 pinned and clean at launch; 2 ms physics, 20 ms control.

All 3,840 evaluated first episodes hit and complete, with no timeouts. Every candidate passes the preregistered screen utility guard (≥116 hits/128, ≥90% own-original median delivered impulse, median tracking RMS≤21.622820 mm, separately at every goal). This permissive screen guard is not qualification.

Checkpoint, reset, CPU/CUDA RNG, initial proposed-action and controller-contract identities match banked originals and SDK-reference runs. Every trace row has the requested joint6 gains; first-five gains match the native exponential map of that row's own action. The center setting reproduces all banked SDK per-joint exceedance counts and hits exactly; maximum absolute median delivered-impulse difference is 0.0000503510 N·s. Traces need not be byte-identical across GPU runs.

Runner review: gain_runner_review approved spec and task quality. Separate actuator_sources review approved the launch and corrected analyzer. Pure-runner tests:12 passed. Whole-repository test run was stopped after997 passed/1 skipped and is **not** a full-suite pass.

## Main result

Numbers are episodes with **any joint** exceeding either its configured impulse proxy cap or the 3.1415 rad/s velocity cap. Within each cell, goals are ordered1.5/1.85/2.2; denominator128 each. Joint-impulse caps remain provisional simulation thresholds.

| Fixed wrist Kp / Kd | Dose0.2: any-load counts | Dose0.3: any-load counts |
|---|---|---|
|128 /25.6|3 /11 /15|3 /0 /12|
|256 /12.8|19 /24 /48|6 /6 /24|
|256 /25.6 (SDK-derived reference)|16 /29 /35|6 /1 /21|
|256 /51.2|4 /16 /21|5 /0 /16|
|512 /25.6|11 /33 /42|39 /54 /44|

The **128/25.6 point** is the most promising common candidate in this small cross: fewer whole-arm exceedance episodes at all six comparisons against256/25.6 (44/768 versus108/768 pooled descriptively), with preserved strike utility. Wrist impulse exceedances are15/768, versus75/768 at the reference and766/768 in the original controller; wrist velocity exceedances are0/768, versus768/768 originally. This does not establish a continuous safe gain range or stiffness/damping interaction surface.

## Candidate utility, per condition

All rows128/128 hits and completions. Delivered values are native first-strike **success-censored** axial sensor impulse, not a full uncensored contact-window integral.

|Dose|Goal m/s|Median delivered N·s|Retention vs original|Median tracking RMS mm|Median absolute speed error m/s|
|---|---|---|---|---|---|
|0.2|1.5|0.306223|101.87%|10.976|0.19381|
|0.2|1.85|0.367579|104.08%|10.295|0.09702|
|0.2|2.2|0.384602|106.29%|10.353|0.07584|
|0.3|1.5|0.305265|103.64%|11.942|0.09119|
|0.3|1.85|0.305957|96.04%|11.209|0.05528|
|0.3|2.2|0.386197|106.25%|9.779|0.05165|

Neither policy passes the existing all-goal0.07 m/s speed-error target. Softer wrist gains do not solve speed selection/tracking by themselves.

## Remaining load tradeoffs — do not hide these

- Dose0.2 high-goal J2 impulse exceedances rise from5/128 original to10/128 candidate (SDK9/128); worst J2 ratio rises from about1.092 to1.151. Dose0.3 high-goal J2 counts rise3→7/128 (SDK9/128), worst ratio1.042.
- Candidate J6 impulse counts: dose0.2 **1/4/4**, dose0.3 **0/0/6**. Largest candidate wrist ratio is1.583 at dose0.2 middle goal. No wrist speed violations, but not zero wrist load violations.
- Candidate velocity exceedances: dose0.2 low J4=2; middle J2=6 andJ4=1; highJ4=1. Dose0.3 lowJ4=3; middle/high none. Maximum candidate J4 speed is3.80245 rad/s at dose0.2 low goal. Lower aggregate counts do not imply every tail metric improved.
- One trained seed per dose, multiple reset worlds rather than training-seed replication. No hardware test, fatigue model or certified actuator range. Other joints' gain ranges unchanged. The reaction proxy is contact-gated absolute baseline-subtracted generalized constraint force integrated in a50 ms sliding window, not literal harmonic-drive damage.

## Next bounded check

Capture native processed targets and all six gains, then replay the original controller and128/25.6 candidate at2/1/0.5 ms with the same command tape and physical window durations. First prove native2 ms replay matches its source prefix. Start with both doses/high goal, explicitly include success-censored and passive uncensored50 ms delivered impulse and whole-arm load tails. Dose0.2 middle-goal wrist tail merits follow-up. Keep post-termination hold separate as a counterfactual. A numerically unresolved result is legitimate; do not force a safety or retraining endorsement.

Artifacts: [full analysis](assets/2026-09-22_wrist_calibration/analysis.json); per-arm summaries/traces/episodes/intervention notices in the same asset directory. Experiment scripts and review records retained in `.superpowers/sdd/2026-09-22-wrist-calibration/`. Goal remains active pending contact replay and final evidence review.

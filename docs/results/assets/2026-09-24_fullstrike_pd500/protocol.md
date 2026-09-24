# Approved full-strike diagnostic — 2026-09-24

User approved the full-strike follow-up to the seven-setting startup sweep.
Fixed canonical Kp=[256,384,384,256,192,128], Kd=6.4 each arm joint.
Gripper100/20 unchanged. Effort caps and3.1415rad/s speed limit unchanged.
Policy targets50Hz, sampled/clipped/held torque500Hz, physics .25/.125ms.
Horizon280ms:14 target ticks and140 PD ticks. No resets/learning/production edits.

Replay both128-world existing command tapes, retaining final source command
after each source termination. These counterfactual tails are labeled and
excluded from source-prefix comparisons. They are not closed-loop evaluations.
Count contacts and complete50ms first-contact windows explicitly. Do not
classify incomplete windows as zero impulse; misses are distinct.

Two identical runs per source/resolution, with and without the additional
read-only contact-row recording (eight replays). The existing built-in
contact-row metric stays enabled in both; this tests only the additional
recording/reconstruction, not all contact observation versus none.
Require bitwise equality of
every common original trace and command array. Reconstruct all-row qfrc and
compare to native constraint force; report maximum discrepancy. Isolated
hammer-face/nail generalized force is not internal gearbox load.

Report six-joint speed maxima/counts, saturated reversals, saturation, first
contact times,50ms delivered impulse only for complete windows, and per-joint
absolute contact-row impulse over the same window. Report numerical differences
between resolutions on paired complete-contact worlds, plus censoring and
onset mismatch counts. Source-prefix and full counterfactual horizon differ.

Speed numerical screen remains per-joint/source p95 gap<=.05*3.1415 and
maximum gap<=.10*3.1415rad/s. Report joint-load differences in physical units,
without transferring old safety thresholds to this different measure.
If contact/physical results disagree, no fresh training recommendation based
on this diagnostic. No numerical or hardware safety claim from absence of
startup chatter. No adaptive gain search or automatic training in this step.

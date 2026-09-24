# Closed-loop feasibility diagnostic

Evaluate the two existing dose.2/.3 actors, no learning. Validate/load each
actor against its unchanged original2ms/decimation10 training environment
and original checkpoint contract. Then retain that actor for a separately
labeled diagnostic environment; never rewrite or weaken the checkpoint contract.

Diagnostic: same task,128 worlds, seed2026092101, external speed goal2.2m/s
(same as prior tapes), bounds1.5–2.2, cost.8493676, dose.2/.3 unchanged.
Physics .25/.125ms; targets50Hz; sampled/clipped/held torque500Hz.
Fixed Kp=[256,384,384,256,192,128], Kd6.4; gripper/limits unchanged.
Policy outputs joint targets anew each20ms from actual observations. Its
stiffness outputs are overridden by the stated fixed-gain intervention;
speed remains externally supplied, matching these checkpoints' phase.
This is not a learned-impedance or policy-selected-speed qualification.

Preserve native reset/termination/reward handling. Score only each world's
first episode, bounded to500ms (25 action steps); mark surviving episodes as
censored, not failures or successes. Observe all physics substeps, including
the terminal control step, before autoreset. Do not mix subsequent episodes.
Report first-episode contact, native outcome, six-joint speed maxima,
saturation/chatter, complete50ms contact windows, and tracking.
Compare initial evidence and both physics resolutions. No manufacturing or
hard-safety claim; retain provisional-load/measurement caveats.

Four CPU jobs: two actors x two resolutions. Preserve prior evidence.
Bounded first launch; failures stop for diagnosis, no automatic training.

# Independent review request: simplified Z1 impulse-CaT experiment

Review this experiment plan independently and bluntly. Do not assume the proposed next step is
correct. Distinguish observed evidence from inference, and do not invent actuator damage limits.

## System and evidence

- Simulated Unitree Z1 hammering task with a six-joint variable-impedance PPO policy.
- Velocity soft-CaT is active. Impulse soft-CaT uses baseline-subtracted, contact-masked joint
  reaction impulse accumulated at 500 Hz over a rolling 50 ms window.
- The two pressures combine as `max(delta_velocity, delta_impulse)`.
- Provisional project caps are `[0.82, 1.64, 0.82, 0.82, 0.82, 0.82] N.m.s`.
- Diagnostic-only caps are `0.9` times that vector. Neither vector is a manufacturer or hardware
  damage limit, and soft-CaT is not a runtime clamp.

One matched seed-2 training experiment used the diagnostic caps for 500 PPO iterations:

- control: `imp_max_p=0`;
- target: `imp_max_p=0.5`;
- both otherwise used the same task and velocity CaT.

Frozen native-horizon evaluation used one fixed 64-world population and three paired stochastic
4096-world populations. Descriptively pooled results were:

- provisional-cap episode risk: `0.814% -> 0.008%`;
- diagnostic-cap episode risk: `1.261% -> 0.057%`;
- true 500 Hz velocity-limit episode risk: `0.618% -> 1.058%`;
- task success: `100% -> 100%`;
- target/control first-event delivered impulse ratio: about `1.02`;
- native median episode duration: `140 ms -> 100 ms`.

Impulse pressure was nonzero and strictly won the max soft-OR on every cap-active read observed at
offline `p=0.5`. This is one PPO training seed; the three evaluation populations are not three
training seeds. Contact often remained active at native success, so completed-event impulse is
uncertain, but native episode endpoints remain directly observable.

## Proposed simplified next step

1. Treat native episode impulse risk/tail, true velocity risk, and hammering utility as primary.
2. Make post-success contact/window flushing optional sensitivity analysis, not a prerequisite.
3. Use offline replay only to confirm that `p=0.2` would not be completely masked by velocity;
   do not claim that replay selects an optimal learning dose.
4. Train exactly one new 500-iteration seed-2 target using the provisional caps and
   `imp_max_p=0.2`.
5. Reuse the banked seed-2 `p=0` control only after an exact test reconfirms that log-only cap
   values cannot alter actions, rewards, PPO data, or policy updates.
6. Evaluate native fixed-64 and the same three paired 4096-world populations.
7. Stop if impulse does not improve, velocity risk rises beyond the preregistered margin, or task
   utility degrades. Do not automatically try a higher dose.
8. Only after the bridge passes, run matched independent training seeds for the thesis claim.

Candidate screen thresholds are: upper paired one-sided 97.5% bound for provisional episode-risk
difference below `-0.50` percentage points; upper bound for true-velocity risk difference at most
`+0.10` percentage points; success and productive-strike differences no worse than `-1` percentage
point; and lower bound for delivered-impulse ratio at least `0.90`. These are experiment criteria,
not safety limits.

## Required response

Return no more than 700 words:

1. `VERDICT: PROCEED | REVISE | REJECT`
2. What has actually been established?
3. What is the most serious unsupported claim or confound?
4. Should contact/window flushing be primary, optional, or dropped?
5. Is one predeclared `p=0.2` bridge the smallest decisive next experiment? If not, give one
   simpler alternative.
6. Are the stop thresholds defensible for a screening experiment?
7. State the single most important reason this plan could mislead us.

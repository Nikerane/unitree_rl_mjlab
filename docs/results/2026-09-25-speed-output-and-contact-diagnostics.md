# Speed-output and contact diagnostics

## Material Passport

- Origin: academic-research-suite / experiment-agent, validate
- Date: 2026-09-25
- Status: ANALYZED for existing results; contact-aligned replay pending
- Source: completed LEFT pilot, `/cephhome/eunikhilr/campaigns/z1-impact-speed-retry-20260924.IGsdQz/runs/run-{0..5}`; first episodes only, common-window protocol, 384 attempts/run.
- No new empirical safety qualification. Three training seeds and one evaluation seed; no independent training reproduction.

## What the policy actually outputs

The architecture has 13 actor coordinates: six joint targets, six impedance requests and one desired downward impact speed. `scripts/vega_impact_campaign.py:26` sets `speed_mode="external"`. In that mode, `ImpactSpeedState.reset` samples a supplied goal and sets pending=0; `select` cannot replace the goal. `ImpactSpeedModel` masks the selector out of PPO log likelihood, entropy and KL when pending=0. A numerical 13th output still exists but it is not a learned autonomous speed decision in this pilot.

Saved first-episode records confirm zero selections and null `selected_raw` for every run. In policy mode pending=1 at reset, the first speed action is clamped/mapped to [1.5, 2.2] m/s and latched once. It is a desired speed, not direct velocity actuation or a prediction of the achieved speed. The newly selected goal becomes available to subsequent control decisions; the same first motor command is computed before that selection. Actual speed is a separate first-contact onset finite difference projected onto the downward axis, clipped below at zero (`first_strike.py`). It is not full 3D velocity magnitude.

## Existing speed accuracy and load trade-off

Each row pools 128 first episodes from each of three trained seeds. The ±0.1 m/s band is a descriptive post-hoc diagnostic, not a preregistered qualification threshold.

| Mode | Goal (m/s) | Measured mean (m/s) | Within ±0.1 m/s | Any joint-load exceedance | J2 exceedance | J6 exceedance |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 1.50 | 1.193 | 18.49% | 55.21% | 8.59% | 51.82% |
| Fixed | 1.85 | 0.962 | 0.26% | 42.45% | 10.16% | 35.42% |
| Fixed | 2.20 | 1.143 | 0.52% | 44.01% | 14.32% | 33.33% |
| Learned | 1.50 | 1.090 | 11.20% | 27.34% | 4.69% | 26.30% |
| Learned | 1.85 | 1.246 | 11.98% | 56.25% | 19.53% | 48.18% |
| Learned | 2.20 | 1.838 | 20.05% | 55.21% | 33.33% | 42.97% |

The learned policy responds more strongly in pooled means, but this is not accurate or uniformly monotone goal tracking across individual seeds. Its improved high-goal speed accompanies higher J2 exceedance. This is association within a learned policy response, not proof of which mechanical term causes the violation.

The event speed cost is `min(((measured-goal)/0.7)^2, 1)` times coefficient 0.8493676. Errors of at least 0.7 m/s receive the same saturated cost. Among learned-policy records the saturated proportions are 15.89%, 45.05%, 12.76% for the three goals; fixed-policy proportions are 5.21%, 67.97%, 74.22%. This can make large errors indistinguishable to that reward component; it does not prove the source of training failure or eliminate other policy-gradient learning signals. No cost change is made here.

## What impedance averages establish—and do not

Fixed Kp is [256,384,384,256,192,128], Kd=[6.4]*6. Learned per-attempt episode-average Kp across all seeds is approximately [255.50,358.73,356.43,224.00,191.05,131.11]. The largest mean reduction is J4 (about 12.5%), followed by J3 and J2. The gain request is bounded to a 0.8–1.25 stiffness multiplier; damping is coupled by the square root. These averages establish effective gain use, not timing relative to contact. Averages cannot establish pre-impact softening, reactive softening, torque saturation, or a causal explanation for J2/J6 loads. Those require the approved trace replay.

## Attribution of code changes

The paired pilot changes fixed versus variable sampled-gain execution while retaining the same path, external speed range, controller clocks, plant and safety settings. It measures the performance of policies trained in those two treatments. It does not isolate a particular joint's gain effect, separate gain changes from learned joint-target changes, or establish a before/after effect for every feature added to the branch. Both arms already include the speed-goal observation/cost and LEFT spline guidance; there is no no-speed-head or no-spline ablation. The three-route task and autonomous selector were implemented but untested by this batch.

The precision fix was test-only; it did not change controller behavior. Common-window measurement extends observation beyond native success termination and reports more violations in aggregate. However, separately executed protocols do not have identical physical prefixes despite matching initial reset hashes. Across the three fixed seeds, 42 native violators become common-window nonviolators and 109 change in the reverse direction; for learned impedance the corresponding counts are 17 and 201. The net increases of 67 and 184 therefore cannot be attributed exclusively to post-success loading. Timing attribution requires the same-rollout native-success marker and threshold crossings in the approved recorder. Soft-CaT discounts positive reward and future value; it is not a hard online shield that guarantees load caps. Existing caps remain unchanged.

## Approved next work

Evaluation-only physics-cadence recorder and bounded replay of all six saved policies: record gain requests/applied gains, torque demand/clipping, velocity and authoritative loads; align first episodes around accepted contact, retain failures/misses and compare observer-on/off dynamics. One GPU job, 30-minute cap, no learning. Then bounded paired external/autonomous three-route pilot, with per-route evidence and explicit selector activation checks. No claim that desired speed equals achieved speed.

Statistical caution: inherits the 11-type fallacy scan from `2026-09-24-z1-impact-speed-comparison.md`; added goal strata expose aggregation effects, post-hoc accuracy bands are labeled, episode means are not temporal/causal evidence, and no significance tests or hardware extrapolation are made.

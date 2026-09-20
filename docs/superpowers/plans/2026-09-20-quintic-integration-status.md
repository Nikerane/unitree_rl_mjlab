# Quintic pilot integration status

Task: `Unitree-Z1-Hammer-Quintic-M20-Pilot`.

Implemented opt-in quintic geometry in the existing shared strike reference. The curve starts at the realized −20 mm pose, passes through the frozen nail top, then continues vertically for 20 mm. Progress is monotone height-indexed, gated by distance from the corresponding curve point; it is not nearest-curve arc-length projection. Reward preview remains read-only. Existing task IDs retain straight geometry and defaults.

No prescribed time or impact speed enters learned-policy observations/rewards. The 12-dimensional joint-position/stiffness actions, tracking weight 0.2, sigma 0.05 m, and CaT p=0.2 are inherited. This single-curve stage uses the existing local phase/error observations; future-reference conditioning for a multi-curve policy is not implemented here.

The video trace helper now samples the quintic rather than drawing the chord between endpoints. The legacy straight-reference provenance guard accepts the added optional constructor parameter only while its default remains straight, preserving the historical spec/digest fields.

Verification:

- New geometry/reset tests failed before implementation; pilot-registration test failed before registration; overlay test failed on two-endpoint rendering before correction.
- 186 focused tests passed (reference, geometry, direct-reference provenance, impact rewards, impulse and soft-CaT).
- Live CPU environment reset and three steps passed with 12 actions and a quintic reference.
- Initial CPU smoke computed one update with 4 environments × 8 steps but exited nonzero on checkpoint saving: existing VIC telemetry requires a 24-step rollout. This is not a successful end-to-end training run. A corrected integration check preserves the required 24-step length, uses one PPO epoch/minibatch, and writes to `/tmp/quintic-integration-one-update-24`. It is not a scientific policy result.
- Corrected 24-step smoke completed and exited zero, including checkpoint saving. The run uses four CPU environments and a single iteration; it only qualifies the plumbing.
- Independent review identified registration ordering and renderer bypass; both were corrected. Remaining qualifications: implicit timing, no future-curve preview, and no orientation qualification.
- Full-suite run was started before implementation and remains tracked separately in `/tmp/spline-full-baseline.log`; do not treat this document as an all-suite-pass claim.

Open issue: scripted position-only execution contacts near the nail rim. A failed reset-attitude IK diagnostic does not establish global orientation infeasibility. Keep first-contact location, impulse, speed, joint loads and tracking jointly visible in pilot evaluation. No GPU campaign launched; no commit or push made.

## Final-code qualification

The clean final-production-code full suite completed: 3611 passed, 3 failed, 4 skipped (35m13s). The three failures were two outdated test inventories: the approved registry list omitted the new pilot, and the joint-policy family list omitted it, causing train/play cases to misclassify it as Cartesian. Added exactly one pilot entry to each list, without altering production code or weakening assertions. Complete affected config files plus the new quintic tests then passed: 203 tests. All observed failures are resolved; a second all-suite run after these test-only edits was not performed. Do not label the first run as wholly green.

Fresh reward phases A–M, contact-sensor verification, and random-policy reward setup also passed. GLM 5.3 and Qwen 3.7 Max completed external reviews with no supported blockers after source clarification; see `2026-09-20-quintic-external-review.md`. Training submission is a separate recorded step.

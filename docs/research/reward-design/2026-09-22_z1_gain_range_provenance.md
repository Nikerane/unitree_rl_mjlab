# Z1 gain range: provenance versus hardware evidence

Research only, 2026-09-22. No controller changes.

## Why the current wrist range is 800–1250

The simulator nominal wrist gains are Kp=1000, Kd=100, inherited from original scene XML, as documented in `src/assets/robots/unitree_z1/z1_constants.py`. This provenance does not establish manufacturer calibration.

The [August 13 prototype design](../../superpowers/specs/2026-08-13-z1-native-vic-design.md) deliberately froze C=1.25 as a conservative *relative perturbation* of the existing fixed controller. The implemented map is Kp=Kp0*C^p, Kd=Kd0*sqrt(C^p), p∈[-1,1]. Hence wrist Kp∈[800,1250], Kd∈[89.443,111.803]. Calling the perturbation conservative does not prove the absolute gains are physically appropriate or numerically benign.

The earlier [gain-contract research](../wayfinder/vic_gain_contract.md) explicitly says the paper supplies no transferable Z1 numeric bounds and proposes qualifying C=1.25,1.5,2. The prototype selected only 1.25; this is not evidence that 1.25 was the largest feasible envelope. Later impact-speed work preserved that mapping for comparison continuity. The default/map was introduced in commit `5e74e08`.

## Official Unitree reference, with unit conversion

[Unitree's Z1 SDK header](https://github.com/unitreerobotics/z1_sdk/blob/master/include/unitree_arm_sdk/message/LowlevelCmd.h) documents default low-command kp=[20,30,30,20,15,10] and kd=[2000,2000,2000,2000,2000,2000]. These are SDK/protocol numbers, not directly MuJoCo gains.

[Official Z1 SDK documentation](https://dev-z1.unitree.com/use/introduction.html) gives torque=25.6*kp*position_error + .0128*kd*velocity_error + feedforward. Applying this documented conversion yields effective stiffness [512,768,768,512,384,256] N·m/rad and damping 25.6 N·m·s/rad for all six joints. These derived values interpret the documented defaults; they are not measured gains on our physical arm or a manufacturer-certified admissible range.

Thus the documented SDK wrist default corresponds to Kp=256, Kd=25.6, versus our current minimum 800,89.443. This is a meaningful calibration discrepancy to investigate, not proof the SDK default is suitable for our hammer/payload or a blanket instruction to copy it.

No manufacturer-certified per-joint safe min/max stiffness envelope was established from the official material inspected. Protocol representability, an SDK default, stable simulation gains, and hardware-safe task-dependent gains are different concepts. Confirm firmware/control mode, payload and supported bounds with Unitree before hardware use.

## Design implication

Widening C alone lowers the minimum **and raises the maximum**. With nominal1000, reaching256 would require C≈3.906, also permitting Kp≈3906. That is not a justified way to provide only more softening. First compare validated lower-gain candidates in isolated no-learning diagnostics, with appropriate damping and intact force limits. A lower nominal or asymmetric/per-joint bounds would be a separately designed action-contract change and would require matched evaluation/new training, not silently altered old-checkpoint semantics.

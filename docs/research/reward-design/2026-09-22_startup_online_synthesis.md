# Online/source cross-check: startup oscillation

Research only, 2026-09-22. No new jobs or controller changes. Complements the [matched hold diagnostic](../../results/2026-09-22_z1_startup_hold_diagnostic.md).

## Strongest lead: saturated native PD and timestep sensitivity

Our task uses native `BuiltinPositionActuatorCfg`, not Python torque-level PD. `z1_constants.py` configures J6 nominal kp=1000, kd=100, effort limit=30 N·m, armature=.01. The observed VIC startup uses kp=800 and kd≈89.44. The .0551646-rad initial position error at zero velocity requests about 44.13 N·m before force clipping. These are simulator parameters, not newly verified hardware specifications.

Installed mjlab `sim/sim.py` defaults to implicitfast; hammer config supplies timestep .002 without overriding that integrator. Variable impedance writes native actuator gain/bias arrays each action application. Therefore neither “switch from Euler” nor “replace opaque Python PD” is an evidence-based diagnosis here.

[MuJoCo Warp v3.8.1 derivative source](https://github.com/google-deepmind/mujoco_warp/blob/v3.8.1/mujoco_warp/_src/derivative.py) explicitly sets the actuator velocity derivative to zero when force reaches either configured force bound. The installed Vega source has this same branch (lines101–107). Thus a saturated actuator contributes no local velocity derivative to the implicit update. This is not equivalent to removing the damping term from the underlying PD law: the total PD force is still computed and clipped. This distinction is essential.

The first-step torque demand and alternating ±velocity trace make saturation-dependent integration a strong hypothesis, not a confirmed cause. Applied actuator force and solver inputs were not recorded in the hold probe. Whole-arm coupling remains another contributor.

## What the primary sources support

- [MuJoCo 3.8 computation](https://mujoco.readthedocs.io/en/3.8.0/computation/index.html): implicitfast handles velocity derivatives, not an unconditional guarantee for stiff position feedback. Numerical stability depends on the model and integration settings.
- [MuJoCo modeling: force limits](https://mujoco.readthedocs.io/en/stable/modeling.html#actuator-force-clamping): control limits, actuator-force limits and summed joint-force limits have different semantics. Changing torque limits changes the model; it is not a neutral numerical repair.
- [Warp derivative source](https://github.com/google-deepmind/mujoco_warp/blob/v3.8.1/mujoco_warp/_src/derivative.py): the clamp-dependent derivative branch above is directly inspectable and matches installed code.
- A [public MuJoCo discussion](https://github.com/google-deepmind/mujoco/discussions/3498) raises a similar force-saturation/implicit-damping question. This is corroborating motivation only, not a maintainer-confirmed diagnosis of our problem.

Latest online docs contain newer actuator/integrator features; do not assume they exist in our installed stack or upgrade dependencies on that basis.

## Recommended bounded diagnostic, not a patch

1. Record actual compiled integrator, actuator gains/biases, force ranges, gear, armature and damping. Trace unclipped PD demand, applied actuator force, position and signed velocity at every physics step through the first command, before contact. Confirm time alignment: preintegration force must be compared with the state that generated it, not a later refreshed position.
2. Replay the identical initial state and fixed command/gain schedule at 2, 1 and .5 ms physics steps, keeping the physical observation horizon and 20-ms controller interval fixed. If alternating response/peaks change strongly or converge with smaller steps, numerical resolution is implicated. Re-running the feedback policy is a separate test because its resulting actions can change.
3. Independently reduce initial command magnitude at the original timestep/gains, keeping torque limits intact, to test whether entering saturation is decisive. Separate wrist-only and whole-arm changes if attribution requires it. Smaller motion also changes physical demand, so this does not alone prove a numerical defect.
4. Only after the above select a correction. Do not automatically lower gains, inflate armature, remove torque limits, smooth commands permanently, or retrain.

The current checkpoint contract explicitly requires .002-s physics timestep (`rl/runner.py`), so a timestep study needs a clearly separate open-loop diagnostic harness. Do not disable strict loading and present altered-timestep evidence as native checkpoint qualification. No such study has run in this research turn.

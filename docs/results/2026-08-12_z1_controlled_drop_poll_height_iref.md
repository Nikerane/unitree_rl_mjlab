# Z1 controlled-drop hammer-poll-height impulse-reference calibration

## Objective and protocol

This simulation-only calibration estimates the primary delivered-impulse reference for the Z1 hammer task after correcting the detached cylindrical striker to the approved hammer-poll axial height. A passive `0.200 kg` cylinder, `0.024 m` in diameter and `0.0175 m` in total axial height, was released from rest along one frictionless, undamped vertical slide axis onto the unchanged production nail. Its lower face began exactly `0.150 m` above the nail-head contact surface. The run used the production first-strike tracker with axis `(0, 0, -1)`, a 25-substep inclusive window, and `progress_eps=5e-4`.

The authoritative execution ran exactly five sequential releases, each in a fresh one-environment mjlab instance. No trial was retried, filtered, weighted, trimmed, or excluded.

## Execution identity

- Slurm job: `41087111`
- State and exit: `COMPLETED`, `0:0`
- Node and elapsed time: `gn21`, `00:02:51`
- GPU: `NVIDIA A100-SXM4-40GB`
- Device/backend: `cuda:0`, `mujoco-warp`
- Code revision: `c5beb76c60ab7d420bbf1474637aef3d56bfb216`
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Versions: mjlab `1.4.0`, MuJoCo `3.8.1`, mujoco-warp `3.8.1`
- Physics timestep: `0.002 s`
- Striker geometry: mass `0.200 kg`; radius `0.012 m`; half-height `0.00875 m` (total height `0.0175 m`); lower-face clearance `0.150 m`
- Result SHA-256: `a64a735e2d945ebae950986e92b8045f6c942efb4fc733dbe43a1cc722980e18`
- Complete Slurm-output SHA-256: `2338d14f0c3813bf00329c34786be8b284eb26d98ae9ced189ac3c7a463b7b2d` (`13245` bytes)

Only Slurm job `41087111` was submitted for this corrected calibration. It completed once with `Restarts=0`; all five releases came from that one Python invocation.

## Raw result

The fixed release clearance was `h0=0.150 m` for every trial.

| Trial | Pre-contact velocity (m/s) | First-event impulse (N·s) | Depth at contact (m) | Peak depth (m) | Finalization | Productive |
|---:|---:|---:|---:|---:|:---|:---:|
| 1 | 1.7069429159164429 | 0.2799950838088989 | 0.0 | 0.03164022043347359 | `success` | yes |
| 2 | 1.7069429159164429 | 0.2799950838088989 | 0.0 | 0.03164022043347359 | `success` | yes |
| 3 | 1.7069429159164429 | 0.2799950838088989 | 0.0 | 0.03164022043347359 | `success` | yes |
| 4 | 1.7069429159164429 | 0.2799950838088989 | 0.0 | 0.03164022043347359 | `success` | yes |
| 5 | 1.7069429159164429 | 0.2799950838088989 | 0.0 | 0.03164022043347359 | `success` | yes |

All five releases contacted, finalized at the public `success` horizon, produced positive finite pre-contact velocity and impulse, started from zero velocity, and advanced the nail by more than `0.0005 m` during the first-event window.

The independently recomputed, exact unrounded arithmetic mean is:

`I_ref candidate = math.fsum([0.2799950838088989, 0.2799950838088989, 0.2799950838088989, 0.2799950838088989, 0.2799950838088989]) / 5 = 0.2799950838088989 N·s`

Observed spread across the five unfiltered trials: pre-contact velocity `0.0 m/s`; impulse `0.0 N·s`.

## Interpretation and boundary

The value is a banked object-side scalar from a constrained-cylinder simulation fixture with a `17.5 mm` hammer-poll-height surrogate. It is not a hardware safety calibration, a measurement of the articulated arm's effective mass, or a per-joint CaT cap. The centered cylinder preserves the nail-matched contact footprint while correcting the prior short contact coupon's axial thickness.

This record is simulation-only. Determinism or spread within this fixed MuJoCo/mujoco-warp protocol is descriptive of that protocol, not a claim of physical repeatability or hardware safety. The result does not by itself validate contact dynamics, material parameters, impact transfer, or safety margins on the physical Z1/hammer system.

The corrected arithmetic mean is a frozen calibration input only after the separate approved FIC-integration change. This calibration does not itself modify D4, FIC registrations, live `I_ref`, CaT activation, gains, curriculum, domain randomization, or variable impedance.

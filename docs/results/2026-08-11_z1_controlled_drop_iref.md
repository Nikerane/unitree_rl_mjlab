# Z1 controlled-drop impulse-reference calibration

> **Superseded on 2026-08-12:** This `8 mm` contact-coupon result is preserved as historical evidence but must not be used for FIC normalization. The corrected `17.5 mm` hammer-poll-height calibration is recorded in [`2026-08-12_z1_controlled_drop_poll_height_iref.md`](2026-08-12_z1_controlled_drop_poll_height_iref.md).

## Objective and protocol

This simulation-only calibration estimates the primary delivered-impulse reference for the Z1 hammer task. A passive `0.200 kg` cylindrical impactor was released from rest along one frictionless, undamped slide axis onto the unchanged production nail. Its lower face began exactly `0.150 m` above the nail-head contact surface. The run used the production first-strike tracker with axis `(0, 0, -1)`, a 25-substep inclusive window, and `progress_eps=5e-4`.

The authoritative execution ran exactly five sequential releases, each in a fresh one-environment mjlab instance. No trial was retried, filtered, weighted, trimmed, or excluded.

## Execution identity

- Slurm job: `41085760`
- State and exit: `COMPLETED`, `0:0`
- Node and elapsed time: `gn33`, `00:01:28`
- GPU: `NVIDIA A100-SXM4-40GB`
- Device/backend: `cuda:0`, `mujoco-warp`
- Code revision: `b769ed6838fe3171ff1cb3c0827aad5399a7e5b8`
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Versions: mjlab `1.4.0`, MuJoCo `3.8.1`, mujoco-warp `3.8.1`
- Physics timestep: `0.002 s`
- Result SHA-256: `615853fa03d3ba25dabad8e0c94c10949f57c45d9af0d1406be878f011b70a80`

Two earlier Slurm submissions (`41085757` and `41085758`) stopped before Python because Git was unavailable on the compute-node path; both ran zero releases and created no result. Shell-only diagnostic `41085759` identified Vega's standard Git module as the missing bootstrap and also ran zero releases. The corrected job changed only that operational bootstrap; the code, assets, resources, device, scientific command, and result path remained frozen.

## Raw result

The fixed release clearance was `h0=0.150 m` for every trial.

| Trial | Pre-contact velocity (m/s) | First-event impulse (N·s) | Finalization | Productive |
|---:|---:|---:|:---|:---:|
| 1 | 1.7069429159164429 | 0.10635668784379959 | `window` | yes |
| 2 | 1.7069429159164429 | 0.10635668784379959 | `window` | yes |
| 3 | 1.7069429159164429 | 0.10635668784379959 | `window` | yes |
| 4 | 1.7069429159164429 | 0.10635668784379959 | `window` | yes |
| 5 | 1.7069429159164429 | 0.10635668784379959 | `window` | yes |

All five releases contacted, finalized, produced positive finite velocity and impulse, started from zero velocity, and advanced the nail by `0.014746523462235928 m` during the first-event window.

The independently recomputed arithmetic mean is:

`I_ref candidate = math.fsum(raw impulses) / 5 = 0.10635668784379959 N·s`

The deterministic five-release run showed zero observed spread in both the recorded pre-contact velocity and impulse. This is a descriptive property of this fixed simulation protocol, not a physical repeatability claim.

## Interpretation and boundary

The value is a banked object-side scalar from a constrained-cylinder simulation fixture. It is not a hardware safety calibration, a measurement of the articulated arm's effective mass, or a per-joint CaT cap. The centered coupon deliberately matches the nail-head primitive rather than the production claw-hammer face geometry.

The next decision is to propagate this exact banked mean through a separate FIC integration plan and then execute the matched FIC-0/FIC-TT pilot. This calibration does not modify either live `I_ref` constant, D4, FIC registrations, CaT activation, gains, curriculum, domain randomization, or variable impedance.

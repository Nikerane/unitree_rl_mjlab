# Z1 fixed-impedance joint-policy pilot: FIC-0 versus FIC-TT

## Question and boundary

This matched simulation pilot qualifies the first learned absolute-joint-policy
arms on the Z1 hammer task. It asks whether the lean fixed-impedance treatments
can train and evaluate cleanly with the corrected controlled-drop impulse
reference, and records the behavior of the one approved treatment difference:

- **FIC-0:** no joint-command trackability cost.
- **FIC-TT:** adds the live cost
  `k_tt * sum_j(q_des_applied(t) - q_actual(t+1))^2` with exactly `k_tt=1.0`
  and reward weight `-1.0`.

The FIC-TT cost is ungated, remains active through contact, and stays in CaT's
unscaled negative-return partition. The rejected RTT-Q90 calibration is not
present. Both arms retain the same fixed gains, six-joint absolute action,
baseline reward terms apart from `r_tt`, velocity CaT, and log-only impulse
CaT.

This is a one-seed simulation pilot, not a multi-seed treatment conclusion or
a hardware-safety result. FIC-TT was not required to outperform FIC-0.

## Frozen treatment and protocol

- Training/evaluator revision:
  `0b270e26fac03cea5b2fedac5a34b60276dd9a4d`
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
- Runtime: mjlab `1.4.0`, MuJoCo `3.8.1`, mujoco-warp `3.8.1`
- GPU: one `NVIDIA A100-SXM4-40GB` per arm
- Controller: fixed-impedance absolute joint position, `joint1` through
  `joint6`; no gain action and no `set_gains`
- Training: seed `2`, `4,096` environments, `200` iterations, save interval
  `50`; both arms launched as one two-element Slurm array
- Evaluation: seed `2026081202`, `64` environments, exactly the first episode
  from each environment, deterministic mean policy, no auto-reset, `4.0 s`
  horizon
- Shared initial-population SHA-256:
  `320efae8c21b3303c5dc3f18ae7df00e887653abf26aa8fb413803cf78d094cb`
- Ordered-waypoint reward `P=8.0`; delivered-impulse reward `D4=4.0`
- Delivered-impulse reference:
  `I_ref=0.2799950838088989 N s`, from controlled-drop job `41087111` with
  lower-face release clearance `h0=0.150 m` and cylinder height `0.0175 m`
- Velocity CaT: active from the six-joint substep tracker
- Impulse CaT: measured but log-only; per-joint project caps remain
  `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64] N m s`
- Contact-row attribution: disabled for these FIC production runs
- Curriculum, domain randomization, variable impedance, and active impulse-CaT:
  absent

## Execution identity

### Clean CUDA gate

- Slurm job: `41094432`
- State and exit: `COMPLETED`, `0:0`
- Node and elapsed time: `gn32`, `00:05:57`
- Invocation: one `--task all --device cuda:0 --num-envs 8 --steps 3`
  live-manager smoke
- Result: FIC-0 `19/19`; FIC-TT `19/19`; empty stderr; post-smoke code and
  asset checkouts clean

### Matched training/evaluation array

- Slurm array: `41094536_0` (FIC-0), `41094536_1` (FIC-TT)
- States and exits: both `COMPLETED`, `0:0`
- Nodes and elapsed times: FIC-0 `gn32`, `00:13:22`; FIC-TT `gn49`,
  `00:13:18`
- Complete Slurm logs:
  `/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-41094536_{0,1}.{out,err}`;
  both stderr files are zero bytes

FIC-0 attempt and final checkpoint:

- Attempt:
  `/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot/0b270e26fac03cea5b2fedac5a34b60276dd9a4d/41094536_0_fic0_seed2`
- Checkpoint:
  `/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot/0b270e26fac03cea5b2fedac5a34b60276dd9a4d/41094536_0_fic0_seed2/logs/rsl_rl/z1_hammer/2026-08-12_14-08-34_fic_pilot_fic0_seed2_job41094536_0/model_199.pt`
- Checkpoint SHA-256:
  `f6ac3f597b9904c1687320f178e8cfe2731ccd52ca2e0cf56ea71adcf788e356`
- Banked JSON SHA-256:
  `666d4a816253ce0de73cb0be99f2b42250d2672f80c6c84981f27eb3dd4f90a4`

FIC-TT attempt and final checkpoint:

- Attempt:
  `/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot/0b270e26fac03cea5b2fedac5a34b60276dd9a4d/41094536_1_fictt_seed2`
- Checkpoint:
  `/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot/0b270e26fac03cea5b2fedac5a34b60276dd9a4d/41094536_1_fictt_seed2/logs/rsl_rl/z1_hammer/2026-08-12_14-08-44_fic_pilot_fictt_seed2_job41094536_1/model_199.pt`
- Checkpoint SHA-256:
  `534de95cd00e2b6627f1f305f62b58bb7bef6282bce4916cd9c7113ad6eeeebe`
- Banked JSON SHA-256:
  `fc22f2693fb896abacf702d95972f44c1686dc40672ca70d964ec64ef5963b46`

## Results

| Arm | Success | Productive first strike | First-event impulse mean ± population SD (N s) | Min–max (N s) | Mean / `I_ref` |
|---|---:|---:|---:|---:|---:|
| FIC-0 | 64/64 | 64/64 | 0.232262 ± 0.008718 | 0.225290–0.286978 | 0.8295 |
| FIC-TT | 64/64 | 64/64 | 0.292930 ± 0.006522 | 0.270463–0.307726 | 1.0462 |

FIC-TT's mean first-event impulse is `0.060668 N s` (`26.12%`) higher than
FIC-0 in this matched population.

Per-joint episode-peak impulse (`N m s`):

| Joint | FIC-0 p95 | FIC-0 max | FIC-TT p95 | FIC-TT max |
|---|---:|---:|---:|---:|
| joint1 | 0.139574 | 0.194690 | 0.182260 | 0.186546 |
| joint2 | 0.098834 | 0.130012 | 0.132702 | 0.135539 |
| joint3 | 0.156261 | 0.226244 | 0.205703 | 0.208356 |
| joint4 | 0.100771 | 0.132292 | 0.132006 | 0.133134 |
| joint5 | 0.108466 | 0.122086 | 0.133033 | 0.135631 |
| joint6 | 0.005454 | 0.014565 | 0.007306 | 0.009235 |

The largest observed cap utilization is FIC-0 joint3 at approximately `13.8%`
of its `1.64 N m s` project cap. No active impulse-CaT claim follows from these
log-only measurements.

## Verification and audit

Before GPU submission, the exact training revision passed:

- reward validation phases A--M;
- primary contact-sensor verification;
- the random-policy reward-setup sweep;
- both CPU live-manager smokes (`19/19` each);
- genuine one-iteration CPU CatPPO smokes for both arms (`192` samples each,
  finite learned and checkpoint tensors); and
- the full CPU suite: `2,589 passed, 4 skipped`, exit `0`.

Repository Standards and Spec reviews passed. Opus and DeepSeek production-slice
reviews reported no Critical or Important findings. Gemini itself was
unavailable because its configured model/quota could not serve the review, so
no Gemini verdict is claimed; a separately labeled fresh GPT-5.6 ultra review
of the immutable whole package passed with no findings.

After execution, an independent validator checked both JSONs against the live
checkpoints. It required canonical one-line JSON, exact frozen code/asset/task
and treatment identities, `64` ordered finite episode rows, independently
recomputed aggregates, live checkpoint hashes, and the identical reset
population hash. The audit returned `FIC_RESULT_AUDIT=PASS`. The local banked
files are byte-identical to the Vega originals.

## Interpretation and next boundary

Both fixed-impedance joint-policy arms learned a successful strike under this
one-seed protocol. The trackability-cost arm did not trade task completion for
the added cost and, descriptively, delivered more first-event impulse. It also
had higher p95 joint-impulse values on all six joints, while absolute maxima
were mixed and every observed value remained far below the current project
caps.

The pilot does **not** identify a general FIC-TT treatment effect: there is only
one training seed, and the frozen evaluator does not export the joint-target
tracking error itself. It also does not activate or qualify impulse CaT, infer
hardware safety, or test variable impedance. Those are separate later stages.
This FIC pilot stops here, before active impulse-CaT or VIC work; it adds no
further controlled-drop experiment beyond the already banked calibration used
as `I_ref`.

Banked machine-readable results:

- `evaluation/results/2026-08-12_z1_fic_joint_pilot/fic0.json`
- `evaluation/results/2026-08-12_z1_fic_joint_pilot/fictt.json`

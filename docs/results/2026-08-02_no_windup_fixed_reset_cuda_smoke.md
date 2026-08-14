# No-wind-up fixed-reset CUDA smoke

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer impulse cap”
> wording below refers only to the historical registered-task boundary, not a validated
> Z1 reaction-impulse or damage limit. The smoke record remains frozen; see
> `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Date:** 2026-08-02  
**Scope:** Unitree Z1 simulation; Vega A100; scripted direct reference; fixed impedance; log-only impulse machinery (`imp_max_p=0.0`)

## Result

Both production guideline arms passed the native 256-environment CUDA smoke
at the same clean code and asset revisions. Every environment reached accepted
contact, completed all six ordered gates, drove the nail successfully, kept the
substep impulse signal live, and stayed below the 3.1415 rad/s joint-speed rail.
C0 paid no gate reward; C-Gate paid the configured positive gate reward.

| Arm | CUDA pass | Six gates | Manager `r_gate` payout | Impossible success | Dead Lambda | Peak arm qvel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | yes | 256/256 | exactly 0 | 0 | 0 | 2.530754 rad/s |
| C-Gate | yes | 256/256 | 0.16000001 | 0 | 0 | 2.530754 rad/s |

There were no contract-required failed predicates in either arm. The treatment
digests confirm the intended experiment identity: both use the same fixed
reset, six gates, DiffIK action signature, fixed actuator signature, reward
readers, manufacturer impulse caps, and `imp_max_p=0.0`; only C-Gate enables
`r_gate`.

## Provenance

- Code revision: `67714191d727ad99a2601c2fc6dacdc9de7a3729`, clean before and after both smokes.
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`, clean before and after both smokes.
- Device: NVIDIA A100-SXM4-40GB, UUID `dfa99eb1-2e00-bb44-2d9a-46b87c113858`.
- Environments: 256 per arm; one GPU per run.
- Evidence directory: `evaluation/results/2026-08-02_no_windup_fixed_reset_cuda_smoke/`.

| Artifact | SHA-256 |
| --- | --- |
| `c0.json` | `bfcd069e93437e7c67f5544d9689fbff88aa15aef873ee748ba6553195af5ba4` |
| `c0.slurm.log` | `5e4a5005a7db763542ff1bd25e8e0d4cbb7d796f811a952b8efa8fb84e90acbc` |
| `cgate.json` | `2ad2520382dbffb4ca89addb888241f8ad75511ee4804e249331937804634be5` |
| `cgate.slurm.log` | `e323dafd47eed8343712ccdaa06e60f59d04c04cd4c7437a00225caca62d85cd` |

Two earlier C0 invocations failed before environment construction or rollout:
attempt 1 lacked the Vega compute-node git module; attempt 2 stopped during
task-module import because the expected sibling asset-repository link was
absent. Their logs are retained as infrastructure evidence.
The successful retry changed only the external execution environment
(`module load git`, `GIT_PYTHON_REFRESH=quiet`, and a symlink to the same clean
canonical asset checkout); task, seed/configuration, code, and assets were
unchanged.

## Interpretation and boundary

This qualifies the CUDA implementation and the C0/C-Gate treatment plumbing.
It does **not** show that PPO learns a straighter path, that the gate reward
improves success, or that the manufacturer impulse cap can bind. It also does
not authorize enforcement or variable impedance. Before a training pilot, the
strict learned-policy evaluator must persist and compare sampled guideline
geometry, success, impact speed, impulse, and hardware-qvel outcomes without
selecting seeds by performance.

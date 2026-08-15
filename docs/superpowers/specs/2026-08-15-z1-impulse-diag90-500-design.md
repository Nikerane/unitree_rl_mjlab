# Z1 diagnostic 0.9-boundary 500-iteration matched pair

**Status:** user-approved engineering experiment on 2026-08-15. This supersedes the Step-1
training stop only for the exact pair below. It does not authorize other thresholds, doses, seeds,
retries, or treatments.

## Question

Does a full 500-iteration exposure to active impulse soft-CaT change learned impulse behavior at the
uniform `0.9` diagnostic boundary, compared with an otherwise identical log-only control?

The completed 50-iteration pair established activation plumbing only. This experiment extends that
same comparison to the ordinary 500-iteration VIC training budget; it does not turn soft CaT into a
hard clamp and does not make the diagnostic boundary a hardware limit.

## Frozen comparison

| Field | Control | Target |
|---|---:|---:|
| Task | VIC-TT | VIC-TT |
| Training initialization | fresh, seed 2 | fresh, seed 2 |
| Environments x steps | 4,096 x 24 | 4,096 x 24 |
| PPO iterations | 500 | 500 |
| Velocity CaT | registered active treatment | registered active treatment |
| Impulse boundary, J1--J6 | `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N·m·s` | same |
| `imp_max_p` | `0.0` | `0.5` |

Only run name/output leaf and `imp_max_p` may differ. The pair is an array `0-1` on one A100 per
arm. It uses the exact pushed code revision and canonical clean asset revision, forbids resume and
distributed variables, requires the pinned mjlab/MuJoCo runtime, and refuses reused output leaves.

## Launcher and outputs

Create a separate `scripts/slurm/vega_vic_impulse_diag90_500.sbatch`; do not change the banked
50-iteration launcher. Freeze `--agent.max-iterations 500`, `--agent.save-interval 50`, seed 2,
4,096 environments, and 24 steps per environment. Require exactly checkpoints
`model_{0,50,100,150,200,250,300,350,400,450,499}.pt`, validate finite tensor state and stored
iteration identity, and print SHA-256 for every checkpoint.

Use a distinct `z1-vic-impulse-diag90-500` campaign root and a 90-minute allocation. Set
`#SBATCH --no-requeue`; never automatically retry a failed arm.

## Interpretation and follow-up

Training completion alone answers only whether the exact treatment ran. After both arms finish,
evaluate `model_499.pt` from each arm on the same fixed and sampled populations, with separate
velocity/impulse telemetry, and compare provisional/diagnostic violation rate, per-joint Lambda,
utilization, winner shares, and task behavior. Aggregate training scalars are descriptive and cannot
establish impulse-only causality.

One seed cannot support a general performance or safety claim. The uniform `0.9` vector remains a
diagnostic project threshold; Unitree publishes no corresponding external-reaction impulse damage
limit.

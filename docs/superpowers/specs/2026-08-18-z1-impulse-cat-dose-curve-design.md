# Z1 impulse-CaT compact dose-curve design

**Status:** Approved for execution by the user on 2026-08-18.

## Question

Under the same project-defined provisional impulse caps, does the seed-2 VIC policy retain the
successful `p=0.2` behavior at a lower impulse-CaT dose (`p=0.1`), and does a higher dose (`p=0.3`)
begin to trade impulse reduction for joint-velocity risk?

## Treatments

Train exactly two new targets:

| Role | `imp_max_p` | Scientific purpose |
|---|---:|---|
| `dose_p01_target` | `0.1` | Lowest-dose efficacy screen |
| `dose_p03_target` | `0.3` | Stronger-dose velocity-trade-off screen |

Both use the existing VIC-TT task, seed `2`, `4096` environments, `24` rollout steps, `500` PPO
iterations, save interval `50`, and provisional caps
`[0.82,1.64,0.82,0.82,0.82,0.82] N.m.s`. Velocity CaT, rewards, observations, actions, physics,
reference trajectory, VIC gain range, domain randomization, and curriculum remain unchanged.

The existing frozen policies are reused descriptively:

- `dose_p0_control`: the banked log-only control, after the existing exact cap-vector identity proof;
- `dose_p02_target`: the successful banked `p=0.2` bridge target.

The historical `p=0.5` diagnostic policy is excluded from the primary curve because it trained
against the uniformly 0.9-scaled diagnostic caps rather than the provisional caps.

## Execution and evaluation

- Train the two new targets as one two-arm Slurm array so they run concurrently when resources
  allow. Each arm gets one A100, eight CPUs, 40 GB, a 90-minute limit, and `--no-requeue`.
- Submit once after one accepted `sbatch --test-only`; never retry, requeue, extend, or substitute a
  failed arm automatically.
- After both final checkpoint hashes are bound in code, evaluate all four frozen policies in one
  four-arm array using the existing survey with live `imp_max_p=0` for every role.
- Use the same fixed 64-world population and the same three matched stochastic 4096-world
  populations at seeds `2`, `2026081701`, and `2026081702`.
- Require exact role/hash/provenance, matched RNG/population identities, finite traces, zero live
  impulse delta, and exact max soft-OR.

## Analysis

Judge `p=0.1`, `p=0.2`, and `p=0.3` separately against the same `p=0` control with the already
preregistered bridge gates. No pooled population or cross-dose average may rescue a failed dose.
The fixed-64 population remains descriptive.

The dose curve is a one-training-seed screen. It may establish which tested doses produced useful
native-observed behavior in these policy instances, but it cannot establish an optimal dose,
cross-training-seed reliability, hard constraint enforcement, hardware safety, actuator damage
limits, or complete-contact loading.

## Stop boundary

After banking the four-policy result and a concise external interpretation, stop. Do not begin
curriculum learning, additional domain randomization, another dose, independent training seeds,
contact flushing, torque CaT, or hardware transfer without new explicit approval.

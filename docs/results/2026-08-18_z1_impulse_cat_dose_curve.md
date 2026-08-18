# Z1 provisional-cap impulse-CaT compact dose curve

**Status:** banked one-training-seed simulation screen, 2026-08-18. **Strict verdict:** every tested
dose fails the preregistered every-population gate. The separate matrix is `p=.1` FAIL/FAIL/FAIL,
`p=.2` PASS/FAIL/PASS, and `p=.3` FAIL/FAIL/FAIL. Thus `p=.1 is too weak`; `p=.3 is not
monotonically better` and has worse impulse-tail/velocity results plus two censored utility
populations; `p=.2 is the strongest tested policy instance`, but it is borderline and **not a
formal PASS**. The next scientific question is independent training-seed confirmation of the same
`p=.2` treatment, not an increased dose.

This is native observed horizon evidence from one training seed under provisional project caps;
soft pressure, not a clamp, was applied during training. This record makes no manufacturer or
hardware-safety claim, no complete-contact claim, and no optimal-dose claim; evaluation replicas
are not training seeds.

## Frozen treatment and execution identity

All policies were VIC-TT seed 2 with velocity CaT active. The three active targets used 4,096
environments x 24 steps, 500 PPO iterations, save interval 50, and caps
`[0.82, 1.64, 0.82, 0.82, 0.82, 0.82] N.m.s`; rewards, observations, actions, physics, reference,
VIC range, domain randomization, and curriculum were unchanged.

| Role | Training dose | Training identity | Elapsed | Final checkpoint SHA-256 |
|---|---:|---|---:|---|
| `dose_p0_control` | 0 | reused log-only job `41290111_0`; no new training | banked separately | `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3` |
| `dose_p01_target` | .1 | array `41504493_0`, code `e7665b328d175a6dfe329ef2fb652a71d0feea5e` | 00:25:31 | `92f1d97c8ff1476cb26c0b648478a0bc3c522e4e1eb7087389fb8e0d6bf73f86` |
| `dose_p02_target` | .2 | job `41461595`, code `8ff0ccfb577d2218638a25170a11023255a78b9d` | 00:20:58 | `57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de` |
| `dose_p03_target` | .3 | array `41504493_1`, code `e7665b328d175a6dfe329ef2fb652a71d0feea5e` | 00:25:11 | `4c0a665fffc077d488630a28b258c1593050f969b4f6227147c3594097dc4efd` |

The shared asset revision was `b58ccd2f81fd246f27c1e8d88cf86484cd888703`. New p=.1/.3
training used one accepted dry-run (`41504485`) and one two-arm submission, both arms
`COMPLETED/0:0`, zero restarts, empty stderr, exactly 11 checkpoints, NVIDIA
A100-SXM4-40GB, mjlab 1.4.0, MuJoCo 3.8.1, and mujoco-warp 3.8.1. Their runtime was 0.845 allocated
A100 GPU-hours. The p=.2 identity and runtime are independently banked in the bridge record.

The frozen evaluation used code `9718a7958cbd7ec6b2f3e126f8fbae8ef81762df`, the same asset
revision, one accepted dry-run (`41513713`), and one array `41513714`:

| Task | Role | Node | State / exit | Elapsed | stderr |
|---:|---|---|---|---:|---:|
| 0 | `dose_p0_control` | gn10 | `COMPLETED` / `0:0` | 1:37 | 0 bytes |
| 1 | `dose_p01_target` | gn33 | `COMPLETED` / `0:0` | 1:43 | 0 bytes |
| 2 | `dose_p02_target` | gn51 | `COMPLETED` / `0:0` | 1:38 | 0 bytes |
| 3 | `dose_p03_target` | gn51 | `COMPLETED` / `0:0` | 1:43 | 0 bytes |

All arms ran with live `imp_max_p=0`: `delta_impulse` was identically zero and combined delta was
exactly `max(delta_velocity, delta_impulse)`. Fixed-64 was descriptive only. The inferential
populations were three separate matched 4,096-world replicas at seeds `2`, `2026081701`, and
`2026081702`; whole paired environments, not 50 Hz reads, were resampling units. Each interval used
10,000 resamples and bootstrap seed `2026081703`; populations were never pooled to rescue a dose.

Evaluation manifest SHA-256 values were control
`0b24cab9bcc59e1507b2ca4f3ccdcf63d1bf58901b03a263a7bb1b8cee60371c`, p=.1
`110215b9ee5101b85f82aa3d44e2642e7c83bffe0e54088d33f5844ab737b1b5`, p=.2
`497572c6506494ba7750135057422e02421fa4828558a53388319ef8c6187747`, and p=.3
`17939ed145beb0d81036b31e0ce8e169e14e1d0a14ee25f8e3e8512368fd61ac`.

## Preregistered endpoints

Risks and CI upper bounds are environment percentages. `rho` is global peak cap utilization;
`deliv` is the target/control first-event delivered-impulse mean ratio with its 97.5% lower bound.
`C->T` means control to target. `P/F` is the corresponding gate. Fixed-64 does not enter verdicts.

| p | Seed | Impulse risk C->T (diff; CI-hi), gate | rho p95 / p99 C->T, gate | Velocity risk C->T (diff; CI-hi), gate | Utility: success/productive diff; deliv (lower), gate |
|---:|---:|---|---|---|---|
| .1 | 2 | 0.8545->0.3418 (-0.5127; -0.1709), F | .8067->.8161 / .9735->.9485, F | 0.5127->0.7324 (+0.2197; +0.4883), F | 0/0; .9827 (.9817), P |
| .1 | 2026081701 | 0.7324->0.2686 (-0.4639; -0.1709), F | .8070->.8108 / .9071->.9383, F | 0.6836->0.6104 (-0.0732; +0.2197), F | censored initial fragment, F |
| .1 | 2026081702 | 0.8545->0.2197 (-0.6348; -0.3418), F | .8069->.8097 / .9870->.9418, F | 0.6592->1.1719 (+0.5127; +0.8301), F | 0/0; .9834 (.9825), P |
| .2 | 2 | 0.8545->0 (-0.8545; -0.5859), P | .8067->.5158 / .9735->.6434, P | 0.5127->0.1709 (-0.3418; -0.1221), P | 0/0; .9954 (.9948), P |
| .2 | 2026081701 | 0.7324->0 (-0.7324; -0.4883), **F** | .8070->.4559 / .9071->.6361, P | 0.6836->0.1953 (-0.4883; -0.2441), P | 0/0; .9964 (.9957), P |
| .2 | 2026081702 | 0.8545->0 (-0.8545; -0.5859), P | .8069->.5208 / .9870->.6335, P | 0.6592->0.2441 (-0.4150; -0.1465), P | 0/0; .9955 (.9950), P |
| .3 | 2 | 0.8545->0.0732 (-0.7812; -0.4883), F | .8067->.7877 / .9735->.8787, F | 0.5127->0.6104 (+0.0977; +0.3662), F | censored initial fragment, F |
| .3 | 2026081701 | 0.7324->0.0488 (-0.6836; -0.4150), F | .8070->.7866 / .9071->.8753, F | 0.6836->0.6348 (-0.0488; +0.2197), F | 0/0; .9740 (.9729), P |
| .3 | 2026081702 | 0.8545->0.0977 (-0.7568; -0.4639), F | .8069->.7866 / .9870->.8842, F | 0.6592->0.9033 (+0.2441; +0.5371), F | censored initial fragment, F |

The p=.1 dose failed impulse-tail and velocity gates in all populations. The p=.3 dose failed
global-tail and velocity gates in all populations and produced censored utility fragments in two.
Only p=.2 passed tail, velocity, joint-tail, identity/native-claim, and utility gates throughout;
its middle population failed only the strict impulse-risk gate.

For auditability, the unchanged `analysis.json` retains every fixed/stochastic J1-J6 utilization
p50/p95/p99/max, native episode duration, contact-prefix/censoring descriptor, cap-associated read,
VIC gain, and action statistic. Across stochastic populations the control had `35/30/35`
cap-violating reads; targets had p=.1 `14/11/9`, p=.2 `0/0/0`, and p=.3 `4/3/4`.
Activation-window reads were one per window except one two-read p=.3 window in seed 2 and one in
seed 2026081701. Median native episode duration was control `140/140/140 ms`, p=.1
`160/160/160 ms`, p=.2 `140/140/140 ms`, and p=.3 `160/160/160 ms`. These are episode-prefix
durations, not complete physical-event durations; contact censoring forbids a complete-event claim.

## Why the earlier p=.2 PASS changed

The old and new p=.2 evaluations used identical declared initial-population hashes and RNG stream
IDs, not different samples. They were separate A100 executions, and documented GPU atomic-order
nondeterminism produced about `1e-6` preterminal action differences that contact and termination
amplified.

Exactly control environment `3253` changed in seed `2026081701`. The old run reached read 7 with
J3 Lambda `0.8731314` and margin `+0.0531314`; the new run succeeded one read earlier at read 6,
with Lambda `0.54809135` and margin `-0.27190864`. Consequently the risk-difference CI-high moved
from `-0.005126953125` (PASS) to `-0.0048828125` (FAIL), missing the strict `< -0.005` rule by
`0.0001171875`. Classification uses the correct native-float margin; this is not threshold
rounding and not an analyzer bug. The sensitivity is one environment near a termination boundary,
which is precisely why p=.2 remains the strongest tested instance but not a formal PASS.

## Artifact ledger and decision

The authoritative analysis is 575,179 bytes with SHA-256
`8289a3c0247e086ae244c70bbbddb0820cf6390543ee61a90d6d71026da24135`. It was retrieved once,
verified finite/provenanced, and banked byte-for-byte unchanged. The adjacent `SHA256SUMS` binds
that JSON, the exact analyzer source, this record, and the sanitized review packet.

**Decision:** do not increase dose on this evidence. The bounded next proposal is matched
independent training-seed confirmation of `p=.2`; it is not authorized by this record. No further
training, evaluation, curriculum, domain-randomization, contact-flush, torque-CaT, or hardware run
was performed while banking this result.

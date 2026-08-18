# Z1 impulse-CaT compact dose-curve independent reviews

## Material passport

- Content class: sanitized aggregate simulation-result critique
- Date: 2026-08-18
- Transport: OpenCode `1.18.18`, OpenCode Zen provider, `run --pure`, `max` variant
- Source packet SHA-256:
  `485bf07d472d94c3a200721b72732ec2735ba1fea02219619fac80cb0a51a6cd`
- Raw traces, checkpoints, source code, repository access, credentials, personal information, and
  private repository paths supplied: no
- Review authority: advisory interpretation only; opinions cannot turn numerical FAIL into PASS

The unchanged source packet remained 3,278 bytes, including its terminal newline. A post-session
audit found a small shell-transport deviation that must not be hidden: all five OpenCode user
messages had the same SHA-256
`86e1f9e8cc8a96b535e0e3267abce2a78ea1ecd7a53652a23116fadf3f4e6c02`, consisting of the packet
text without its terminal newline and with one literal opening and closing quote around it. The
text between those quotes exactly matched the source packet without the terminal newline, SHA-256
`f7ba41a559af3cd73f2fbf0e5cc1c00b983a56c0f9225e31623750c030a7f548`. Thus the semantic packet
was identical for all reviewers, but transport was not byte-identical to the source file. The
one-shot rule forbade corrective inference retries, so none was attempted.

Before the user's explicit external-transmission approval, two pre-launch authorization blocks
rejected the proposed Kimi command. Both occurred before process creation and produced no process,
session, inference, or cost. After explicit approval, exactly one successful inference attempt per
model ran. There was one inference attempt per model and no inference retry. All five exported
session summaries reported zero added, deleted, or modified files. The models ran in an isolated
temporary directory containing only the prompt copy and had no repository access.

## Exact provenance ledger

Token columns are session-export `input / output / reasoning / cache-read / cache-write`. Response
hashes cover the concatenated `text` parts of the final assistant message exactly; hidden reasoning
and tool output are excluded.

| Model | Variant | Session ID | Verdict | Final assistant-text SHA-256 | Tokens input / output / reasoning / cache-read / cache-write | Provider `cost` field |
|---|---|---|---|---|---|---:|
| `opencode/kimi-k3` | max | `ses_fea78634effejHlbQaCDimILyc` | `CONFIRM` | `e52485ddd075716b015e79c0d9a6dd5417b284eda3c8f59b4fa3ba4bb9f5ae6a` | 11557 / 1534 / 0 / 20139 / 0 | 0.0637227 |
| `opencode/glm-5.2` | max | `ses_fea778cc1ffeHj1ZmfxvMgTY8v` | `CONFIRM` | `38ccfef1095d37ecda28dcfafea547a36b076821e351a894795f6054e661daea` | 9911 / 3587 / 0 / 152 / 0 | 0.02969772 |
| `opencode/qwen3.6-plus` | max | `ses_fea763a7effet5fEug5xht5c0P` | `STOP` | `ae588f1eb460c9cea5755ceffc7b633a37eafbc34828d060d712d1aac27540b0` | 6 / 713 / 0 / 0 / 10609 | 0.008772625 |
| `opencode/deepseek-v4-pro` | max | `ses_fea75787affeRP4SV1D9ZYolmI` | `CONFIRM` | `85b40e5d7f61485af08f6dad8225f909bac2713c9c4bb3cec672831498601b25` | 10435 / 909 / 0 / 0 / 0 | 0.02164746 |
| `opencode/claude-opus-5` | max | `ses_fea7501f6ffeWDX31EMku6uNAH` | `REVISE` | `d91c641923f55f2a88d1cfad57e2a354866dc40352e66bb8c4d7a7a92a6698bb` | 4 / 6748 / 0 / 14995 / 20176 | 0.3023175 |

The session-level provider `cost` fields sum to `0.426158005`. OpenCode names this field `cost`
but supplies no unit or currency, so it has unknown unit and currency and is not reported as
dollars.

## Faithful interpretations

- **Kimi K3 — `CONFIRM`.** It treated the non-monotone screen as successful candidate selection:
  p=.1 was too weak, increasing to p=.3 was worse, and p=.2 was the sole surviving candidate. It
  requested at least two independent PPO training seeds plus deterministic or repeated evaluation,
  while explicitly opposing post-hoc relaxation of the strict gate.
- **GLM-5.2 — `CONFIRM`.** It likewise prioritized the failed higher dose and the one-environment
  execution sensitivity. It requested independent training seeds and an evaluation procedure able
  to distinguish training variation from the approximately one-lattice-step execution variation.
- **Qwen3.6-Plus — `STOP`.** It held that p=.2 must not advance because the preregistered
  every-population rule failed, and proposed a larger evaluation population before spending new
  training seeds.
- **DeepSeek V4 Pro — `CONFIRM`.** It accepted p=.2 because p=.3 was worse, then proposed a fresh-RNG
  deterministic replay of the failed population and treating a crossing as a pass.
- **Claude Opus 5 — `REVISE`.** It identified the discrete support of the 4,096-environment risk
  interval: the old and new CI highs were exactly `-21/4096` and `-20/4096`, while the threshold was
  `-20.48/4096`. It therefore requested a replay-only execution-nondeterminism envelope before new
  training-seed expenditure, without changing the current numerical verdict.

## Convergence and disagreement

None recommended increasing directly to p=.3. All five treated p=.2 as the strongest tested policy
instance. DeepSeek explicitly overcalled it an optimum; the adjudication below rejects that claim.
All accepted that the same declared population and RNG can cross the strict boundary under
separate A100 executions. They split on what that means next: Kimi, GLM, and DeepSeek voted to
advance p=.2; Qwen voted to stop before training seeds; Opus asked to revise the evaluation
decision procedure first. Kimi and GLM would combine independent training-seed confirmation with
deterministic or repeated evaluation. Qwen asked for a larger evaluation population. Opus asked
first for repeated executions of the existing frozen pair and population. These are advisory
design proposals, not new experimental evidence or authorization.

## Factual adjudication

The preregistered result remains authoritative: p=.1, p=.2, and p=.3 all numerically FAIL the
every-population rule, and all three trained targets still represent only one PPO training seed.
Three advisory `CONFIRM` votes cannot convert p=.2 to PASS. p=.2 remains the strongest tested
policy instance because it alone produced zero target impulse violations in all three populations
while retaining the other tail, velocity, and utility gates; that supports further investigation,
not confirmation or optimality.

Opus's lattice arithmetic is correct and materially useful. With 4,096 whole-environment units,
one step is `1/4096 = 0.000244140625`; the old and new bounds straddled the threshold by exactly one
step. This does not make the analyzer wrong and does not authorize retroactively relaxing or
rescoring the preregistered gate. It does show that future confirmation cannot interpret a
one-step boundary change without an explicit execution-repeat policy.

Qwen's `0.000117 N.m.s` unit is wrong: `0.0001171875` is a dimensionless risk-difference fraction,
not joint impulse. Its proposed larger population could refine the estimator's lattice but cannot
rescue the already failed preregistered screen. DeepSeek's optimum claim is unsupported; the data
only identify p=.2 as strongest among three tested active doses. Its fresh-RNG rescue and proposal
to count a favorable replay as PASS are rejected because they would replace the preregistered RNG
stream and post hoc overturn a strict result.

## Bounded next proposal

Before new training, the smallest defensible proposal is a separately preregistered, replay-only
execution-nondeterminism envelope: repeat the frozen p=0 versus p=.2 evaluation for population
`2026081701` ten times with the identical population hash and RNG stream IDs, and report the
distribution of control/target risks and impulse-risk CI-high lattice values. The existing FAIL is
not rescored. The follow-on confirmation protocol must then predeclare how execution repeats,
population size, and any indeterminate band are handled before independent training seeds are
spent. This document authorizes no replay, no new training, no new dose, no threshold relaxation,
and no hardware claim.

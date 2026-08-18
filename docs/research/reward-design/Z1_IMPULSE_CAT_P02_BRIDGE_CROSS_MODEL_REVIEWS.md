# Z1 impulse-CaT `p=0.2` bridge independent reviews

## Material Passport

- Content class: sanitized aggregate experiment-result critique
- Date: 2026-08-18
- Transport: OpenCode `1.18.18`, OpenCode Zen provider
- Packet SHA-256: `fd8fc307bdef15ce17b6ebe157776083faba1a291422becbccca958ee1b72cb5`
- Raw traces, checkpoints, source code, private paths, credentials, and personal information sent:
  no
- Review authority: advisory interpretation only; the preregistered numerical PASS controls the
  scientific verdict

**External review status: complete.** Five initial one-shot attempts used the identical banked
packet. Qwen, DeepSeek, and Opus reached inference and returned `CONFIRM`; the Kimi K3 and GLM-5.2
commands failed before inference. The user then explicitly overrode the original no-retry rule for
those transport-only failures. Two user-authorized single corrected attempts used the same packet;
both reached inference and returned `CONFIRM`. The final panel is five successful reviews from
seven transparent CLI invocations, not five hidden first-try successes.

## Provenance ledger

| Model | Variant | Initial attempt | Corrected/successful session | Verdict | Final assistant-text SHA-256 | Provider-reported cost field |
|---|---|---|---|---|---|---:|
| `opencode/kimi-k3` | max | original attempt failed before inference | `ses_feba5cbe1ffekvYzZOXyvSXDix` | `CONFIRM` | `de852428883961a84674ad703548fdc864fe690aad732ca30ff7d53fa4a7718d` | 0.11468700 |
| `opencode/glm-5.2` | max | original attempt failed before inference | `ses_feba3f003ffekOBTJjyglZ79o1` | `CONFIRM` | `65ecac553b87dd6f3ee30c80813b4c617111260c1d15d8d7cfeb84eea8c10992` | 0.03152272 |
| `opencode/qwen3.6-plus` | max | success | `ses_febacb884ffeHpIAmgnOoybzXb` | `CONFIRM` | `fa0cc4994a88f6ef0978c88e83d59444c06e1e1f2bbd36f2235b4de392064ed1` | 0.01036425 |
| `opencode/deepseek-v4-pro` | max | success | `ses_febac38e0ffefKkxZE6AEEwb7I` | `CONFIRM` | `ddd8aa9a70f2a70861555a886d37b4b5648c8e5017836480910f7c731f715f5c` | 0.02372724 |
| `opencode/claude-opus-5` | max | success | `ses_febaba3bcffelKhCypldwCohwf` | `CONFIRM` | `e4bf976dce79e5a01e8a29625ec95921278149211b179800478ed662c9557b16` | 0.30906200 |

The successful provider-reported cost fields sum to `0.48936321`. OpenCode labels this numeric
field `cost` but provides no unit or currency here, so the value has unknown unit and currency and
is not described as dollars. The Opus row uses the final exported **session** cost, not the smaller
last-message cost.

Response hashes cover the concatenated final assistant `text` parts exactly; hidden reasoning and
tool output are excluded. Read-only `opencode export <session> --pure` recovered all five texts and
recomputed all five hashes exactly. All five successful session summaries reported zero added,
deleted, or modified files. Opus used its built-in shell only to inspect its empty working directory
and list `/private/tmp`; it found no supporting artifact, did not access or modify this repository,
and then reviewed the self-contained packet.

## Transparent transport failures and override

- **Kimi K3:** the initial command-line `--file` option greedily parsed the trailing request as
  another file name, so the original attempt failed before inference with
  `Error: File not found: Review the attached packet exactly as requested.` It produced no session
  or cost.
- **GLM-5.2:** the initial invocation supplied no parsed message and failed before inference with
  `Error: You must provide a message or a command`. It produced no session or cost.

Neither failed invocation was automatically retried. After both failures were reported, the user
explicitly authorized exactly one corrected retry for each. Those two corrected invocations are the
Kimi and GLM success rows above. No further retry occurred.

## Convergence across all five reviews

All five reviewers independently agreed that:

- the bridge supports moving to an independently trained-seed confirmation stage, but does not by
  itself establish a general method effect;
- the single PPO training seed is the main unresolved limitation; three evaluation populations are
  stability checks on the same policy pair, not three training replicates;
- the strongest wording is restricted to this simulated training pair, the project-defined caps,
  and native observed reaction-impulse exposure;
- zero observed target violations is not zero risk, a hard clamp, actuator protection, or hardware
  safety; and
- no dose sweep, new threshold, or hardware conclusion should be introduced at confirmation.

Qwen, DeepSeek, and GLM said contact/window-completion sensitivity is needed only before a
complete-contact or actuator-loading claim, not before native-horizon training-seed confirmation.
Opus also did not make it a prerequisite, but recommended cheap completed-window and contact-onset
descriptors alongside confirmation. Kimi uniquely requested one cheap completion-conditioned or
censoring-robust sensitivity before confirmation. This is a design disagreement, not a failed
bridge gate.

## Stronger cautions and factual adjudication

Opus and GLM framed the result as a **single matched training-pair screen**, not a property of
impulse-CaT. Their strongest competing explanation is a training-run confound: the target's
simultaneous improvement in velocity risk could indicate a luckier or better-converged overall PPO
run rather than a channel-specific effect. Opus and GLM therefore favored retraining both `p=0` and
`p=0.2` in paired new training seeds. DeepSeek and Kimi explicitly proposed new target seeds with
reuse of the qualified frozen control; Qwen requested replication across new independent training
seeds without resolving that control-reuse detail. This disagreement is scientifically material.

Opus flagged that physical-contact-prefix counts exceed the 4,096 environment count and differ by
arm. The analysis defines these as counts of physical event prefixes, not unique environments, so
they are not a population-size mismatch and are not inferential units. The heavy censoring and arm
difference nevertheless remain mandatory descriptors and continue to prohibit complete-event or
actuator-loading claims.

Kimi accidentally described the target as having about 94.9% completed prefixes and the control as
having about 98.4%. Those are the **right-censored**, not completed, fractions: only about 5.1% and
1.6%, respectively, completed. The target therefore exposes more completed contact, so this
particular asymmetry does not support a shorter-observation artifact. It does not remove the broader
complete-contact limitation.

Reviewers also proposed differing seed counts and aggregate rules. These suggestions would change
the confirmation design and compute budget. Paired newly trained controls, a cheap censoring
sensitivity, the number of seeds, and any aggregate rule may be scientifically valuable, but they
are not automatically authorized by these reviews.

## Adjudication

The external reviews do not vote on the experiment result. The preregistered numerical screen had
already passed in all three stochastic populations, and that PASS remains the authoritative bridge
verdict. The unanimous advisory conclusion is narrower: independent training-seed confirmation is
scientifically justified.

The next design must explicitly resolve control reuse versus paired newly trained controls and
whether a cheap censoring sensitivity precedes or accompanies confirmation. It must be separately
testable, budgeted, and user-approved. Until then: no new training, no extra dose, no contact flush,
no torque CaT, and no rescue experiment.

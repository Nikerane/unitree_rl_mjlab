# Z1 impulse-CaT simplified-plan independent reviews

## Material Passport

- Content class: sanitized aggregate experiment-plan critique
- Date: 2026-08-18
- Transport: OpenCode `1.18.18`, OpenCode Zen provider
- Prompt SHA-256: `7a63e68fc7ea08cc6a2a65ed25e9e25ac62e214a656cc844f72173b9afcfb2e6`
- Raw traces, checkpoints, source code, private notes, and repository files sent: no
- Review authority: advisory hypotheses only; no vote and no scientific evidence

Each model received the same 547-word packet in an empty `/private/tmp` directory with `--pure`,
the `max` reasoning variant, no external plugins, and no `--auto`. All sessions reported zero file
changes. The exact transmitted packet is
`docs/research/reward-design/Z1_IMPULSE_CAT_CAMPAIGN_CROSS_MODEL_PACKET.md`.

## Provenance ledger

| Model | Session ID | Verdict | Response SHA-256 | Provider-reported cost field |
|---|---|---|---|---:|
| `opencode/kimi-k3` | `ses_fec9f3ed1ffe5LeGitzGk2abka` | REVISE | `ea601a2d3c52c255ecc9e368a9d532b45dcb63010344641dc5e44350185f6847` | 0.09463770 |
| `opencode/glm-5.2` | `ses_fec9f45ddffef5VvNiY2FFiDDC` | REVISE | `01421d7461f507d28072963c14318d7dfbd76ee036000480481d1d79b5722f66` | 0.02935392 |
| `opencode/qwen3.6-plus` | `ses_fec9f45ecffeQM6T2sftEX2Dwt` | REVISE | `977d02d63655ef0c6d9e14b8a451c9b33e1beca3500c07be6518fbe1e173a19e` | 0.01194825 |
| `opencode/deepseek-v4-pro` | `ses_fec9f3e39ffeMIaDGDTOo87QJY` | REVISE | `de7a90e1b2539254d301bc783775a857fd4a9288a9cde67ac358c313bf56bbd9` | 0.02728200 |
| `opencode/claude-opus-5` | `ses_fec9f43e3ffe2iGVLYGcl5PIhH` | REVISE | `f3010aa98ddd7202597d519ce139cace03e0ae735ee28e7992ceb0a99e4a9f4b` | 0.33131425 |

Total provider-reported cost field: `0.49453612`. OpenCode's session export names this field
`cost`; the CLI catalog does not label its currency or unit, so none is inferred here.
Response hashes cover the concatenated final assistant `text` parts only; hidden reasoning and
tool output are excluded.

## Consensus worth accepting

All five reviewers agreed on the central scientific boundary:

- one PPO training seed is not robust evidence;
- the increased true velocity risk is the decisive warning;
- no result supports hard capping, hardware safety, or manufacturer damage limits;
- the 29% shorter target episode can make native impulse exposure look better and must remain
  visible in interpretation;
- `p=0.2` is a bridge hypothesis, not an optimized or physically calibrated dose.

Kimi, Qwen, and DeepSeek called contact/window flushing optional. GLM called it optional for native
episode risk but relevant to delivered-event interpretation. Opus instead requested a primary
censoring-invariance analysis. The adopted design keeps the native task endpoint primary, makes
duration/censoring and contact-aligned observed prefixes mandatory descriptors, and leaves full
contact completion optional. Consequently, the claim is limited to native observed exposure.

## Disagreements and adjudication

| Review suggestion | Decision | Reason |
|---|---|---|
| Repeat the diagnostic-cap `p=0.5` treatment before trying a lower dose | Reject | It would replicate a treatment already associated with the unwanted velocity trade-off and would not answer whether a lower dose avoids it. |
| Treat changing from the historical diagnostic `p=0.5` target to a provisional `p=0.2` target as a two-variable causal comparison | Clarify | The bridge's causal comparison is banked `p=0` control versus new `p=0.2` target, both analyzed at provisional caps. At `p=0`, the cap is log-only and must pass exact identity before reuse. The old `p=0.5` target is context only. |
| Sweep offline `p` on a frozen trained policy to determine how velocity risk responds to dose | Reject | Offline replay changes only counterfactual pressure. It cannot change frozen actions or velocity and therefore cannot estimate a learned dose-response curve. |
| Require complete contact before any bridge | Reject | This would answer a stronger complete-event question than the simplified native-task screen requires. It remains an optional sensitivity analysis and limits the wording of any result. |
| Add a cheap matched-exposure check on existing traces | Accept narrowly | Reconfirm the direction of the already banked contact-aligned observed-prefix tail before the bridge and stop on a reversal. Do not treat overlapping windows as independent, create a new event framework, or claim the check optimizes `p`. |
| Infer unchanged joint loading from the approximately `1.02` delivered hammer-impulse ratio | Reject | Delivered hammer/nail impulse is a task-utility quantity; the constrained quantity is per-joint reaction Lambda. Retaining the former does not prove the latter is unchanged. |
| Use the provisional-cap endpoint as hardware evidence | Reject | The endpoint is explicitly project-defined simulation evidence only. |

## Resulting honest verdict

The telemetry and learning pressure work. The high-dose one-seed result is promising for native
impulse exposure but failed the velocity criterion, so it is not clean enforcement. The smallest
useful next experiment is one predeclared provisional-cap `p=0.2` bridge with a hard velocity stop.
Passing that screen would justify independent training seeds; failing it ends this dose path rather
than triggering a stronger dose automatically.

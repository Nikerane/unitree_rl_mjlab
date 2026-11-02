# Z1 Karolina migration and code review — 2 November 2026

The current `z1-safe-striking` code is installed on Karolina. The final focused
CPU run passed **243 tests**, with **one historical-evidence check skipped**.
No current-source GPU learning update, checkpoint reload on GPU, or physical
evaluation has been run there during this migration.

Active workspace: `/scratch/project/eu-26-66/nikhil/z1/current/source`.
It points to `z1-safe-striking/source` under the same Z1 project directory.
The tested implementation commit is `c4bf79b93ab6539c265e8ab8ddceb3ce6d2256f9`;
subsequent documentation updates do not change that implementation.

## What moved and what stayed the same

A fresh Git checkout replaced no existing release. The October releases,
historical outputs, Python environment and original diagnostic failures remain
preserved. A sibling link reuses the verified external robot/hammer/nail assets.
The current-workspace link and short README are navigation only; no new
scheduler, launcher, campaign controller or goal was created.

All three selected task rewards remain ON: normalized delivered impulse,
bounded native-depth progress, and once-only completion. Split 50/200 ms payout,
actor43/event critic47, ordinary learned Gaussian exploration, variable
impedance, the existing constraints and 200 ms aftermath remain unchanged.
Optional032 exploration code is present, but this migration runs no gSDE study.

## Evidence checked

| Check | Result and scope |
|---|---|
| Deployed source | Fresh tracked checkout of the pushed branch; clean before and after checks |
| Scientific source | All 151 `src/` Python files match the saved final reward037 source manifest |
| Selected dependency closure | Independent expanded 144-file static/helper/vendor audit found required files tracked and matching the final frozen source before the explicit input-check fix below |
| Evaluator dependencies | All 25 pinned vendor files match; actual frontend import and reader seams pass without physics |
| External assets | All 32 files match the saved reward037 asset hashes |
| Backend Python source | All 289 relevant mjlab/mujoco-warp/rsl_rl files match the saved Vega prepared031-v9 runtime manifest |
| Final Karolina CPU run | 243 passed, 1 skipped, 0 failed; 259.70 s at implementation commit `c4bf79b` |
| New binding regressions | Both evaluation test files together: 47 passed locally and on Karolina |
| GPU and physical behavior | Not checked in this migration; Python-source identity does not establish binary, kernel or trajectory equivalence |

The preservation commit intentionally excluded unrelated dirty legacy
controller/campaign edits and large evidence trees from the local worktree.
Audits therefore compared Git objects and the actual clone, rather than
assuming all local working files were pushed. The current selected pipeline's
dependencies were checked separately from historical script inventories.

## Standards review

No hard documented-standard violation or supported source-migration regression
was found in the bounded current-pipeline review. Existing private-helper reuse
and repeated recipe checks are historical maintainability costs; adding a
replacement framework would not improve this migration. No broad refactor was
performed.

One test-isolation cleanup now restores the module search path and removes
only newly loaded preserved-vendor modules after its frontend test. This keeps
the production loader's rejection of ambient modules intact and allows both
test files to run together reliably.

## Spec and correctness review

The reward, controller, constraint, checkpoint-clock and nominal-height-overlay
implementation matches the completed experiment's final source. The selected
height evaluations remain nominal-height comparisons, not height robustness.

**One historical P2 input-association defect was reproduced and fixed.** Opus
noticed that a valid checkpoint and a valid initial-state record from different
runs could be combined because same-setting seeds share the treatment contract.
Metadata-only synthetic tests reproduced acceptance without deserializing a
policy or running physics. This was a missing guard, not evidence that any saved
study result actually used a mismatched checkpoint.

Commit `c4bf79b` adds a small check in `record_for_checkpoint`:

- The checkpoint must be in the supplied initial state's same-run `train/` directory.
- Scientific checkpoints require the existing live-save receipt with matching
  SHA256, exact integer iteration and transition counter.
- Technical model0 requires the existing technical-result receipt with matching
  schema, status, selected cell, update/transition clock and SHA256.

No new receipt format was introduced. Ultra independently checked compatibility
with actual saved037 production/technical evidence and sampled036 receipts.
The change protects future input association; it does not change reward,
measurement, action, reset-bank or constraint calculations. Directory/receipt
consistency is not cryptographic authentication of every initialization field.

## The test timeout

The first Karolina run had 238 passes, one skip and one subprocess timeout at
40 seconds. The exact registry import/assertion subsequently passed in
93.582 seconds with a diagnostic time allowance. Commit `769f18c` changes only
that test's deadline to 180 seconds, with its command and assertions unchanged.
The targeted test and final combined run passed. Original failure and diagnostic
logs remain preserved; no physical safety threshold was changed.

## Requested model reviews

| Requested reviewer | Actual outcome |
|---|---|
| Opus | Completed; provider reports `claude-opus-5-5`. Found the association gap above; native and Ultra follow-up reviewed its fix. |
| Ultra | Completed independent source/runtime review and separate patch review. No remaining blocker found in the examined scope. |
| Fable | One tools-disabled invocation timed out after 900 s with no captured review. Actual model identity was not reported. |
| Kimi 3.1 Pro | One requested `opencode-go/kimi-k3.1-pro` invocation failed HTTP402, insufficient account funds; no completed review. |
| Gemini | One requested `opencode-go/gemini-4-argon` invocation failed HTTP402, insufficient account funds; no completed review. |

The full requested five-model panel is **incomplete**. Failed attempts were not
retried or counted as approvals. Prompts, raw responses, failures, metadata,
independent source checks and patch adjudication are retained in
`.superpowers/sdd/2026-11-02-z1-karolina-migration-review/` in the analysis worktree.
Provider packets predate the narrow fixes; independent native and Ultra
follow-up reviews cover those changes. Model review is advice, not experimental
safety evidence.

## Remaining boundary before new experiments

The next operational check is a current-source GPU update/save/reload check on
Karolina using the existing pipeline and correct allocated-GPU environment.
The technical checker qualifies actor/critic and environment-clock reload; it
does not qualify optimizer resume. Current training remains fresh-only.

These focused checks are not a full-repository test-suite claim. The previously
documented older exact global task-catalog assertion remains stale outside this
focused set. The historical height pilot's load violation and both029 failed
combined closures remain unchanged. Neither code review nor zero violations
in a finite saved bank certifies safety or hardware readiness.

The [handoff](../handoffs/2026-11-02-z1-karolina-code-handoff.md) contains the
environment commands, backup locations and same-run checkpoint layout.

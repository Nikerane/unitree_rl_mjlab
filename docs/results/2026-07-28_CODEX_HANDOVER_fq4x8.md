# fq4x8 — Codex handover (2026-07-28)

Written by Claude at the end of operational execution. Read this top-to-bottom before
touching anything. Every fail-closed boundary below is load-bearing.

---

## 0. TL;DR — where things stand

| Stage | State |
|---|---|
| Training (32 runs) | ✅ COMPLETE, 0 failed |
| Accepted-training manifest | ✅ FROZEN, sha256 `fb55f214…` |
| Evaluation attempt2 (job 40237539) | ✅ COMPLETE, `COMPLETED / 0:0`, 01:20:17, gn54 |
| Phase-2 read-only verification | ✅ PASS, 16,384 episodes, evidence frozen read-only |
| `build-evaluation-manifest` builder + ratified validity gates | ✅ IMPLEMENTED, 610 passed / 1 skipped full regression |
| Reviewer B (scientific/provenance) | ✅ COMPLETE — 1 CRITICAL + 2 IMPORTANT, **all resolved** |
| Reviewer A (correctness) | ✅ COMPLETE — 2 CRITICAL + 4 IMPORTANT, **all resolved** |
| Width-8 ratification reviews | ✅ GPT-5.6 independent review + Claude Opus review, **approved** |
| Initial width/sentinel commit + push | ✅ `9c2683eb550366bae61620434d1cab1bc44b5f1f` |
| Float32 replay correction | ✅ `6ae9ff3f60ca4e0d6b31cd5e675ecabf2b45acb2`, 610/1 regression |
| `accepted_evaluations.tsv` freeze | ✅ 32 rows, sha256 `8679604440712d…`, read-only |
| Task 9 analysis | ❌ NOT STARTED — **yours** |

The worktree `/private/tmp/unitree_rl_mjlab-first-strike-quality` is on branch
`first-strike-quality`. Initial hardening was committed at `9c2683e`; the reviewed
float32 correction was committed and pushed at `6ae9ff3`.

**NO TREATMENT OUTCOME HAS BEEN INSPECTED.** No arm mean, contrast, ranking, p-value,
success comparison or video selection has been computed by anyone. Keep it that way until
the manifest is frozen.

---

## 1. Frozen provenance — do not relabel any of these

```
training_code_revision    = 9cc63599a54788703e2348019b27908718912377
evaluation_code_revision  = ffac038c6ee5ad903fbd0feb33311b7c562699e4
manifest_builder_revision = 18c6d7c6ba442a0b4f7a096f442e0d74a129f410   (pre-builder; a NEW
                            builder revision must be recorded SEPARATELY after commit)
asset_revision            = b58ccd2f81fd246f27c1e8d88cf86484cd888703
accepted_manifest_sha256  = fb55f214d6e0cb2da308e6580ef535d4823038bc8ab842a05ca4085ab346ec14
evaluator RNG             = reset 2036072919 / observation 2046072933 / action 2056072941
```

Design: 4 arms × 8 seeds (8–15) = 32. Arms `F8 / F0 / D0 / FQ-min`, shorts `f8/f0/d0/fq`.
Weights (impact, delivered): F8 (8,2), F0 (0,2), D0 (8,0), FQ-min (8,0).
**D0 and FQ-min share weights (8,0) — weights ALONE cannot distinguish them; use `task`.**
FQ-min reward: `8 * q_contact * clip(v_precontact / 1.4598331451416016, 0, 1)`.

---

## 2. Immutable evidence (Vega, all `chmod a-w`)

```
~/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv   fb55f214…
~/unitree_rl_mjlab_eval/fq4x8/attempt2/                           32 artifacts + summary.csv
~/unitree_rl_mjlab_eval/fq4x8/attempt2_verification_report.json   3b39f2e7e32141e4b6abe54fd3c2eb89170ced06834eb9ca65787b64a85c0e58
~/unitree_rl_mjlab_eval/fq4x8/attempt2_artifact_inventory.sha256  442b4a55a478d12ce7ac56d6a23bb9b633f5ba3a0c4bb070538e0522b4946f73  (35 entries)
```

`attempt1` is **immutable failed evidence** — a persistence-capacity defect. Never reuse,
repair, or cite its rows. attempt2 is a full from-scratch rerun of all 32 checkpoints.

Phase-2 verified, all recomputed with the SHIPPED validators imported from the pinned
`ffac038` checkout (never reimplemented): 32 rows, 4×8 identity complete, 512 episodes per
checkpoint, 16,384 schema-v3 episodes, every checkpoint SHA cross-bound to the frozen
training manifest, every artifact SHA + `sampled_trace_digest` + per-episode
`reset_state_digest` + `trace_digest` recomputed, RNG identities match, no `-dirty`,
`impossible_success_n == 0`, `lambda_dead_n == 0`.

Four tamper negative-controls confirm the battery actually fails when it should.

---

## 3. What was built (uncommitted)

Five named files:

- `evaluation/analysis/fq4x8_manifests.py` — new section from the banner
  `build-evaluation-manifest:` onward: `_decode_sampled_artifact`, `_is_finite_value`,
  `_derive_validity_sentinels`, `_require_clean`, `build_evaluation_manifest`,
  `write_evaluation_manifest`, CLI `build-evaluation-manifest`; plus imports
  (`csv`, `math`, `Counter`, `numpy`).
- `evaluation/analysis/first_strike_quality_campaign.py` — the same fail-closed raw-quality
  validator shared by the manifest freeze and Task 9, including the exact eight-slot gate.
- `tests/test_fq4x8_evaluation_builder.py` — builder, sentinel, identity, and width-gate
  regression coverage; RED observed before each implementation step.
- `tests/test_first_strike_quality_campaign.py` — direct Task-9 width 1/7/9 rejection
  coverage and realistic eight-slot fixtures.
- `docs/results/2026-07-28_CODEX_HANDOVER_fq4x8.md` — owner-ratified definitions and the
  current operational handover.

Emits exactly the existing 39-field `EVALUATION_FIELDS`, reusing the existing
serializers and `validate_evaluation_manifest`. No new reward, physics, evaluation or
statistical behaviour.

### Digest strategy (important to understand before reviewing)
`payload_digest` is the canonical digest of the WHOLE schema-v3 payload, which already
contains every episode's `trace_digest` and `reset_state_digest`. Recomputing it binds all
512 per-episode digests per checkpoint without re-walking them. That those banked digests
*describe the physics* was established separately in Phase 2 and frozen.

### Derived sentinels — owner-ratified 2026-07-28
The preregistration names four sentinels but does not define their derivation.
After a round-table review by Claude Opus 5, Gemini Pro, GPT-5.6 Ultra, an
adversarial reviewer, and an independent moderator, the owner explicitly
ratified the following definitions:

| sentinel | derivation |
|---|---|
| `quality_overflow_n` | episodes with a genuine accepted-onset capacity overflow: well-formed raw/event/snapshot records all agree `True`; post-onset overflow is outside this sentinel |
| `quality_nonfinite_n` | episodes with missing, malformed, non-finite, wrong-shaped, or mutually inconsistent required quality evidence; this legacy name is an evidence-integrity count, not only a floating-point-finiteness count |
| `liveness_failure_n` | either dense tracker-quality or dense `physical.contact` stream is absent, malformed, empty, or their lengths differ |
| `quota_failure_n` | the exact `(env_id, episode_ordinal)` population is not `0..255 × {0,1}`, once each |

`liveness_failure_n` certifies recording/slicing structure only. It does not
mean contact occurred and does not independently prove functional sensor
responsiveness. A complete no-contact trace remains valid. Separately, every
accepted artifact must carry the frozen **eight-slot** raw quality shape; this
outcome-blind structural check prevents the one-slot missing-sensor fallback
from masquerading as a weak policy.

No-contact and zero-positive-normal-force episodes remain valid finite
zero-quality episodes when their complete evidence is present. General depth,
six-joint Lambda, delivered-impulse, and qvel numerical validity fail closed
separately.

`impossible_success_n` and `lambda_dead_n` are task-specific cross-channel
consistency alarms, not universal physical laws. Their sampled components are
recomputed from every digest-bound episode using the evaluator's exact-zero
predicates. Exact nonnegative mean-rollout counts are taken from the
digest-bound payload; they are tamper-evident but not independently
reconstructable because raw mean-rollout traces were not retained. The
unhashed `summary.csv` totals must agree exactly but are redundant evidence,
never the source of truth.

Any genuine accepted-onset overflow or other registered validity failure makes
the complete preregistered confirmatory family **non-evaluable**. All evidence
is retained for clearly labelled forensic/descriptive diagnosis, but no seed,
row, or favorable contrast may be salvaged as confirmatory.

---

## 4. Defects found and fixed (7)

By reading:
1. Sentinels were emitted as hard-coded zeros instead of derived counts.
2. No artifact↔row binding (task / training_seed / rng_streams now checked).

By the real-data dry run (both passed all 33 synthetic tests because the fixture encoded the
same wrong assumption as the code — tests and implementation agreed with each other and
disagreed with reality):

3. Artifact `treatment` is the evaluator's own label (`"F"` for arm `F8`), not the arm label.
   Check removed rather than adding a drift-prone 5th mapping; `task` is unique per arm.
4. `episode_peak_lambda` is a 6-wide **per-joint vector**, not a scalar — the finiteness
   check crashed on real data.

By Reviewer B (all independently reproduced before fixing):

5. **CRITICAL** — the artifact was bound to neither the attempt nor the checkpoint.
   `sampled_trace_path` accepted ANY absolute path, and the payload's own
   `provenance.checkpoint_sha256` was ignored. A row could point at another attempt's
   artifact and self-consistently re-hash it, yielding a manifest labelled one attempt whose
   rows are a MIXTURE of attempts — artifact-level post-outcome selection, below the layer
   seed-level checks police. Fixed: containment under the attempt root + checkpoint
   provenance binding.
6. **IMPORTANT** — `_require_clean` failed OPEN on a blank value and on a missing column.
   Fixed: absent or blank provenance now raises.
7. **IMPORTANT** — the two nonfinite tests never reached the code they claimed to cover
   (the digest guard fires first). Rewritten to use JSON-legal `null`.

Also hardened per Reviewer B: liveness pinned to the dense substep series (so a future
switch to sparse recording fails loudly instead of silently filtering weak seeds); the weak
fixture corrected to `contact_quality_valid = False` (it was physically impossible before);
and the load-bearing test strengthened.

By Reviewer A (all reproduced as failing tests before fixing — 14 RED, then GREEN):

8. **CRITICAL** — `write_evaluation_manifest` never ran the real validator. Rows with a wrong
   `task` and a nonzero sentinel serialized, published and returned a SHA, because
   `parse_evaluation_manifest` only checks structure and types. It now REQUIRES
   `training_rows` + `training_manifest_sha256` and runs `validate_evaluation_manifest`
   both before writing and on the reparsed bytes.
9. **CRITICAL** — the four evaluator config-identity columns in `summary.csv` were never
   checked, and the builder emitted the TRAINING manifest's values, masking any drift.
   **Resolution is nuanced, read before changing:** `fixed_action_signature_sha256` and
   `fixed_impedance_signature_sha256` really are the same quantity in both files (verified
   byte-identical on all 32 real rows) and are now cross-bound. But
   `campaign_config_sha256` / `treatment_config_sha256` differ on **all 32** real rows with
   different cardinality (3 and 4 distinct values) — the signature of a DIFFERENT
   DERIVATION, not of drift. Cross-binding those would encode a false identity, so they are
   instead checked for **per-arm consistency** (all eight seeds of an arm must agree), which
   still catches a checkpoint evaluated under an odd config. Do not "fix" this by asserting
   equality.
10. **IMPORTANT** — lossy `int(float(...))` coercions: `schema_version=3.9` passed as 3,
    `training_seed=8.9` passed as 8, and sentinel `"0.5"` truncated to a passing `0`.
    Replaced with `_exact_int` (JSON ints must be `int`; text must match an exact
    nonnegative-integer grammar; booleans rejected) at all six sites.
11. **IMPORTANT** — quota counted only per-env totals, so 512 episodes that were two copies
    of ordinal 0 from every env passed. Now compares the exact `(env_id, episode_ordinal)`
    pair set, each exactly once.
12. **IMPORTANT** — `[]` and `True` counted as finite measurements. `_is_finite_value` now
    rejects booleans and empty sequences.
13. **IMPORTANT** — a fixed `.tmp` name let a concurrent writer swap the published bytes
    after hashing. Now uses an exclusively-created unique temp in the destination directory.

### Key evidence that the hardening is sound
The hardened builder produces a **byte-identical** manifest on the real attempt:
`e92766265796218035e07bcef878c7b5c09de15072952fd947b92db755de6a60`, unchanged from before
hardening. The new checks reject forgeries **without** altering legitimate output.

### The load-bearing safeguard
`test_performance_changes_nothing_but_content_addresses` builds all-32-weak vs all-32-strong
attempts and asserts the rows differ in **exactly two columns** — `sampled_trace_digest` and
`sampled_trace_artifact_sha256`, both content addresses. This proves no performance quantity
reaches ANY emitted column. **Do not weaken this test to make another one pass.**

---

## 5. YOUR TASKS

### Manifest-freeze invocation audit (2026-07-28)

Publication invocation 1 used clean builder revision
`9c2683eb550366bae61620434d1cab1bc44b5f1f` against immutable evaluation
attempt2. It failed before writing `accepted_evaluations.tsv`:

```
MANIFEST_FAIL: evaluation attempt: F8/seed8:
sentinel quality_nonfinite_n is nonzero (506)
```

This was a manifest-validator defect, not an evaluation retry or campaign-row
failure. The validator recomputed producer-native Torch CUDA float32 latches
with NumPy/float64 and an inappropriate absolute tolerance. A Torch float32
replay demonstrated that the banked evidence was valid; no checkpoint,
evaluation artifact, or frozen input changed. No outcome aggregate, contrast,
ranking, p-value, or video was inspected.

The correction now:

- replays the producer operation order in Torch float32;
- uses fixed field-specific, `rtol=0` forward-error bounds only for
  raw-to-snapshot recomputation;
- keeps every event-to-snapshot latch exact and immutable;
- rejects nonnumeric, nonfinite, out-of-float32-range, wrong-width, malformed,
  and out-of-domain evidence;
- derives genuine overflow independently from malformed-quality and liveness
  sentinels.

Verification: direct analysis `91/91`; builder `249/249`; original four-suite
regression `610 passed, 1 skipped`. Two independent reviews found no remaining
reproducible fail-open blocker.

Publication invocation 2 at clean detached Vega builder revision `6ae9ff3`
succeeded with exactly 32 rows and all six sentinels zero. Canonical validation
passed, then these files were frozen read-only:

- `accepted_evaluations.tsv`:
  `8679604440712d276996b8768842fa2358b8119c798ab94208c7a504a4f34136`
- `accepted_evaluations.tsv.sha256`:
  `3ebdbff4d918005253813b99330cf2ec52bb78b676458f2714b58cca1c0dbe37`
- `accepted_evaluations_inventory.sha256`:
  `ae4ff87dc4f40cb4193a993ce47fcf5bd46e39be99d21ba9f214d16792cbef0c`

The earlier expected SHA `e927662…` was reproduced from the preserved local
dry run and diagnosed exactly: its 32 retry-history cells contained only
`attempt2:accepted`. Replacing those cells with the frozen lineage
`attempt1:infra_failure;attempt2:accepted` makes it byte-identical to the Vega
manifest and yields `86796044…`. No other byte differs; therefore the mismatch
was a stale provenance-only expectation, not evidence or outcome drift.

### Task A — Phase 4 is COMPLETE; nothing to redo
Both reviews are done and every Critical/Important finding is resolved with a test.
Note for any future review: one Reviewer-A dispatch reviewed the WRONG worktree
(`~/repos/unitree_rl_mjlab` has none of this code) and was discarded. Always work in
`/private/tmp/unitree_rl_mjlab-first-strike-quality` and verify first with:
```
cd /private/tmp/unitree_rl_mjlab-first-strike-quality && grep -c build_evaluation_manifest evaluation/analysis/fq4x8_manifests.py
```

### Task B — Phase 5 freeze
1. Run the full regression, then commit and push **only** these five named files:
   `evaluation/analysis/fq4x8_manifests.py`,
   `evaluation/analysis/first_strike_quality_campaign.py`,
   `tests/test_fq4x8_evaluation_builder.py`,
   `tests/test_first_strike_quality_campaign.py`, and
   `docs/results/2026-07-28_CODEX_HANDOVER_fq4x8.md`.
   Stage named files only — never `git add -A`. **No `Co-Authored-By` trailer.**
2. Record the NEW manifest-builder revision separately. Do NOT relabel the training or
   evaluation revision.
3. Deploy a clean detached checkout on Vega (`$HOME/campaigns/fq4x8_evalmanifest_<sha>/`),
   never scp tracked source.
4. Run one publication attempt for the newly reviewed builder revision against
   immutable attempt2. Failed pre-publication invocations remain in this audit
   trail; they are not evaluation retries:
```
python -m evaluation.analysis.fq4x8_manifests build-evaluation-manifest \
  --attempt-dir  $HOME/unitree_rl_mjlab_eval/fq4x8/attempt2 \
  --training-manifest $HOME/unitree_rl_mjlab_eval/fq4x8/accepted_training_checkpoints.tsv \
  --training-manifest-sha256 fb55f214d6e0cb2da308e6580ef535d4823038bc8ab842a05ca4085ab346ec14 \
  --evaluation-attempt attempt2 \
  --evaluation-retry-history 'attempt1:infra_failure;attempt2:accepted' \
  --expected-evaluation-code-revision ffac038c6ee5ad903fbd0feb33311b7c562699e4 \
  --out $HOME/unitree_rl_mjlab_eval/fq4x8/accepted_evaluations.tsv
```
5. Require exactly 32 rows and the frozen 4×8 identity. Validate with the canonical validator.
6. Bank the `accepted_evaluations.tsv` SHA-256; make it and a SHA sidecar read-only; add both
   to the evidence inventory; update the ledger.

Authoritative SHA:
`8679604440712d276996b8768842fa2358b8119c798ab94208c7a504a4f34136`.
The retired dry-run SHA `e927662…` omitted the failed-attempt lineage; see the
byte-exact diagnosis above.

### Task C — Task 9 analysis (yours from the start)
Only AFTER the manifest is frozen. Statistics, figures, video selection.
Decision-eligible fields are ONLY the five `*_sampled` fields in the prereg. Primary
contrasts: `F0−F8`, `D0−F8`, `FQ-min−D0`. Any nonzero sentinel, non-schema-v3 artifact,
dirty provenance, or incomplete quota invalidates the **complete** campaign analysis, not
just the offending row.

---

## 6. Standing guardrails

Fixed impedance only — never `set_gains` or command stiffness. Keep all manufacturer
`IMP_J_LIMIT` caps unchanged. Keep `imp_max_p = 0.0` (log-only). No superlinear
excess-over-reference reward. Never edit installed `mjlab`/`rsl_rl`/`mujoco`/`mujoco_warp`.
Never accept dirty or `-dirty` provenance. Never replace a completed weak seed. Retry only
documented infrastructure failures, with identical seed/config. Unset and reject inherited
`IMPACT_W`, `DELIVERED_W`, `NAIL_DRIVEN_W`.

---

## 7. Owner decisions resolved 2026-07-28

1. The corrected sentinel meanings in §3 are ratified.
2. The exact eight-slot raw quality shape is a freeze-blocking,
   outcome-blind structural prerequisite applied uniformly to all 32
   checkpoints.
3. `impossible_success_n` / `lambda_dead_n` use digest-bound sampled and
   mean-rollout evidence; `summary.csv` is an exact redundant agreement check.
4. Eight slots remain an informative-censoring limitation. Zero observed
   accepted-onset overflow validates capacity only over this campaign's
   realized behavior distribution, not universally.
5. Any registered validity failure makes the complete confirmatory family
   non-evaluable. Evidence remains available only for quarantined forensic and
   descriptive reporting; selective seed, row, or contrast salvage is
   forbidden.

Durable ledger with the full chronological record:
`.superpowers/sdd/2026-07-26-fq4x8-claude-execution/progress.md`

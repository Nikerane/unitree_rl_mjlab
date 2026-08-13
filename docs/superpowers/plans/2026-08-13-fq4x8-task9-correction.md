# fq4x8 Task-9 Ratio Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the preregistered paired arm-mean ratio bootstrap, re-run the immutable fq4x8 Task-9 analysis without retraining, and bank the corrected conclusions.

**Architecture:** Keep all 32 training runs and 16,384 evaluation episodes immutable. Change only the pure statistical helper and its regression tests, then run the existing analysis against the frozen manifests in a clean detached Vega checkout. Bank a new analysis bundle instead of overwriting the original bundle.

**Tech Stack:** Python 3.12, NumPy PCG64, pytest, Matplotlib, Git, Vega CPU execution.

## Global Constraints

- Base code revision is `ca5e83bc84bc979bac857088a9330f2a2bd8034b`; work only on branch `codex/fq4x8-task9-correction` in `/private/tmp/unitree_rl_mjlab-fq4x8-task9`.
- Do not modify or regenerate training checkpoints, evaluator JSON/NPZ artifacts, accepted manifests, or the original `task9_0fba76b` result bundle.
- Preserve `samples=100_000`, `seed=20_260_726`, matched-seed resampling, NumPy `PCG64`, and the registered 5th/2.5th/97.5th quantiles.
- A ratio is valid only when the observed control arm mean and every resampled control arm mean are finite and strictly positive.
- One zero-valued control seed is not itself an error when all observed/resampled arm means remain positive.
- The corrected analysis must continue to reject FQ-min as a practical replacement because useful-speed preservation fails, and continue to reject the D0 mechanism claim because its success guardrails fail.
- Re-run only CPU analysis/plotting; no GPU allocation, training, or policy evaluation.
- Publish to a new collision-safe result directory and preserve all attempt logs and hashes.

---

### Task 1: Correct the paired ratio bootstrap test-first

**Files:**
- Modify: `evaluation/analysis/first_strike_quality_campaign.py`
- Modify: `tests/test_first_strike_quality_campaign.py`

**Interfaces:**
- Consumes: `paired_bootstrap_ratio(treatment, control, *, samples=100_000, seed=20_260_726)`.
- Produces: the same result mapping with `estimate`, `one_sided_95_lower`, `two_sided_95_interval`, `samples`, and `seed`; no caller or schema changes.

- [ ] **Step 1: Write the failing zero-seed regression**

Add a test beside `test_holm_mapping_and_paired_pcg64_bootstrap` that uses matched finite vectors with one zero control seed and seven positive control seeds. Freeze the expected result as independently audited literal values rather than recomputing the production RNG/mean/quantile algorithm inside the assertion. Keep the existing all-zero and non-finite denominator rejection cases.

Add a second deterministic guard case whose observed control arm mean is positive but whose fixed PCG64 resamples include a zero control arm mean; independently assert that such a resample exists, then require the helper to reject it. This freezes the new fail-closed resampled-denominator contract rather than testing only the accepted path.

- [ ] **Step 2: Run the new regression and verify RED**

Run:

```bash
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_first_strike_quality_campaign.py -k 'paired_pcg64_bootstrap or zero_control_seed or zero_resampled_arm_mean'
```

Expected: the new regression fails with `ValueError: ratio bootstrap requires a positive denominator` from the old per-seed guard.

- [ ] **Step 3: Implement the minimal arm-mean guard**

In `paired_bootstrap_ratio`, validate `samples` before allocating indices; compute the observed treatment/control means; reject non-finite or non-positive observed control mean; draw matched indices; compute resampled treatment and control arm means; reject any non-finite or non-positive resampled control mean; divide the resampled arm means. Do not change the RNG, quantiles, return keys, or seed matching.

- [ ] **Step 4: Add campaign-shaped decision regression**

Use the frozen eight-seed aggregate vectors from the corrected Task-9 audit and assert:

- FQ-min/F8 useful-speed estimate `1.020483` and one-sided lower `0.903143` (tolerance `1e-6`), gate false.
- FQ-min/F8 depth estimate `1.130463` and one-sided lower `1.003629` (tolerance `1e-6`), gate true.
- D0/F8 depth estimate `0.991462` and one-sided lower `0.988661` (tolerance `1e-6`), preservation true.
- FQ-min practical acceptance remains false and D0 mechanism claim remains false.

Use literal frozen vectors from `analysis.json`; do not call the production helper to construct expected values.

- [ ] **Step 5: Run focused GREEN tests**

Run:

```bash
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_first_strike_quality_campaign.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add evaluation/analysis/first_strike_quality_campaign.py tests/test_first_strike_quality_campaign.py
git commit -m "fix(analysis): bootstrap fq4x8 arm-mean ratios"
```

### Task 2: Review and qualify the corrected analysis code

**Files:**
- Review only: Task-1 commit.

**Interfaces:**
- Consumes: Task-1 commit and frozen Task-9 statistical specification.
- Produces: independent Standards and Spec PASS, or bounded test-first fixes followed by re-review.

- [ ] **Step 1: Run the bounded analysis suites**

```bash
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_first_strike_quality_campaign.py \
  tests/test_fq4x8_manifests.py \
  tests/test_fq4x8_evaluation_builder.py
git diff --check ca5e83bc84bc979bac857088a9330f2a2bd8034b..HEAD
```

- [ ] **Step 2: Perform independent Spec and Standards reviews**

Review the fixed commit against the constraints above. Any Critical/Important finding must receive a failing regression, the smallest production fix, and a scoped re-review before proceeding.

- [ ] **Step 3: Freeze the reviewed candidate SHA**

Record the full candidate SHA and the exact test outputs. Push only after all review findings are closed and the worktree is clean.

### Task 3: Add a tracked, atomic Task-9 CLI

**Files:**
- Create: `evaluation/analysis/fq4x8_task9.py`
- Create: `tests/test_fq4x8_task9.py`

**Interfaces:**
- Consumes: immutable accepted-training manifest SHA `fb55f214d6e0cb2da308e6580ef535d4823038bc8ab842a05ca4085ab346ec14`; immutable accepted-evaluation manifest SHA `8679604440712d276996b8768842fa2358b8119c798ab94208c7a504a4f34136`; immutable attempt-2 summary SHA `3e2469627ed0daa94c98a94eced580ca87d581d2a721bca97893d9ea03eeaa3a`.
- Produces: `load_frozen_inputs(...)`, `run_task9(...)`, and `main(argv=None)`, plus a canonical result directory with `analysis.json`, exactly two PNGs, and a sorted three-entry SHA-256 manifest.

- [ ] **Step 1: Add and test a tracked Task-9 CLI**

Create a narrow tracked CLI which accepts only `--accepted-training-manifest`, `--accepted-evaluation-manifest`, `--summary`, and `--output-dir`. Read each input once as bytes and verify all three hard-coded hashes before decoding. Reuse `fq4x8_manifests` parsers/validators and `render_quality_figures`; do not copy analysis logic or run the analyzer twice.

Reject any existing output inode with `os.path.lexists`, including dangling symlinks. Create a unique sibling staging directory, serialize sorted two-space JSON with a final newline and `allow_nan=False`, require exactly the two registered PNGs, write a lexicographically sorted three-entry basename-only checksum manifest, then publish the complete directory with same-filesystem `os.rename`. Remove only the owned staging directory after pre-publication failure. Emit runtime/input/output provenance as one JSON object on stdout, outside the canonical bundle.

Test first: exact four-argument CLI; one-byte drift in each input; cross-inconsistent hash-correct manifests; preexisting file/directory/dangling symlink; invalid analysis/unexpected renderer outputs; serialization/publication failure cleanup; exact successful four-file inventory and independently verified manifest; runtime provenance excluded from the bundle; one real full-shape fixture integration.

- [ ] **Step 2: Review and commit the CLI**

Run the new CLI tests plus the quality/manifests suites. Perform independent Spec and Standards reviews. Fix Critical/Important findings test-first, then commit only the module and test.

### Task 4: Re-run corrected Task 9 once on Vega

**Files:**
- Create on Vega: one new immutable Task-9 result directory under `/ceph/hpc/home/eunikhilr/unitree_rl_mjlab_eval/fq4x8/`.
- Create on Vega: separate Slurm stdout/stderr and command/runtime provenance outside the canonical result directory.

**Interfaces:**
- Consumes: reviewed Task-1 + Task-3 candidate SHA and the three immutable frozen inputs.
- Produces: one verified corrected Task-9 bundle; no training or policy evaluation.

- [ ] **Step 1: Show the exact command before submission**

Resolve and print the clean detached checkout, immutable manifest paths/hashes, new collision-safe output directory, analysis invocation, timeout, and monitored stdout/stderr paths. Do not run until those exact values have been surfaced in the active conversation.

- [ ] **Step 2: Run CPU-only Task 9 once**

Use the new tracked CLI and immutable attempt-2 artifacts. Monitor process-alive and the declared timeout. Do not auto-retry a crash.

- [ ] **Step 3: Verify the new bundle independently**

Require all manifest hashes, 32 seed aggregates, 16,384 sampled episodes, all sentinels zero, corrected ratio values/gates, unchanged primary Holm/sign-flip conclusions, and exactly two preregistered figures. Compare old and corrected decoded `analysis.json` after replacing only the three ratio records and their dependent gate booleans: every primary contrast, seed aggregate, valid-contact coordinate, nail geometry, and all unrelated fields must be exactly equal. Verify that the analysis dependency closure under `evaluation/analysis/` is unchanged between `0fba76bca7a10e46a618f0715db15da3f520a7b9` and the candidate except for the reviewed ratio helper and tracked CLI. Record Python, NumPy, and Matplotlib versions. Require the contact-map bytes to remain unchanged under the matched runtime and require the paired-effects figure to change because the two formerly unavailable ratio bars become visible.

### Task 5: Bank the corrected result and update current truth

**Files:**
- Create: one dated result record under `docs/results/`.
- Create: a compact result package under `docs/results/assets/`.
- Create: one compact integrity test.
- Modify: `docs/results/README.md`.
- Modify: `docs/results/2026-07-28_CODEX_HANDOVER_fq4x8.md` by appending a dated completion/correction addendum only.

**Interfaces:**
- Consumes: independently verified Task-4 remote bundle and provenance.
- Produces: a conservative, hash-bound local result record; no reward-stack change.

- [ ] **Step 1: Bank the corrected result test-first**

Add a compact integrity test that independently loads the banked JSON, recomputes the corrected ratios/decisions, verifies every local artifact hash, and checks the dated record/index link. The canonical remote core inventory is exactly `analysis.json`, `paired_seed_effects.png`, and `aggregate_nail_plane_contact_map.png`, with a sorted three-entry checksum manifest; command, Slurm log, runtime versions, and input hashes are separate provenance artifacts. Locally bank deterministic `analysis.json.gz` plus the decompressed JSON SHA rather than an uncompressed 11.13 MB duplicate, unless an explicit size audit supports the raw copy. Write the result record with the unit of inference (`n=8` training seeds), null-is-not-equivalence caveat, old Cartesian/fixed-impedance applicability boundary, and outcome-stable bug correction. Add a dated completion/correction addendum to the historical 2026-07-28 handover; do not rewrite statements that were true at handover time.

- [ ] **Step 2: Commit and final-review the bank**

Run the result test, docs-current tests, bounded analysis suites, and `git diff --check`; then perform final Spec/Standards review before pushing the bank commit.

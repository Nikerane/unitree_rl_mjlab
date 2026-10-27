# Vega Resident Policy Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete six-policy evaluation data on Vega with independent streamed verification, then continue the authorized staged evaluation without bulk Mac downloads.

**Architecture:** New external stdlib archive tools create an exclusive full Vega backup of each terminal run/private pair, verify it on Vega and independently read every saved byte on Mac. Full indexes, controls and normalized consumer receipts stay on Vega; Mac retains bounded compact records and graphs/videos. The frozen physical ten-file package, first-then-five qualification consumer and scientific inputs remain unchanged.

**Tech Stack:** Python stdlib compatible with Vega Python3.9 and Mac Python3.14; gzip/tar/JSON/SHA256; SSH; pure unittest fixtures; existing CPU supervisor/sealer contracts.

**Spec:** `.superpowers/sdd/2026-10-23-z1-overnight-execution/evaluation-vega-resident-preservation-preparation003/DESIGN.md` and `LIFECYCLE-DESIGN.md`; current human execution instruction in sibling `evaluation-vega-resident-preservation-implementation004/human-instruction001.json`.

## Global Constraints

- Use only `/Users/nikerane/.codex/worktrees/a370/unitree_rl_mjlab`; root operator01a178a1-b70b-7cf2-9e57-da2011f96d75 owns canonical/admission writes. Existing automation stays PAUSED; no competing automation or native goal.
- Frozen proposal002's201files/64dirs and all training/failed evidence remain byte/metadata exact. No retraining, deletion, SSD write or policy/task/controller/reward change.
- Physical package stays exactly10files, supervisor SHA565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd. Use its fresh unconsumed staged-integration002 paths/claims only after live absence checks.
- Up to6fresh once-only own32 invocations/192engineering starts, first[0], remaining[3,6,9,12,15] only after actual first full qualification/reviews/backup/seal. Any physical failure stops unissued stages; no automatic retry or old claim/job reuse.
- Original science remains1920starts: both banks20261041/2026101601 perpolicy, allfive exact cases, own32seal perpolicy. Native800/inclusive801, whole0–200ms velocity/per-joint impulse safety and utility stay explicit.
- Archive caps: source167772160B, compressed201326592B,8192nodes, metadata4194304B, evidence8388608B, wall≤600s. Exact run/private roots and genuine saved scheduler identity are mandatory.
- Vega hard future minimum11274289152B plus570425344B floor/controls stays intact before and during writes. Ledger covers all remaining raw/private, separate archives, full provenance and derived outputs. No apparent filesystem-wide free-space substitution.
- Proposed Mac envelope1400897536B:512MiB floor+32MiB helper controls+88MiB working controls+640MiB media/derived+64MiB aggregate retained compact controls. Enforce retained64MiB across the explicit output root. No full archive or raw extraction locally.
- Full archive includes all files, empty dirs, symlinks as metadata, logs, failures, partials and caches; never follow archived symlinks, import payloads or omit unknown files. Preserve mode/uid/gid/nsmtime and available original xattrs/restore metadata, with source inode/device/ctime/link identity evidence and disclosed read-induced atime differences.
- Only bounded pure tests before actual approved qualification; tests must not construct an environment, import Torch/MuJoCo or execute a policy. Existing unrelated dirty files remain untouched. Commit only this plan's explicit new source paths.

## Review Focus

- A streamed gzip/tar ending early or hiding extra payload must not produce backup PASS; exact full compressed SHA and EOF/CRC/tail are checked.
- A complete archive of the wrong policy/private root/job/release must not satisfy the unchanged qualification consumer.
- A source mutation or failed archive/control publication leaves originals and partial evidence intact, with no consumer receipt or remaining-stage admission.
- Local controls accumulate across18slots and remote full provenance remains allocated; both aggregate budgets require real enforcement.
- Remote-only bulk storage must still support actual raw/NPZ audits, independently checked receipt publication, and the original first-then-five gate.

---

### Task 1: Bounded nonseekable archive verification

**Files:** Create `scripts/vega_policy_archive/stream_reader.py`, `tests/test_stream_reader.py` and `__init__.py` within that package. Test invocation uses its own tests directory.

**Interfaces:** `verify_stream(stream, before, *, max_archive_bytes, max_source_bytes, max_metadata_bytes, application_names=(), max_evidence_bytes=8388608, deadline=None) -> dict`. Returns exact `origin_map`, parsed `application_records`, full `archive_sha256`, `archive_bytes`, node/file/byte counts; returns only after complete validation. `before['nodes']` is the existing two-root inventory map; its rows carry source_path/type/mode/uid/gid/mtime_ns and file bytes/SHA or symlink target, with optional exact restore metadata validated when present. No networking or filesystem extraction.

- [ ] Write failing pure tests for a two-root fixture with file/directory/symlink metadata and bounded application JSON; nonseekable reader must succeed without any local archive.
- [ ] Observe RED, then implement bounded chunk reads, source cap/member cap/metadata bounds, complete compressed SHA/size and exact file/node validation. JSON records are checked in memory, individually≤evidence cap, aggregate≤2×evidence cap.
- [ ] Add RED→GREEN negatives for missing/extra/duplicate/unsafe/hardlink members, body/metadata/restore drift, truncated gzip, trailing nonzero/extra compressed payload, oversized PAX/decompression, invalid/oversized JSON and deadline/byte exhaustion. Preserve RED evidence.
- [ ] Run all this package's pure tests normally and under `python -O`, compile via explicit source without pycache; commit only Task1paths. Report test evidence and limits for review.

### Task 2: Exclusive Vega archive and control store

**Files:** Create `scripts/vega_policy_archive/resident_store.py`, `tests/test_resident_store.py`; copy the frozen002 `archive_contract.py`, `remote_policy_archive.py`, `policy_archive.py` into an explicit compatibility subpackage after the first test RED. The legacy sources remain exact except an explicit optional record-reader seam in `validate_archived_application_records`, covered by old/new behavior tests.

**Interfaces:** `create_archive(request, destination, *, inventory_reader, free_reader, deadline) -> dict`; `publish_control(destination, data, expected_sha256, *, free_reader, deadline) -> dict`. CLI remote actions validate exact CPU roots/layout/terminal/scheduler identity using the frozen validators, and restrict destinations to an explicit separate campaign backup root. Full before/after inventory and reader result bind stored archive SHA/size and source stability. Dependency callbacks allow synthetic roots/quota in pure tests; CLI has no unrestricted arbitrary path escape.

- [ ] Write failing synthetic tests that create a complete archive+primary verification at a new private destination and preserve originals. Add tests for mandatory distinct raw/private/backup paths and all-parent nonsymlink validation.
- [ ] Observe RED, implement exclusive0600 creation, per-chunk compressed cap/quota guard, fsync, complete source content/metadata checks before/after and full primary verify via Task1reader. Preserve partial archive and explicit failure evidence; no overwrite/deletion/retry or PASS publication on failure.
- [ ] Preserve exact available Linux restore metadata in the index/PAX; test xattrs and identity/atime semantics without silently weakening source drift checks. Test source replacement, symlink/hardlink policy, private extras and empty directories.
- [ ] Test control publication hash/size, exclusive conflicts, actual quota exhaustion/short write/fsync/deadline failures, wrong terminal/policy/layout/job identities and stream-back of only a bound complete archive. Run full package tests normal/-O and commit only Task2paths.

### Task 3: Mac coordinator, capacity and unchanged consumer integration

**Files:** Create `scripts/vega_policy_archive/resident_archive.py`, `capacity.py`, `tests/test_resident_archive.py`, `tests/test_capacity.py`, package README and file-hash manifest. Retain original archive request validation and application-record logic; the coordinator calls the new backend and Task1reader rather than the old local archive writer.

**Interfaces:** `run(request_path, output) -> dict`; new request schema `z1_horizon_vega_resident_archive_request_v1` includes explicit remote archive/control root, helper package binding and preservation-mode/capacity ledger references. It carries all existing actual manifest/release/layout/scheduler/terminal/shared-source bindings. Produce a full normalized `z1_horizon_terminal_multiroot_archive_receipt_v1` on Vega with identical member/tree/policy fields required by `verify_qualification`, plus explicit remote archive location/full independent Mac readback. Retain a compact local receipt≤1MiB with bound remote artifact references. `capacity.py` exposes pure `remaining_budget(ledger)` and guarded exclusive aggregate control writes; no pending or failed science slot disappears from the ledger.

- [ ] RED→GREEN tests for validate→remote exclusive archive→independent nonseek stream→complete bound application records→publish/readback full receipt on Vega→compact local receipt. Inject a synthetic transport that exercises real serialization/archive gates; separately test owned SSH processes, stdout/stderr bounds, timeout kill/reap and nonzero exits without SSH to real hosts.
- [ ] Add RED→GREEN negatives for source/request/package/observation/authority drift, wrong job/policy/private data, partial publication, forged backup flags, expired deadlines and full local64MiB aggregate exhaustion. No local tar/NPZ output on success or failure.
- [ ] Test exact remaining18slot ledger, per-slot raw160MiB/backup192MiB/provenance88MiB, hard Vega future+floor and Mac working/retained/media budgets; include staging and actual retained allocations explicitly. Validate fake successful streamed receipts through the unchanged real supervisor qualification/remaining gate using synthetic fixture files, including mandatory first seal and raw NPZ audit evidence.
- [ ] Run full pure package tests normal/-O and relevant frozen002 pure integration suites; record full results and any explicit exclusions. Commit only Task3paths and source package manifest.

### Task 4: Complete reviewed successor controls and preservation

**Files:** New files only under existing campaign `evaluation-vega-resident-preservation-implementation004/`: capacity ledger, helper staging map/package manifest, derived prospective manifest/release/authority projections, templates, test/review receipts and full proposal index. Reuse byte-identical physical10/sealer/parent/first/five scripts and exact unused002identities; new external helper destination names include `resident004`.

- [ ] Generate exact source/hash delta and whole-change review package from fixed git BASE; independently review spec and code quality including all consumer/staging/authority/preservation bindings. Resolve blocking findings with retained fix evidence and bounded reviews.
- [ ] Fresh file-only Vega/Mac quota/interpreter/shared inputs/checkpoint/package/parent absence checks. Record actual all-remaining capacity; if insufficient, prepare scoped safe alternatives without deleting data or weakening reserves.
- [ ] Freeze complete new proposal, index all files/metadata, preserve it on Vega and independently verify its complete archive and current source. Preserve current human instruction and readback acceptance before actual stage authority.
- [ ] Root alone creates actual first-stage authority and release after every preceding gate passes; independently preserve/readback actual controls before any parent-only setup and once-only submission. No extra human approval is required for work already explicitly authorized in human-instruction001.

### Task 5: Authorized staged evaluation and scientific closure

- [ ] Root alone verifies fresh all-input/helper/interpreter/capacity/parent/path/claim/readback gates and submits only index0 once with flagless supervisor command. Capture genuine running/terminal scheduler identity rather than pending placeholders.
- [ ] On actual failure, stop all unissued stages, preserve complete output/private/claims/logs/controls, diagnose purely and review; no physical retry is authorized by this plan.
- [ ] On complete first own32, review all raw/reset/runtime/native800/inclusive801/parity evidence, preserve full Vega data with the new independently streamed backend, and build the unchanged source-bound own32seal. Only then back/admit/submit remaining[3,6,9,12,15] once.
- [ ] Apply the same per-policy full review/backup/seal; bind exact already-reviewed science poses and both original banks before science authority. Execute the original1920starts with all5cases only for qualified policies and unchanged caps. Stop unissued work on any failure.
- [ ] Preserve all successful/failed evidence, produce the complete six-policy comparison with safety+utility and stated horizon/payout confounds, at most3plots perseed and requested bounded videos. Keep critical data on Vega; SSD archival waits for a connected verified device and separately enumerated eligible cold data.

## Implementation rulings

- The user explicitly approved this design and execution method; proceed continuously through implementation/reviews, not another plan-confirmation loop.
- The working tree has preexisting unrelated tracked/untracked changes. Use a new `codex/vega-resident-20261027` branch and explicit path staging; no reset, clean or all-files commit.
- Archive tests are pure. A generic whole-repo test command that constructs physics is outside the user's prescribed pure-regression preparation scope; run all new-package pure tests and applicable frozen pure suites instead and disclose that boundary.
- Existing frozen proposal source is a reference, not a mutable working directory. New tools are tracked small source files; operational artifacts and complete scientific evidence remain in the existing ignored campaign workspace with durable Vega preservation.

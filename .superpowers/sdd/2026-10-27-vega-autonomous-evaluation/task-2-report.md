# Task2 report — DONE_WITH_CONCERNS

Implementation is concrete and pure-tested in the assigned a370 worktree on
`codex/vega-resident-20261027`, based on accepted Task1 commit
`41d72c1fd21580ae339549fcaec50f1c6aed7633`. Implementation commit: `c88fc8e` (controller source, tests, manifest and initial
report). The later report-only commit records this exact implementation ID. Only new
`scripts/vega_evaluation_controller005/` source/tests/docs/evidence and this
explicitly requested report are staged. No preservation005/002/004/c548 source,
physical package, training/evaluation authority, canonical control or evidence
was edited. Unrelated dirty work was left alone. No SSH, real Slurm, remote
staging, real lease, policy run, target/model/Torch/MuJoCo import or subagent
was used by this implementer.

## Exact source graph and pin

Controller manifest byte SHA256:
`2ba48eb56047f4a03a549bc2306a6cd54848e481e177878ca8a778f2e5e2b773`.
The exact five-file source graph is enumerated by
`scripts/vega_evaluation_controller005/source-manifest.json`:

- `entry.py` (8197 bytes): isolated stdlib bootstrap, root-supplied delegation
  SHA/config/nonce/source admission, exact source-only inventory and verified-byte
  loading. `--check-source` executes no candidate modules. `--reconcile` is
  read-only after authority admission, including from a login node without a current Slurm allocation. `--activate` explicitly HOLDs until Task3
  wires its literal reviewed adapter; no plugin/module/command selector exists.
- `journal.py` (7541 bytes): permanent O_EXCL lease + held flock; exclusive fsynced
  JSON publication; source/control bounds; hash-chain fsync journal; torn/hash/
  duplicate intent rejection; 8MiB journal with6MiB regular/2MiB emergency space.
- `scheduler.py` (9738 bytes): closed Slurm argv, approved batch hash checks,
  bounded actual child transport, exact raw argv/return/stdout/stderr plus bytes,
  base64/hash/truncation/EOF/process metadata, materialized scontrol+sacct identity,
  unique name/index/UID binding, one concrete-ID cancel and no retry API. A short
  IssuePermit checks shutdown/expiry directly before subprocess creation.
- `coordinator.py` (15781 bytes): one writer, exact one-slot phase order,
  prepare-once/current-only preflight, separate durable actual reservation and
  descriptor/ledger references, durable intent before submission, sticky HOLD,
  concurrent bounded scheduler/adapter operations with owner live resource checks,
  pending/deadline/signal drain, preservation of terminal failure outcomes, and
  read-only restart/reconciliation.
- `controller.sbatch` (1214 bytes): closed8h/1CPU/2048MiB/no-requeue request with
  B:USR1@180, fixed approved source entry and three root-supplied byte hashes.
  Site-assigned actual CPU/memory/end time are recorded separately; no claim that
  requested1CPU equals actual2CPU. This request has not been submitted.

`entry.py` loads only verified journal/scheduler/coordinator bytes into their
literal module names, without sys.path/bytecode fallback. Its caller must pin
entry bytes first. Runtime staged directory must contain exactly those five
sources plus the manifest. The development tests/reports/README are excluded
from runtime staging. Task3 must add its explicit source(s), extend the literal
inventory, regenerate this separate manifest and then pin the full proposal.
Preservation's accepted13-Python manifest
`be0dca5fe9175b253b5c297ce3e67b72012678bee99389cea42d71e114aae47b`
was not altered.

## API, activation config and exact adapter handoff

The complete binding schemas, source-only staging rules, fixed CPU/CONTROL/
EVIDENCE roots and signatures are in the new controller README. Internal APIs:

```python
journal.acquire_new_lease(root, context) -> Lease
journal.Journal(path).append(event, *, emergency=False) -> {sequence, sha256}
coordinator.reconcile(root_or_Runtime) -> {status: READ_ONLY, can_submit: False, ...}
coordinator.run(Runtime(root, context), adapter, scheduler,
                *, stop=None, clock=time.monotonic) -> dict
scheduler.Scheduler(actual_controller_job_id, uid, approved_batch_references)
```

`run` refuses existing lease/journal before any scheduler access; no cleanup or
stale takeover exists. It authenticates genuine actual controller identity/end
and complete successful nested compute `sbatch --test-only` for both closed
phase batches before a permanent lease. It records actual capability evidence,
requested controller resources and observed actual controller resources.

Root-approved exact config schema is `z1_vega_controller_config_v1`, fields
`schema,nonce,source_manifest,preservation_manifest,controller_job_id,uid,root,
log_root,batches`. Root-approved exact delegation schema is
`z1_vega_controller_delegation_v1`, fields
`schema,config,source_manifest,nonce,permissions`; permissions are exactly
remaining-five/science-twelve/preserve-controls/publish-authority, in the README's
closed order. The delegation byte SHA is a trusted root caller input. No
self-grant, runtime arbitrary commands, generic plugin imports, unbound source
or Mac callbacks exist. The adapter may not be selected by config.

Task3 supplies this exact closed protocol (all failures/unknowns raise):

```python
prepare_once(phase, context) -> None
preflight_current(phase, index, context) -> None
reserve(phase, index, context) -> dict  # exact {descriptor, ledger, reservation}
resource_check(transaction_or_None, context) -> dict
terminal(phase, index, transaction, identity, observation, context) -> dict
phase_complete(phase, context) -> None
```

The three reserve values are inline authenticated Task1 JSON objects. The owner
exclusively publishes/fsyncs each and extends the transaction with
`descriptor_reference`, `ledger_reference`, `reservation_reference` (the last
points to actual reservation bytes). It journals reservation then SUBMIT_INTENT,
performs one sbatch call and records genuine observed physical identity later.
There is no future-job hash cycle and no fabricated numeric job identity.
Task3 authenticates all18 ledger/Task1 actual reservation/current quota and
shared source/control/derived accounting; no raw or arbitrary-filesystem credit.

Context carries root nonce, source/config SHA, actual controller job ID and UID,
state/provenance roots, monotonic deadline and stop Event. Gate operations get a
stable clamped operation deadline; terminal is limited to1200s and controller
minus120s. Non-resource operations run on a bounded worker; resource_check runs
concurrently on the owner thread, must be thread-safe/read-only/<=.1s and enforce
all Task1 frozen-role/G/quota/shared and source/config-drift gates. Prepare creates
and fsyncs exact slot provenance parents; owner creates/fsyncs its `slurm` child
and binds output/error paths before submission. Current-index preflight checks
only that slot's absent output/private/both claims, never completed slots.

Terminal processes failures/cancellations too, preserving complete partial
artifacts. It returns `qualified: true` only after actual review/physics,
independent verifier reaped exit0 + full EOF, real receipt and unchanged sealer
requirements. Owner additionally requires actual COMPLETED/exit0; no adapter
flag can override a failed child. Phase-complete gates all six own seals and
actual science inputs before the exact science12. Task3 never submits, retries,
cancels or fabricates authority/review results.

## Pure tests, RED/GREEN and exact commands

Final full focused commands, working directory a370:

```text
python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
```

Final result:30 tests pass normal in5.376s;30 pass optimized in5.304s. Complete
outputs are `SDD/GREEN-normal.txt` and `SDD/GREEN-optimized.txt` under the new
controller directory. No full-repository suite was run: the task explicitly
requires pure tests with zero target-environment/model imports, and this isolated
source graph has no dependency on the dirty training tree or preservation source.

Focused TDD/diagnostic commands used the same discovery command with:

```text
-k reservation_reference
-k controller_end
-p test_entry.py
-k bounded_journal
-k FailureEvidence
-k BoundBatch
-k reconcile_is
-k LateIssue
-k LiveResource
```

Retained RED transcripts report the actual failures rather than claiming each
intermediate was green: missing new module (`RED-initial.txt`, import error),
reservation-reference wrapper mismatch (`RED-reference.txt`, assertion), missing
controller end refresh (`RED-endtime.txt`, COMPLETE instead of HOLD), missing
source entry (`RED-entry.txt`, assertion), real regular-journal cap ignored
(`RED-journal-cap.txt`, assertion), hash-valid duplicate intent accepted plus
missing exact binary output fields (`RED-evidence.txt`, failure/error), missing
bound-batch constructor contract (`RED-batch.txt`, API error), restart scheduler
access (`RED-restart.txt`, assertion), and missing IssuePermit
(`RED-permit.txt`, API error). `characterization-live.txt` was already passing
when added and is explicitly **not** claimed as a RED run. An intermediate source
fixture failed because macOS `/var` is a symlink; the disposable fixture now uses
its real resolved path. This was a harness-path correction, not a production
Linux bypass or fake platform claim.

Tests exercise actual process/thread exclusive lease races, fsync hash chains,
torn/modified/hash-valid duplicate intent, durable publication/order, restart
without scheduler access, failed adapter/child gates blocking later slots,
unknown submit with no identity/no replay, empty accounting/no guessed cancel,
actual resource/log/quota-marker growth during a slow real child transport,
<=.25s observed resource intervals, exactly one owned cancellation and drain,
pending>900 and active<1530, low issue budget, actual USR1 before submission,
unknown cancellation with sticky HOLD, journal reserve exhaustion/drain,
inherited environment stripping, exact allowed array index overrides, approved
batch drift before transport, independent actual child late exit1/full EOF,
timeout/output cap/binary raw evidence, pipe/client disconnect survival, revoked
or expired permit without Popen, isolated source inventory/changed/extra/bytecode/
symlink refusal, and no arbitrary adapter selector.

Additional direct checks:

```text
/bin/bash -n scripts/vega_evaluation_controller005/controller.sbatch
python3 -B - <<'PY'  # source compilation, manifest byte checks and generation
... compile(all four production + two test Python sources, no bytecode) ...
PY
python3 -B - <<'PY'  # source-only temporary staging of manifest's exact five files
... python -B -I -S TEMP/entry.py --check-source TEMP/source-manifest.json
    --manifest-sha256 2ba48eb56047f4a03a549bc2306a6cd54848e481e177878ca8a778f2e5e2b773 ...
PY
```

The standalone staged-entry check returned0, stdout
`{"status": "SOURCE_VERIFIED"}`, empty stderr. Exact final child argv/return/stdout/stderr are retained in
`SDD/source-verification.txt`. No Python bytecode was generated.

## True unknowns / remaining integration gates

1. Task3's actual authority/review/physics/preservation adapter is intentionally
   absent; activation explicitly HOLDs. Pure fixtures prove orchestration, not
   scientific qualification, actual CPU release admission, native Slurm formats,
   nested compute submission capability, native controller lifetime or native
   Ceph callback worst-case timing. Root must review/pin the completed source,
   config, nonce/delegation and full proposal before every live operation.
2. Root reports Task1 native18normal/18-O PASS on actual Vega Python3.12/Linux,
   no targets; its read-only resource/quota timing was17.3–19.4ms on a login node.
   These are root-reported separate evidence, not this controller's native test
   results or compute/worst-case timing proof. <=.25s is a reviewed bounded-I/O
   observation envelope, not a guarantee against OS/Ceph stalls. Callback overrun
   or failed observation HOLDs; resource enforcement is not an atomic reservation.
3. Root-reported free20896463654 after an independently reviewed unrelated cache
   offload is only8829734 above zero-creditR20887633920 **before new staging**.
   Conservative credits/caps remain unchanged. Actual fresh all18 staging/quota
   admission is still mandatory; no byte arithmetic here grants authority.
4. Unknown sbatch/empty identity cannot authorize cancellation; the controller
   HOLDs without replay and may leave an independent child for root read-only
   reconciliation. Known owned identity permits one cancel only. Unknown cancel
   stays unresolved/HOLD. Hard kill/node failure has the same independent-child
   limitation. Same-account other writers can consume quota between observations.
5. Journal emergency space is finite and shared-cap staging must reserve it.
   Storage failure preventing durable emergency intent cannot be represented as
   a successful cancellation/drain. Task3 methods must honor stable operation
   deadlines and stop before publication, and must bound/reap their children;
   Python worker timeouts cannot safely kill arbitrary in-process adapter code.
6. The source request uses fixed `/usr/bin/python3`; root must approve/hash the
   actual native interpreter and request before submission. No such request was
   staged or launched here. Resource-site hook assignment may differ from the
   one-CPU request, so the journal preserves genuine observed resources separately.

## Fix round1/5 — T2-I1, T2-I2, T2-I3

Review base: `19130c97e489b55d89b8b5eaafea242d58989144`.
Fix implementation commit: `2ea22574362c4a586f5b1c7ca4704b60238f9b1a`.
Status: all three requested findings implemented and covered locally; independent
scoped re-review and root's corrected native rerun remain pending. No other
feature, authority, credit or policy change is included.

The **current** five-file controller manifest SHA256 is now
`b0ec09d7dbe0cf1c9c20ada199cd6166e9adb4a41d902be51d0570188e2854ed`.
Only `coordinator.py` changed in the operative source graph (now16837bytes).
Its exact source SHA is in the regenerated manifest. Entry, scheduler, journal
and controller request bytes are unchanged. The source graph remains four
Python files plus the request source; preservation's13-source package is
untouched. Adapter/config APIs, roots, exact index orders and all budget/cap
constants are unchanged. README now records the corrected lifecycle and drain
semantics. The initial pin/source sizes and30+30 host-specific results earlier
in this report describe the prior reviewed head, not this fix head.

**T2-I1:** owner now tracks UNISSUED/ACTIVE/TERMINAL explicitly. A genuine,
validated terminal observation transitions to TERMINAL while retaining the
submission and materialized identity for provenance. The1530s cutoff applies
only to ACTIVE. Terminal work has its own stable deadline, the earlier of
postprocessing1200s and controller end minus120s; live monitoring and the returned
result boundary both reject an expired postprocessing deadline. No late result
can complete the slot. The focused regression uses initial3635s lifetime,
899pending,1200policy,10preservation: preservation crosses1530s remaining, slot3
completes, and the next issue correctly HOLDs solely because it lacks3630s.
A second regression rejects an adapter result beyond its clamped postprocessing
deadline. Neither test constructs a target environment or policy.

**T2-I2:** drain now calls a separate bounded `pause_drain`. It continues live
resource checks and actual short sleeps while quiescence/stop is already set;
it never uses the already-set Event as its sleep primitive. The pause respects
the drain deadline and controller finalization boundary. Normal issue polling
retains its previous stop-sensitive pause. A regression requires four drain
observations, the first three nonterminal, and checks actual paced intervals,
one submit/one cancel, <=.25s resource gaps, valid terminal/HOLD journal closure,
and bounded retained emergency evidence. Its test interval is reduced to.12s;
production remains4s, drain180s. Before the fix its observed gaps were
0.000740/0.000652/0.000669s; the pacing assertion failed.

**T2-I3:** resource growth waits for an explicit Event set **inside the issued
slow scheduler observation**. The mutation thread is joined before assertions
or temporary-root cleanup. Quota-marker replacement is atomic, so a transient
empty text file cannot mask the intended quota-breach reason. Assertions retain
the actual HOLD reason, full submit/cancel call sequence, mutation/barrier state,
thread errors and resource timestamps on failure. A separate preissue-resource
breach requires HOLD with no submission, no cancellation and no claimed drain.
The old Timer(.12) fixture was deterministically reproduced with an otherwise
valid.25s preissue reservation delay: all three resource subtests failed their
unconditional drained assertion. The corrected post-submit barrier fixture passes
under that **same** delay, demonstrating that it no longer depends on filesystem
startup/fsync timing.

Covering files: `tests/test_coordinator.py` (lifecycle/drain/resource fixture) and
`tests/test_entry.py` (verified source inventory/entry). All commands below ran
locally in a370, using actual host/interpreter, with no SSH, real Slurm, live
job, authority mutation, target/model/Torch/MuJoCo import or subagent.

Exact normal/optimized covering commands and observed outputs:

```text
python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -k LifecycleReview -v
RED: Ran3tests; FAILED(failures=3): terminal active cutoff, late postprocessing
acceptance, unpaced drain. SDD/round1-RED-lifecycle.txt.
GREEN: Ran3tests in0.416s; OK. SDD/round1-GREEN-lifecycle.txt.

python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
Ran34tests in5.895s; OK. SDD/round1-GREEN-normal.txt.

python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
Ran34tests in5.848s; OK. SDD/round1-GREEN-optimized.txt.

python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -p test_entry.py -v
Ran3tests in0.264s; OK. SDD/round1-GREEN-entry-normal.txt.

python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -p test_entry.py -v
Ran3tests in0.258s; OK. SDD/round1-GREEN-entry-optimized.txt.
```

Exact command used unchanged before and after the I3 test correction:

```sh
python3 -B - <<'PY'
import sys,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('scripts/vega_evaluation_controller005/tests').resolve()))
import test_coordinator as tests
original=tests.FakeAdapter.reserve
def slow_reserve(self,*args):
    time.sleep(.25)
    return original(self,*args)
suite=unittest.defaultTestLoader.loadTestsFromName('LiveResourceTests.test_actual_log_and_quota_growth_during_real_child_wait_are_observed',tests)
with patch.object(tests.FakeAdapter,'reserve',slow_reserve):
    result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

Before: Ran1test/3subtest failures,0.787s; retained in
`SDD/round1-RED-growth.txt` (trailing display whitespace normalized only).
After: Ran1test in3.254s; OK; retained in
`SDD/round1-GREEN-growth-slow-preissue.txt`. Final source-only staging check
compiled the six production/test Python files without bytecode, staged the
manifest's exact five source files, then ran isolated entry `--check-source`
with the new manifest SHA. Return0, SOURCE_VERIFIED, empty stderr; exact final
argv/stdout/stderr are retained in `SDD/round1-source-verification.txt`.
`git diff --cached --check` passed before the implementation commit.

Root-provided native evidence was read, not rerun or changed by this implementer:
`evaluation-vega-autonomous-implementation005/controller-native-pure001/command-result001.json`
records the prior manifest2ba48eb head on genuine Linux `/usr/bin/python3.9`,
30tests/1failure in the quota live-growth subtest, no forbidden imports/modules
or live commands, and no optimized follow-up. The outer extraction harness also
reported its tarfile warning; it is retained as harness evidence, not described
as a clean native pass. Root will rerun the corrected focused native case first,
then final native gates. This local34+34 result does not replace those native
checks, review/physics integration or delegated activation. Activation remains
explicitly HOLD pending Task3. Full-campaign journal/control byte-budget
integration remains root's separately declared Task3/4 gate; this round adds no
new claim about that budget.

## Fix round2/5 — residual T2-I2 refreshed-deadline branch

Review base: `113ecccda7f2ff503250e5db80974a1c0f830a21`.
Fix implementation commit: `b30c4c0ca046304b4095202e5d942eebcda94dfd`.
Status: residual T2-I2 implemented and locally covered; scoped re-review pending.
T2-I1 and T2-I3 were closed by the prior scoped review and are not reopened.

The production change is one outer-drain-loop condition: before every further
snapshot, check the **current refreshed** controller-minus120s boundary as well
as the originally captured bounded drain deadline. The inner real-sleep pause
already respected this boundary; the outer loop now stops too, preventing a
shortened controller end from turning its early return into a scheduler/journal
spin. The existing final DRAIN_UNKNOWN/HOLD path records the unresolved outcome.
It does not claim terminal drain, accept a terminal receipt, issue again, or
repeat cancellation after the boundary. Ordinary paced quiescent drain is
unchanged.

The added focused regression starts quiescence from an issued RUNNING
observation, cancels once, then makes the first post-cancel snapshot shorten the
controller's remaining lifetime to100s. Its scheduler would otherwise report
three COMPLETING states then CANCELLED. Before the fix it made four immediate
post-cancel observations (RED,4!=1, drained=true). After the fix it makes only
that first post-cancel observation, records DRAIN_UNKNOWN/terminal drain deadline,
returns HOLD/drained=false, and makes neither terminal-processing acceptance nor
SLOT_COMPLETE. The test retains the actual submit/cancel sequence and observation
timestamps in diagnostics. Existing four-observation pacing and both terminal
lifetime regressions remain green.

Current exact five-file controller manifest SHA256:
`b66bfbe6500bc82b66eda98c027a83ea4b4e4f28ae7157448596777a446a89be`.
Only coordinator source changed in the operative graph (16882bytes); all other
operative source bytes, source inventory, adapter/config signatures, physics,
all budget/cap constants and preservation13-source package remain unchanged.
README documents the outer refreshed-boundary check. No journal externalization,
credit changes, SSH, live scheduler/authority operation or agent was introduced.

Exact covering commands/output (a370 local actual host/interpreter):

```text
python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -k refreshed_controller_reaches -v
RED: Ran1test in0.009s; FAILED(failures=1),4 post-cancel observations instead of1.
SDD/round2-RED-refreshed-drain.txt.

python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -k LifecycleReviewTests -v
Ran4tests in0.427s; OK. SDD/round2-GREEN-lifecycle-normal.txt.

python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -k LifecycleReviewTests -v
Ran4tests in0.420s; OK. SDD/round2-GREEN-lifecycle-optimized.txt.

python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -p test_entry.py -v
Ran3tests in0.259s; OK. SDD/round2-GREEN-entry-normal.txt.

python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -p test_entry.py -v
Ran3tests in0.262s; OK. SDD/round2-GREEN-entry-optimized.txt.
```

These are the amended lifecycle tests in `test_coordinator.py` plus source-entry
manifest tests in `test_entry.py`,7passing normal and7optimized. No full34/35
suite was repeated in this narrow round. The exact new manifest's five byte/size
records were checked and all six production/test Python files compiled directly
without bytecode; output is retained in `SDD/round2-source-verification.txt`.
`git diff --cached --check` passed before the implementation commit.

Root separately reports that the prior b0ec09d source's corrected growth and
preissue cases passed on actual Linux/Ceph. Its prior native lifecycle selector
used `LifecycleReview` instead of the actual `LifecycleReviewTests` class and
produced three harness-load errors; this is not a full native34-test pass. Root
retains that failed harness evidence in OPS005/controller-native-pure002 and will
correct the selector against this new source. This implementer performed no
native/live operation. Final native gates, independent scoped acceptance and
Task3 integration/delegated activation remain root-owned; activation stays HOLD.

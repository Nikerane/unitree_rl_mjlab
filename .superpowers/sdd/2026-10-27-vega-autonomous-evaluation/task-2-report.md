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

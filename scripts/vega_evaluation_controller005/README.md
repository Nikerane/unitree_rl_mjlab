# Durable evaluation005 controller — Task2

This directory is a separately pinned controller source closure. It does not
change `vega_autonomous_successor005`, physical supervisor565, frozen002/004,
index0/job48670474, or the exact policy resource/worker arguments. It imports no
Torch, MuJoCo, policy, model or environment. Root owns staging, deployment,
canonical authority and live operations. **Activation is disabled until Task3's
literal, reviewed phase adapter is added to this source closure.**

The runtime graph is four Python sources (`entry.py` → verified `journal.py`,
`scheduler.py`, `coordinator.py`) plus the `controller.sbatch` source request.
`source-manifest.json` binds their exact bytes. Its own SHA is passed by root;
the caller pins the entry before executing it. The staged source directory must
contain exactly these five files and the manifest. Tests, reports and this README
are development artifacts outside that runtime directory. There is no import,
bytecode, arbitrary-command, configurable-module or Mac callback fallback.
Task3 extends the explicit `entry.SOURCES` inventory, regenerates this separate
manifest, and pins the full proposal before root can activate.

## Ownership and state

Internal reviewed-code interfaces are:

```python
journal.acquire_new_lease(root: Path, context: dict) -> Lease
journal.Journal(path: Path).append(event: dict, *, emergency=False) -> dict
# Return: {sequence: int, sha256: str}; fsync before return.
coordinator.reconcile(root_or_Runtime) -> dict  # READ_ONLY, can_submit=False
coordinator.run(Runtime(root, context), adapter, scheduler,
                *, stop=None, clock=time.monotonic) -> dict
```

`owner.lease` is an O_EXCL permanent file plus a held flock. Its bounded, fsynced
identity includes nonce, actual hostname/PID, controller Slurm identity and
source/config hashes. No stale takeover, lease removal or automatic restart
exists. Any pre-existing lease/journal denies `run` **before scheduler access**.
Restart uses `reconcile` only: no repairs, scheduler mutations or replay. Torn,
hash-broken and hash-valid duplicate-intent journals fail reconciliation.

For each slot the owner durably publishes separate exclusive descriptor, ledger
and reservation JSON files, records RESERVATION, then fsyncs SUBMIT_INTENT before
one and only one `sbatch` call. `reservation_reference` points to the actual
Task1 reservation JSON, not a wrapper. A timeout/unknown submission permanently
consumes its intent; no second submission is available. A known returned array
ID is materialized through genuine `sacct` numeric allocation rows and matching
`scontrol` identity, UID, nonce-bearing unique job name and exact task index.
No array arithmetic, compressed pending placeholder or guessed cancellation ID
is accepted. Empty/ambiguous accounting causes HOLD.

All lease/journal decisions have one writer. Scheduler and bounded adapter work
run in a worker returning observations; the controller thread checks resources
every .1s, including during scheduler subprocess waits. Resource callbacks must
finish within .1s, giving a <=.25s observation envelope under the reviewed
bounded-I/O assumptions. Measured callback overruns HOLD. Scheduler calls are
bounded (2s each, bounded inventory), snapshots have a 10.5s owner limit and a
4s polling gap, therefore observation attempts are <=14.5s apart. Failure never
falls through to another issue. A two-second IssuePermit rechecks quiescence and
expiry immediately before spawning the submission subprocess, preventing a
stalled pre-spawn worker from issuing after quiescence.

A positive owned identity authorizes one journaled `scancel CONCRETE_JOB_ID`,
then <=180s terminal drain. Drain pauses retain the4s scheduler cadence while
continuing live resource checks; a latched stop Event never turns that pause into
a busy loop. Unknown cancellation stays HOLD and is never repeated.
If submission or identity remains unknown there is no safe cancellation target;
failed/partial evidence remains and root must reconcile. A hard kill/node loss
can leave an independent child alive; permanent intent/ownership still forbids
restart/replay. Terminal processing runs for failed/cancelled outcomes too;
owner independently requires actual COMPLETED/exit0 and adapter qualification.

The journal is capped at8MiB, with6MiB regular decisions and2MiB reserved for
bounded quiescence/drain/HOLD. Controls are capped at64MiB, preserving2MiB local
emergency headroom. JSON events/files are bounded to1MiB; nodes to8192; no links,
special files or control hardlinks. The adapter must include **all** shared
controls/source/staging and derived allocation in its separate shared guard,
reserve emergency headroom in the authenticated ledger, and enforce pending
writes in Task1 preservation. Local bounds do not grant quota or capacity credit.
If the filesystem cannot write even reserved emergency controls, reliable durable
cancellation cannot be claimed; evidence/unknown state must be reconciled by root.

## Exact closed phase-adapter protocol (Task3 handoff)

The adapter is a literal source module wired by Task3 after source admission.
Runtime config may not select module names, commands, callbacks or factories.
The owner is the sole scheduler submitter/canceller; the adapter must never call
a scheduler submission, retry, SSH or Mac/LLM service.

```python
prepare_once(phase: str, context: dict) -> None
preflight_current(phase: str, index: int, context: dict) -> None
reserve(phase: str, index: int, context: dict) -> dict
resource_check(transaction: dict | None, context: dict) -> dict
terminal(phase: str, index: int, transaction: dict, identity: dict,
         observation: dict, context: dict) -> dict
phase_complete(phase: str, context: dict) -> None
```

Every failure/unknown raises; `False` is also rejected. Gate methods return None;
`reserve` returns **exactly** `{descriptor, ledger, reservation}` containing
Task1's authenticated inline JSON objects. It calls actual Task1 descriptor,
reservation and validation APIs from the separately verified preservation
closure. In particular ledger is authenticated all18 state, not filesystem free
bytes or arbitrary retained objects. Use actual CPU manifests to map indices,
not ledger order. The reservation descriptor and reservation bind the requested
index and root nonce. The owner adds `descriptor_reference`, `ledger_reference`
and `reservation_reference` after exclusive fsynced publication. No job ID is
invented in advance. `terminal` receives genuine observed identity/terminal raw
scheduler evidence and those actual references. Return a bounded JSON object
with `qualified: true` only after real physical/review/preservation/seal gates;
include full receipt/seal references required by the next phase. False, missing,
exception, timeout, process partial output/late failure, or unknown blocks issue.

`context` contains nonce, controller_job_id, UID string, source_sha256,
config_sha256, exact state root, exact `log_root`, controller monotonic `deadline`
and threading `stop` Event. Non-resource methods run on a bounded worker; the
resource callback runs on the owner thread concurrently and must be thread-safe,
read-only, bounded and <=.1s. Honor deadline/stop before any new publication;
terminal preservation after quiescence remains allowed within its clamped
1200s/controller-minus120s deadline. Never allow background work after a method's
bound. Terminal verifier acceptance requires reaped exit0 plus both pipe EOFs;
Task1's returned PASS/reference alone is insufficient. No target environment is
constructed by the adapter.

`prepare_once` runs once per phase, creates/fsyncs exact parents, validates
completed anchors and authenticates phase authority. It must prepare
`EVIDENCE_ROOT/slotNN` for that phase's slots. Owner creates/fsyncs its `slurm`
subdirectory before intent; bound `%A_%a.log/.err` consume that slot's provenance.
`preflight_current` checks absence only for **that current unissued slot's**
output/private/both claims; previously completed slots remain immutable evidence.
`resource_check(None, context)` still enforces shared source/controls/derived
caps and quota. With a transaction it additionally uses actual Task1
`current_usage` + `check_reservation` with live current user quota; it counts
unverified raw as frozen-reservation consumption and never credit. It must fail
on quota, role cap, aggregate G, log, node/link or source/config drift. This is
observation-bound enforcement, not atomic Ceph reservation against other writers.

`phase_complete('remaining')` validates all six genuine own seals including
immutable index0 and prepares no extra policy. Science preparation requires
actual pose/tape/geometric controls and all authority/release preservation gates.
Remaining order is exactly `3,6,9,12,15`; science order exactly
`1,2,4,5,7,8,10,11,13,14,16,17` (1920 original starts). Index0 is never issued.
No completed slot is globally required absent, requalified by fabricated data,
or reissued.

## Activation config and delegated authority

The fixed root is:
`/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002/admission-staged-integration002/autonomous005`.

Root supplies `controller-config.json`, with exactly:

```json
{"schema":"z1_vega_controller_config_v1","nonce":"32_TO_64_LOWERCASE_HEX","source_manifest":{"path":"ABSOLUTE_CONTROLLER_MANIFEST","sha256":"SHA"},"preservation_manifest":{"path":"ABSOLUTE_TASK1_MANIFEST","sha256":"be0dca5fe9175b253b5c297ce3e67b72012678bee99389cea42d71e114aae47b"},"controller_job_id":"ACTUAL_SLURM_ID","uid":"ACTUAL_UID","root":"CONTROL_ROOT/controller-state005","log_root":"CPU_ROOT/evidence-autonomous005","batches":{"remaining":{"path":"CONTROL_ROOT/controller-inputs/remaining.sbatch","sha256":"UNCHANGED_REMAINING_SHA"},"science":{"path":"CONTROL_ROOT/controller-inputs/science.sbatch","sha256":"APPROVED_SCIENCE_SHA"}}}
```

Paths shown as CONTROL_ROOT/CPU_ROOT are literal expanded absolute paths in the
actual config, not runtime substitutions. The exact remaining source is the
unchanged `remaining-five-own32.sbatch` under the frozen staged-integration002
proposal, copied by root with identical bytes. The policy request override adds
only one-index array, closed job name and bound output/error log destinations;
physical2CPU/8192MiB/1200s, srun and worker arguments remain in the unchanged
batch. Scheduler rechecks each approved batch hash before test-only and issue.
Remove inherited SLURM_*, SBATCH_*, PYTHONPATH and PYTHONHOME before child spawn.

Root also supplies `controller-delegation.json`, with exactly:

```json
{"schema":"z1_vega_controller_delegation_v1","config":{"path":"CONTROL_ROOT/controller-config.json","sha256":"CONFIG_SHA"},"source_manifest":{"path":"ABSOLUTE_CONTROLLER_MANIFEST","sha256":"MANIFEST_SHA"},"nonce":"SAME_NONCE","permissions":["remaining-five","science-twelve","preserve-controls","publish-authority"]}
```

Its approved byte SHA is a trusted caller input; the controller never grants
itself authority. The source entry authenticates delegation/config/nonce and
honest Linux + actual SLURM_JOB_ID/UID before loading candidate modules. Task3
must wire its literal adapter and construct `Scheduler(job_id, uid,
config['batches'])` plus authenticated `Runtime`, then `run`. The current entry
holds instead. No adapter selector exists.

The closed controller request is8h/1CPU/2048MiB/no-requeue with B:USR1@180;
requested resources and genuine scontrol actual CPUs/memory/end time are recorded
separately (the site hook may assign2CPUs). The controller's local monotonic
lifetime is clamped to actual Slurm end time and refreshed while observing.
New issue needs>=3630s; pending>900s or known-active remaining<1530s
quiesces/drains. A validated terminal observation changes the lifecycle to
TERMINAL while retaining submission/identity provenance. Terminal preservation
uses its separate1200s/controller-minus120s deadline; crossing1530s after a known
terminal outcome does not itself quiesce. Late postprocessing results cannot
complete the slot.
The budgets are policy1200/post1200/drain180/final120/lag30. USR1/TERM/INT latch
stop and prohibit any later issue. Nested compute `sbatch --test-only` for both
bound batches must return complete observed exit0 before permanent lease.

## Pure verification and limits

```text
python3 -B -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
python3 -B -O -m unittest discover -s scripts/vega_evaluation_controller005/tests -v
```

Tests use actual filesystem fsync/hash chains, process/thread lease races, real
child pipes/disconnect/late exit/timeout/output caps and real file/quota-marker
growth synchronized to an issued slow observation, joined mutation threads,
preissue breaches without submission/cancellation, terminal lifetime crossing and
paced multi-observation drain. Scheduler responses and authority/phase
operations are closed pure fixtures. There is no SSH, real Slurm, root lease,
remote staging, target import or physical rollout. Source tests stage disposable
source-only copies and report the actual host/interpreter; native Linux/Ceph and
Slurm identity formats/resource timing need root's integration gate. Root reports
Task1's native18+18 tests pass; that does not activate Task2 or prove its native
scheduler integration. Fresh quota/staging and every Task3 gate remain required.

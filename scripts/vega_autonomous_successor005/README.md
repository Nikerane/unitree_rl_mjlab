# Autonomous005 preservation profile

This is the Task1 source closure, copied from the c548 predecessor and changed
only for honest Linux process verification, its source/authority contract and
one in-flight capacity transaction. It issues no scheduler job and grants no
technical/scientific qualification. Same-cluster storage is not a second
failure-domain replica. The owner, source authority and controller observation
are trusted under the existing same-account boundary.

## Source and authority

`source-manifest.json` binds all **13** production Python sources: the copied
archive ten plus `independent_verify.py`, `control_preservation.py`, and
`reservation.py`. Every source is read bounded/no-follow, compiled from its
verified bytes, and loaded only after preservation authority authentication.
Extra Python/bytecode and any symlink in this closure fail. There is no normal
import/bytecode fallback. `archive/predecessor-source-manifest.json` records the
original ten bindings and is historical, not an operative package manifest.

Stage this preservation closure as its **own source directory**. Later controller
and review source belongs in another separately pinned directory. Adding new
Python alongside this profile intentionally invalidates its closed inventory.
The direct entry itself must also be pinned by the caller before execution.

Closed roots:

- CPU_ROOT `/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002`
- BACKUP_ROOT `/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-autonomous005`
- CONTROL_ROOT `CPU_ROOT/admission-staged-integration002/autonomous005`
- EVIDENCE_ROOT `CPU_ROOT/evidence-autonomous005`

Policy preservation authenticates the new helper through the actual neutral
remaining/science CPU manifest's `preservation_contract`, before source loading;
it then calls actual unchanged565 `load_release` and `require_selected`. The
contract is exactly:

```json
{"schema":"z1_vega_resident_preservation_contract_v1","mode":"VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1","backend_package":{"path":"ABSOLUTE_SOURCE_MANIFEST","sha256":"SHA"},"capacity_contract":{"path":"ABSOLUTE_CAPACITY_CONTRACT","sha256":"SHA"},"backup_root":"BACKUP_ROOT","control_root":"CONTROL_ROOT"}
```

Only actual `CPU_ROOT/admission-staged-integration002/{remaining,science}/release.json`
is accepted. The actual first release/seal/receipt is never reissued or changed.
Slot index is checked against the selected actual CPU manifest, not ledger row
order. In particular actual index3 is extended200-seed2026102301.

## Preservation API and controller acceptance

```python
# independent_verify.py, executes inside a distinct verifier process:
preserve(request_path: str, output: str) -> dict  # exactly {"path": str, "sha256": str}
```

Execute the pinned entry as an isolated child:

```text
APPROVED_VEGA_PYTHON -B -I -S /ABSOLUTE_PROFILE/independent_verify.py --request /ABSOLUTE_REQUEST.json --output CONTROL_ROOT
```

`output` is the exact existing CONTROL_ROOT directory. `wall_seconds` is a
positive integer <=600, additionally clamped by the controller's own deadline.
The controller must use bounded owned pipes and accept the complete bounded JSON
**only after actual reaped exit0 and stdout/stderr EOF**. A PASS file alone,
partial stdout, returned reference before exit, or late exit1 never authorizes
progress. The copied `archive.transport.run_process(argv, payload, *, deadline,
stdout_cap, stderr_cap=1048576, stdin_cap=8388608, consumer=None)` provides this
boundary; its successful result records `pid`, `observed_ppid`, `returncode`,
`complete_eof`, `stdin_complete`, `stdout`, `stdout_bytes`, `stdout_sha256`,
`stderr`, `stderr_bytes`, and `value`. Use e.g. stdout_cap8192 for the verifier's
reference output; parse JSON with duplicate-key/nonfinite rejection.

The verifier starts actual approved local writer children (no SSH callback),
checks full gzip CRC, tar/PAX/member/restore/source-identity metadata and EOF,
compares primary and independent results, checks application JSON and source
freshness, then publishes closed stream/fresh/provenance/full-receipt stages.
Provenance records actual Linux hostname/PID/PPID, resolved interpreter hash,
entrypoint hash, source-manifest reference, observed writer PID and writer exit0.
The creator PID differs from the verifier and its PPID equals the verifier PID.
The verifier's *own* successful exit is observed externally by the controller.

Policy requests retain the predecessor's request schema
`z1_horizon_vega_resident_archive_request_v1` and all bound scheduler identity,
layout, terminal observations, shared source proofs, application, size/deadline
and actual CPU manifest/release fields. Required005 changes are backend mode
`VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1`, `control_root` replacing
`mac_control_root`, all local/remote bound paths on Vega, and a full
`reservation_reference` to the preissue reservation. `remote_archive_root` is
BACKUP_ROOT. `remote_archive_destination` is exactly
`BACKUP_ROOT/POLICY/RUN_LEAF/job-CONCRETE_JOB_ID`. Preserve inherited `ssh_host:
"vega"` as a host identity label; this transport performs no SSH.

The capacity contract is exactly schema
`z1_vega_resident_capacity_contract_v1`, rule_version `capacity-ledger-ruling002`,
`slots=archive.capacity.layout_rows()`, `slot_caps=CAPS`, `shared_caps=SHARED`,
`immutable_reserve_bytes=11844714496`, `control_root=CONTROL_ROOT`,
`shared_controls_bytes=67108864`, `working_metadata_bytes=92274688`.

Policy output is the full normalized unchanged-consumer receipt schema
`z1_horizon_terminal_multiroot_archive_receipt_v1`; compact receipts remain
separate proof controls. Both raw/private regular maps and complete all-node
metadata remain present. Actual unchanged565 consumer and86c98160 sealer are
exercised with the real32x881 NPZ synthetic factory in focused tests.

## Acyclic generic controls

Create a root-authorized canonical `CONTROL_ROOT/preservation-authority.json`:

```json
{"schema":"z1_vega_control_preservation_authority_v1","backend_mode":"VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1","helper_package":{"path":"ABSOLUTE_SOURCE_MANIFEST","sha256":"SHA"},"backup_root":"BACKUP_ROOT","control_root":"CONTROL_ROOT"}
```

This is an explicit trusted root control; the profile cannot grant itself
source authority. A `CONTROL_ONLY` request includes the common schema/mode,
`helper_package_reference` and equal `remote_helper_package_reference`,
`preservation_authority` (full reference above), `control_root`, `control_id`,
`capacity_ledger_reference`, finite caps and `wall_seconds`.
`control_id` matches `[a-z][a-z0-9-]{0,63}`. Source roots are exactly
`CONTROL_ROOT/input-bundles/ID/{controls,private}`; the private directory is
required even if empty. Destination is `BACKUP_ROOT/control-only/ID`.
`max_source_bytes` and `max_archive_bytes` are <=67108864, metadata<=4194304,
evidence<=8388608, members<=8192. Set both future reserve fields to11274289152;
the guard raises the protected reserve from the conservative ledger as needed.
No policy/scope/bank/job or CPU release/manifest fields are allowed. The result
schema is `z1_vega_control_archive_receipt_v1`, kind `CONTROL_ONLY`, with complete
node metadata and separate actual writer/verifier provenance. It cannot qualify
a policy. Creation, independent full stream, source freshness and exact closed
publication all run without a CPU release. Generic archives plus CONTROL_ROOT
files share the64MiB control cap; no arbitrary filesystem free-space fallback.

## Reservation API (Task2 handoff)

Import these functions only through a verified source closure:

```python
preissue_descriptor(*, index: int, release: dict, nonce: str) -> dict
zero_ledger(*, preissue: dict, layout_sha256: str, staging_bytes: int = 0) -> dict
make_reservation(ledger: dict, *, index: int, release: dict, nonce: str,
                 ledger_sha256: str, active: list, credit_request=None,
                 deadline=None) -> dict
validate_reservation(reservation: dict, ledger: dict, *,
                     credit_request=None, deadline=None) -> dict
check_reservation(reservation: dict, usage: dict, *, free_bytes: int,
                  pending: dict | None = None) -> dict
current_usage(reservation: dict, *, deadline: float, max_nodes: int = 8192) -> dict
```

`release` is exact `{path,sha256}` for the actual phase. Nonce is32–64 lowercase
hex characters, supplied by the root-approved controller. Index0 is forbidden.
The preissue descriptor is schema `z1_vega_preissue_descriptor_v1`, kind `SLOT`,
with exact `index`, actual closed `slot` mapping, `release`, `nonce`, `caps`, and
`shared_caps`. It excludes job ID, reservation reference and future scheduler
identity. The controller must bind this descriptor into its durable preissue
record before submitting. Genuine physical job ID is recorded later; never
invent it.

The **new005** ledger schema is `z1_vega_autonomous_capacity_ledger_v1` with
`rule_version`, inline `preissue_descriptor`, canonical
`preissue_intent_sha256=digest(preissue_descriptor)`, actual `release_sha256`,
`layout_sha256`, all18 `slots`, `shared`, and nonnegative `staging_bytes`.
This is deliberately separate from the predecessor's post-terminal
`request_identity_sha256`: runtime archive provenance binds both the preissue
identity/descriptor and the actual archive-request identity. No future-job hash
cycle exists. For generic bootstrap use descriptor exactly
`{"schema":"z1_vega_preissue_descriptor_v1","kind":"CONTROL_ONLY"}` and
`zero_ledger`; its release SHA is null, and it never resolves a future release.

Slot rows retain exact policy/scope/bank/run_root/private_root, fixed caps and
lists of concrete `retained` objects per role. Ledger rows use predecessor
`layout_rows()` order; actual CPU indices use the explicit original CPU order
and are checked against actual565 on archive admission. Each retained object has
exact `path,bytes,sha256,device,inode,link_count,proof`; proof is full
`{path,sha256}`. Nonzero policy credit is admitted only after grouped actual005
full closure authentication and current file identity/SHA checks. Shared credits
and historical firstV1 credits are **disabled in this Task1 profile**; both stay
zero. A separate reviewed honest import/shared-credit extension is needed when
fresh quota cannot cover zero-credit R. Never reduce caps or release unused cap.

Reservation schema `z1_vega_slot_reservation_v1` contains exact `index`, `slot`,
`release`, `ledger_sha256=digest(ledger)` (canonical state SHA), full frozen
`ledger`, `R`, `G`, `caps`, `shared_caps`, `roots`, `nonce`. Write it durably and
exclusively before sbatch and pass its actual byte-hash reference to preservation.
`active` must be empty; never overlap transactions or reclaim a stale reservation.
This API is pure validation, **not** a durable lease or lock service; the sole
controller owns durable serialization.

`roots.raw` is `[run_root,private_root]`, `roots.archive` is
`BACKUP_ROOT/POLICY/RUN_LEAF`, `roots.provenance` is `EVIDENCE_ROOT/slotNN`, and
`roots.controls` is CONTROL_ROOT. Only regular `terminal-policy-full.tar.gz`
files under the exact archive slot root consume archive bytes; other files there
and all slot scheduler/review/log files consume provenance. `current_usage`
counts both actual raw/private roots, rejects links/special nodes/inode overlap,
bounds node count and deadline, and ignores unrelated paths entirely. Shared
lease/journal/source/control and derived allocations must be enforced separately
by the controller; they never consume or earn this transaction's slot allowance.

`usage` and `pending` have exactly integer nonnegative `raw,archive,provenance`.
Caps are167772160/201326592/92274688; G461373440. R is authenticated full all18
remaining byte arithmetic (zero credit20887633920 plus uncovered staging).
`check_reservation` returns `consumed_bytes`, `protected_bytes`,
`required_free_bytes`, `remaining_growth_bytes` and fails unless current quota
protects `max(11844714496,R-G)` and the remaining frozen reservation. At free==R,
raw growth then archive growth consumes **one** G, while raw earns zero credit.
No deletion, whole-slot cap release, arbitrary-path credit or filesystem-wide
free-space substitution is supported. Checks at observation/write boundaries
are not atomic Ceph reservations against other same-account writers.

## Verification and limits

Run only the focused pure suite:

```text
python3 -B -m unittest discover -s scripts/vega_autonomous_successor005/tests -v
python3 -B -O -m unittest discover -s scripts/vega_autonomous_successor005/tests -v
```

Disposable source copies translate only roots, native platform acceptance and
quota/scheduler fixture inputs. Children still report real platform/host/PID;
no production fake-Darwin path exists. Tests construct no Torch, MuJoCo,
environment or policy. No SSH/Slurm/live evaluation was run by Task1. Linux
native/Ceph staging and controller deadline/lease/scheduler orchestration remain
root-owned integration checks. Failed/partial artifacts remain; retries are
not implemented. The full controller must HOLD on any ambiguity or failed gate.

# Task1 report — DONE_WITH_CONCERNS

Implementation commits:

- `c1c499093027fa13c4caf9f9c343a2faab249561` — Linux profile, frozen reservation API, generic controls, focused fixtures/docs.
- `0fb021f3f1fe9ffe65c9ffe3a7f5ffeaab52e953` — honest writer `-c` invocation/source entry metadata and regression.

Baseline was `3d2acd7b4b795e29398a15c61f35bb9f4b5b80d0`. All committed paths are
under `scripts/vega_autonomous_successor005/`. The253 unrelated dirty status
entries remain. No existing c548/frozen002/004/first evidence or physical package
was edited. Root retains independent review and every live operation.

## Source closure and explicit delta

Active13-Python source-manifest SHA256:
`0668288beef977189395f492a2990877b220ab008c19e496e068eb9098a1b0bc`.
`source-manifest.json` and `SDD/source-delta.json` enumerate exact bytes/SHA256 for
all13 sources and each predecessor delta. Original c548 ten-source manifest
SHA256 is `c54885eede1ee612e54818fdbf87abe5c178a788605e28680cb367289da0368b`;
a fresh read found zero original-package byte/size mismatches.

Unchanged copied sources are archive `__init__.py`, `stream_reader.py`,
`compat/archive_contract.py`, and `compat/remote_policy_archive.py`. Copied
`resident_archive.py` changes mode/root/source closure, actual local writer
transport, process provenance, source/phase admission, reservation guard and
exact publication. `resident_store.py` changes backup root and adds explicitly
CONTROL_ONLY metadata paths while retaining full streaming/PAX/source gates.
`capacity.py` changes the005 ledger identity and actual Vega control quota guard;
`transport.py` records observed PID/PPID/EOF/exit and retains bounded nonblocking
pipes. Copied `compat/policy_archive.py` changes only its actual platform check
from Darwin to Linux; its copied compat loader hash follows that explicit seam.
The actual supervisor565 and sealer86c98160 are unchanged.

New deep modules are `reservation.py` (pure budget/bindings plus bounded usage),
`control_preservation.py` (acyclic generic control pipeline), and
`independent_verify.py` (trusted stdlib authority-before-source CLI bootstrap).
No unbound Python, bytecode or symlinks are admitted under this source root.
The new controller/review modules must be staged outside this preservation
source directory; see README source-isolation instructions.

## Exact Task2/3 interface handoff

Authoritative full schemas, roots, request fields and examples are in README.
The following public signatures are final for this Task1 version:

```python
# independent_verify.py; execute in an actual separate verifier process
preserve(request_path: str, output: str) -> dict  # exact {path, sha256} full receipt

# reservation.py, imported from a verified source closure
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

Ledger schema is explicitly **`z1_vega_autonomous_capacity_ledger_v1`** rather than
repurposing the old request identity. Fields are `rule_version`, inline
`preissue_descriptor`, `preissue_intent_sha256=digest(preissue_descriptor)`,
actual `release_sha256`, `layout_sha256`, all18 `slots`, `shared`, and
`staging_bytes`. Preissue descriptor schema is `z1_vega_preissue_descriptor_v1`,
kind `SLOT`, with exact index/slot/release/nonce/caps/shared_caps. It contains no
job ID or reservation reference. Runtime provenance records both preissue
identity/descriptor and actual archive-request identity. For acyclic generic
bootstrap use the exact descriptor `{schema: z1_vega_preissue_descriptor_v1,
kind: CONTROL_ONLY}`; it has null release SHA and never dereferences a release.

Reservation schema is `z1_vega_slot_reservation_v1`, fields exactly index, slot,
release, ledger_sha256 (canonical state SHA), full frozen ledger, R, G, caps,
shared_caps, roots, nonce. Persist exclusively before issue. `active` must be
empty; the caller owns durable locking/serialization and must never reclaim a
stale transaction. Physical job ID is bound later through genuine scheduler
identity. Actual CPU index3 is extended200-seed2026102301. Archive admission
checks selected index against actual unchanged565 manifest processing.

Usage/pending have exact raw/archive/provenance integer roles; returned budget
fields are consumed_bytes/protected_bytes/required_free_bytes/
remaining_growth_bytes. R remains all18 arithmetic; G461373440 includes actual
raw/private plus archive and provenance. Existing unverified raw is consumption,
never global credit. Role caps167772160/201326592/92274688 and immutable reserve
11844714496 remain exact. Shared640MiB derived/64MiB controls stay separate.
Only the exact slot's frozen raw/private/archive/provenance roots count.

Run the pinned verifier entry with approved Vega Python `-B -I -S`, bounded pipes
and controller-clamped deadline. Output is exactly a full `{path,sha256}` receipt
reference. Admit it only after observed reaped verifier exit0 and complete bounded
JSON/stdout/stderr EOF. The copied transport's successful return records actual
pid, observed_ppid, returncode, complete_eof, stdin_complete, stdout bytes/hash
and stderr bytes. Actual writer process uses `-c` and the pinned resident source
entry; verifier uses the pinned independent_verify source entry. Both record
actual platform/host/PID/PPID/interpreter/source-manifest hashes. The writer PID
is observed from Popen and differs from its verifier parent. No PASS file or
claimed exit can substitute for controller observation.

Generic source authority schema is `z1_vega_control_preservation_authority_v1`,
root-owned at exact CONTROL_ROOT/preservation-authority.json. It binds mode,
source manifest, backup and control roots before any candidate source executes.
It does not authorize policy execution. Generic source roots are exact
CONTROL_ROOT/input-bundles/ID/{controls,private}; destination is
BACKUP_ROOT/control-only/ID. Full generic receipt schema is
`z1_vega_control_archive_receipt_v1`, explicitly CONTROL_ONLY. The generic route
runs all-node/PAX/CRC/EOF/restore metadata, independent-process/source freshness
and exact closed publication without a CPU release.

## Commands and observed results

Working directory for every command was the explicitly assigned a370 worktree.

```text
python3 -B -m unittest discover -s scripts/vega_autonomous_successor005/tests -v
python3 -B -O -m unittest discover -s scripts/vega_autonomous_successor005/tests -v
```

Final results: **16 tests pass normal (6.530s), 16 pass optimized (6.439s)**.
Complete outputs are `SDD/GREEN-normal.txt` and `SDD/GREEN-optimized.txt`.
All owned Python was also compiled directly from source, with no bytecode files
created. `git diff --cached --check` passed before each implementation commit.

Initial expected RED evidence in `SDD/RED.txt`:4 failures for missing new
reservation and verifier. `RED-index.txt` exposes the actual index3 ordering
mistake before its fix. `RED-preissue.txt` records missing explicit005 preissue
API before implementation. `RED-writer-entry.txt` records missing actual writer
invocation metadata before its correction. RED-control/process/consumer
transcripts retain exploratory intermediate failures too: fixture compat hash
translation, wrong harness authority roots and missing fixture remote release
were harness errors, not claimed production-gate REDs. Their corrected focused
runs and final suite pass. Commands for individual retained intermediate runs:

```text
python3 -B -m unittest discover -s scripts/vega_autonomous_successor005/tests -p test_preservation.py -v
python3 -B -m unittest discover -s scripts/vega_autonomous_successor005/tests -p test_process_profile.py -v
python3 -B -m unittest discover -s scripts/vega_autonomous_successor005/tests -p test_consumer_compatibility.py -v
```

The suite exercises actual separate child IO; partial JSON/late exit1; real
writer/verifier PID/source metadata and collisions; foreign authority; rebound
source/extra source/bytecode/symlink rejection; Mac operative references; bad CRC,
truncation/trailing bytes; missing private node/restore metadata; source drift
and removal before publication; exact free==R/raw-then-archive consumption;
role caps/negative/bool usage; overlap/rebound reservation/cross-slot/arbitrary
credit; bounded root usage/log accounting and link/node limits.

The32x881 synthetic NPZ fixture runs actual unchanged sealer
`86c98160bd9bda67ca54d4600a73c1280ee38917304e200ee0b348e6cf1c29e0`
and actual unchanged consumer
`565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd`.
It verifies31contact801 worlds plus1 genuine no-contact, including unsafe complete
measurements remaining technically qualifying. An additional actual565 phase
fixture calls load_release/require_selected and rejects reservation index and
interpreter drift. Test-only disposable source/root/native-platform/quota/job
inventory translations are explicit; children report their actual host platform
and process metadata. No fake Darwin, Torch/MuJoCo/model/environment construction,
SSH, scheduler submission or real authority mutation occurred.

## Material concerns and integration limits

1. **Quota admission remains blocked at the root's latest reported free quota.**
   FirstV1 and shared materializations conservatively receive zero credit in this
   Task1 profile. Nonzero new005 policy credits retain grouped full closure/current
   object authentication; shared and historical-first credits are disabled. Root
   reports free20865741317 below zero-creditR20887633920 by21892603 bytes before
   uncovered staging. A separately reviewed honest first-anchor/shared-credit
   import extension is needed; this code must HOLD rather than reduce caps,
   delete evidence or fabricate credit. No bridge/import claim is made here.
2. Native Linux/Ceph execution has not been performed by this implementer.
   Generic CLI and policy byte/application/publication compatibility are proven
   in pure source-translated fixtures; actual remaining CPU release+full CLI+
   Ceph integration still belongs to root after independent review. New005
   nonzero credit closures are retained but not proven in a complete live chain.
3. Shared source/control/journal/staging and derived enforcement, actual current
   slot absence, durable reservation publication/lease, scheduler identity,
   cancellation/drain/deadlines and science/controller orchestration are Task2/3
   work, not supplied by this pure reservation service. No competing owner exists.
4. Storage remains same-cluster/same-account, not independent failure-domain
   replication or an atomic Ceph quota reservation. Other same-account writes
   can force HOLD between observation boundaries. Failed/partial evidence stays.

Root should dispatch independent review against the implementation commits and
this exact source manifest before native staging or operational delegation.

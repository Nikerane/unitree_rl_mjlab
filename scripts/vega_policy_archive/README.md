# Vega resident policy preservation

This external helper preserves both complete terminal policy/private trees on
Vega. The Mac verifies the independently streamed gzip without storing a tar,
NPZ, extracted tree, or full inventory. It then publishes and reads back the full
consumer receipt on Vega, finally retaining a compact Mac record. This package
never allocates jobs, executes policies, imports the target runtime, or grants
technical/scientific qualification.

The entrypoint is `resident_archive.run(request_path, output) -> dict`, or:

```sh
python -B -I -S /absolute/helper/resident_archive.py --request /absolute/request.json --output /absolute/mac-controls
```

Production accepts only the approved lexical CPU interpreter
`/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python` on SSH host `vega`.
The interpreter alias is allowed; its resolved binary path/SHA must equal the
actual selected supervisor intent. The actual physical supervisor source is
compiled by its fixed SHA and its unchanged `load_release` / `require_selected`
gates run. The three release paths are exactly
`admission-staged-integration002/{first,remaining,science}/release.json`, selected
by the actual authority's execution phase. No physical interpreter or package is
changed by this helper.

## Request and preservation contract

Use schema `z1_horizon_vega_resident_archive_request_v1`, retaining every original
CPU archive request field. The copied legacy-schema view is passed through the
unchanged CPU request validator. Additional required public fields are:

- `backend_mode`: `VEGA_RESIDENT_FULL_ARCHIVE_MAC_STREAM_V1`.
- `remote_archive_root`: the fixed `resident_store.BACKUP_ROOT`;
  `remote_archive_destination`: its exact policy/run-leaf/job-derived directory.
- `mac_control_root`: an explicit existing absolute directory; `output` must equal it.
- `helper_package_reference`, `remote_helper_package_reference`: exact
  `{path, sha256}` references to local and Vega package manifests with the same SHA.
- `capacity_contract_reference`, `remote_capacity_contract_reference`: equivalent
  exact references to the same fixed capacity contract bytes.
- `capacity_ledger_reference`: the actual current root ledger `{path, sha256}`.
- `remote_supervisor_release`: actual Vega release `{path, sha256}`, matching
  the already bound local `cpu_supervisor_release` SHA.
- `operation_growth_bytes`: total archive plus all controls/publication growth,
  positive and no larger than the selected residual archive/provenance allocations.

The actual bound CPU manifest must contain exactly this `preservation_contract`:

```json
{
  "schema": "z1_vega_resident_preservation_contract_v1",
  "mode": "VEGA_RESIDENT_FULL_ARCHIVE_MAC_STREAM_V1",
  "backend_package": {"path": "<Vega package-manifest.json>", "sha256": "<SHA>"},
  "capacity_contract": {"path": "<Vega capacity-contract.json>", "sha256": "<SHA>"},
  "backup_root": "<fixed resident_store.BACKUP_ROOT>",
  "mac_control_root": "<explicit Mac directory>"
}
```

Source package schema is `z1_vega_policy_archive_package_v1`, entrypoint
`resident_archive.py`, and `files={relative.py:{sha256,bytes}}`. It covers all ten
production Python sources, including compat sources and the entrypoint. The
manifest excludes itself, README and tests. Missing/extra/unbound Python,
bytecode and symlinks reject before effects. Every source is read and compiled
before helper modules execute. Only source bytes are loaded, including under
`-B -I -S`; no unchecked pycache is consumed. Remote staging contains only the
manifest and these production sources.

Before candidate source loading, the reviewed stdlib bootstrap reads the bound
release and CPU manifest and compares its actual `preservation_contract` package
binding. Local copies must close the already root-bound release/manifest hashes;
the remote bootstrap reads the actual Vega release and manifest. A candidate
cannot authorize itself by rebinding its own source/hash manifest. Trust starts
with the reviewed bootstrap and root-supplied release/request authority, rather
than candidate code or its internally consistent hashes.

## Capacity ledger

The exact contract is the object checked by `capacity_contract`: schema
`z1_vega_resident_capacity_contract_v1`, rule version `capacity-ledger-ruling002`,
all `capacity.layout_rows()` under the fixed original CPU root, `slot_caps` equal
`{raw:167772160, archive:201326592, provenance:92274688}`, `shared_caps` equal
`{derived:671088640, controls:67108864}`, immutable reserve `11844714496`, Mac
envelope `1400897536`, retained limit `67108864` and working limit `92274688`.

Ledger schema `z1_vega_resident_capacity_ledger_v1` contains:

- `rule_version`, `request_identity_sha256`, `release_sha256`, `layout_sha256`.
- All 18 `slots`, each with the five exact layout row fields plus identical
  `caps` and `retained={raw:[],archive:[],provenance:[]}`.
- `shared={derived:{cap,retained:[]},controls:{cap,retained:[]}}` and explicit
  nonnegative integer `staging_bytes` for additional allocations.

The request identity is SHA256 of sorted compact UTF-8 JSON of the public
request, excluding `capacity_ledger_reference` and all internal `_` fields. It is
computed before legacy validator mutation; this avoids a request/ledger hash
cycle. The release and layout hashes remain independently bound. Never serialize
internal fields into a root request.

A retained file record is exactly `{path,bytes,sha256,device,inode,link_count,
proof:{path,sha256}}`. `proof` points to an actual compact Mac verification
record. Every nonzero credit is authenticated against its full Vega
receipt/provenance, original request/release, complete before/after/origin/index/
primary controls, successful create/stream exit bindings, current separate
archive and current file descriptor identity/hash. Unsupported, stale, missing,
or changed proofs reject. During one RPC, items are grouped by compact-proof SHA:
the full proof, original authority and compressed archive are authenticated once
per group, then its maps are released. Every item still receives a current
identity/hash check; an archive item rechecks the full descriptor against the
just-hashed archive binding. This avoids rehashing an archive for each raw file
and avoids retaining all eighteen full inventories in memory. Raw hardlinks are credited once globally by
(device,inode), preserving every path record; sharing an inode across roles
rejects. New archives require `link_count=1`. Only concrete retained pipeline
controls can earn provenance/shared credit; arbitrary declared occupied bytes
cannot earn credit. The default all-zero ledger is valid and conservative.

All failed, pending, missing and unissued slots retain their allocations. Pure
`remaining_budget(ledger)` validates structure/math; it does not authenticate IO.
Production authenticates proof closure before using its result. Initial required
headroom is `20887633920 + staging_bytes`. Completion releases only supported
materialized bytes, never a whole unused slot cap.

For an operation, R is the remaining requirement and G is the declared total
growth. Every write protects `max(11844714496, R-G)` and cumulative owned
archive/control/publication growth remains at most G. The store adds the
570425344-byte floor itself, so the supplied future reserve is
`max(11274289152, protected-570425344)`, preserving a higher selected manifest
reserve. Actual Ceph quota headroom is refreshed throughout; filesystem-wide free
space cannot replace it. Partials remain on Vega and earn no credit.

Mac operations hold the explicit control-root directory lock across all RPCs,
so concurrent coordinators cannot reuse an allocation. Retained controls use that
exclusive directory lock, no-follow descriptor
walks, an 8192-node bound, exclusive writes, per-record limit 1 MiB and aggregate
limit 64 MiB across the entire explicit root, counting failures and partials.
Actual free space must retain the full 1,400,897,536-byte admitted envelope less
already retained bytes. Verification metadata stays in bounded memory; no local
working files or raw archive payloads are written.

## Publication and process guarantees

Order: local validation/package/capacity check → exclusive remote create → read
all individually bound controls → full independent nonseek Mac stream and EOF →
exact primary/create SHA/size/count agreement → actual observed application
records → bound server stream record → fresh remote source comparison/record →
full provenance and consumer receipt
publication/readback → local compact final record. All request/bound inputs and
helper sources are rechecked before and after publication. A failure withholds
the usable local PASS reference; remote originals, partials and controls remain.
No automatic retry or overwrite exists.

The stream RPC writes exclusive `stream-stage.json` only after completing and
flushing the exact bound archive transmission. Its status is
`STREAM_TRANSMITTED`, not an independent-verification verdict. The inspect RPC
requires that exact stream reference and writes `fresh-source-stage.json` with
its complete stable inventory. Both records bind the same request/create/index,
policy/job/roots, helper, release and ledger. They remain within the existing
88 MiB provenance allocation: the conservative closed control-file count is
**eleven**, including failure evidence, and every per-file cap is no larger than
the residual provenance allocation divided by eleven.

The publish RPC requires both completed records, all individually bound archive
controls/application records, an unchanged current source, and exact provenance
schema/field equality. It then compares the entire normalized full receipt with
those proof objects. Missing stages, rebound foreign fields, arbitrary PASS JSON
or incomplete/full-map substitutions reject before a publication write. The full
provenance and receipt expose `protocol_stages={stream:<ref>,fresh:<ref>}`; these
records also participate in later retained-byte credit authentication.

The independent Mac verdict is an assertion from the admitted coordinator. Server
transmission/source records establish sequence and exact proof closure; they
cannot establish a hostile same-account client's actual Mac computation. This
protocol does not provide cryptographic client attestation or protect against
arbitrary same-account shell replacement of the trusted bootstrap, root authority
or filesystem evidence.

The full Vega receipt preserves
`z1_horizon_terminal_multiroot_archive_receipt_v1`, complete unprefixed raw/private
regular member maps and canonical tree digests, every node/restore metadata
record, scheduler and shared-source evidence, and explicit remote archive SHA/
size plus independent Mac EOF readback. It is at most 4 MiB. Its seal reference
is **exactly `{path,sha256}`**; a compact Mac receipt cannot replace it.

Request, authority and proof data travel only over bounded stdin, with the
unchanged **8 MiB** packet limit. The trusted stdlib bootstrap reads and strictly
parses that packet once (duplicate keys and nonfinite values reject), authenticates
its actual remote authority before candidate loading, and passes the same parsed
packet to the verified remote entrypoint. Only the finite wall budget enters the
constant-size command argument; large admitted requests do not enter SSH/Python
argv or trigger platform argument-length limits.

Owned SSH processes have bounded nonblocking stdin/stdout/stderr, concurrent
stderr drainage and request backpressure, EOF plus exit-zero enforcement,
timeout/cancellation group kill and leader reap. The overall Mac wall is at most
600 seconds. Both remote source loading and remote IO have independent outer
alarms; deadlines use Vega's own monotonic clock, clamped to its remaining
bootstrap budget. No Mac monotonic timestamp crosses hosts. Stream stdout is
only gzip bytes; an error exits nonzero with bounded stderr.

## Pure verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts/vega_policy_archive/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -B -O -m unittest discover -s scripts/vega_policy_archive/tests -v
```

Tests use owned synthetic directories and Python child pipes, never real SSH,
Slurm, environments or payload execution. The closed path/interpreter constants
are explicitly translated in remote fixtures; actual source/file/scheduler
serialization, store, stream reader, application validator and unmodified
supervisor authority gates run. Separate full receipts pass the unchanged
first-seal/remaining gate and real 32×881 NPZ audit/sealer. The latter fixtures are
complementary, not an assertion that synthetic consumer evidence is a physical
rollout. Frozen retained regressions preserve their existing documented legacy
full-training positive exclusion. No pure result grants live admission.

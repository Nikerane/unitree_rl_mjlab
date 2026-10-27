# Vega autonomous evaluation implementation plan

> Use superpowers:subagent-driven-development task by task. The user explicitly
> requests implementation, agent reviews and remaining-five execution; no repeat
> permission question for this same scope. Preserve all prior artifacts.

**Goal:** Run the remaining full six-policy campaign autonomously on Vega while
the Mac is closed, with honest independent verification and one submission owner.

**Architecture:** One Slurm controller, durable no-replay journal/lease, separate
pinned verifier processes, unchanged physical supervisor/worker/sealer, remotely
complete evidence and closed phase admission.

**Tech Stack:** Qualified Vega Python3.12.13, stdlib, Slurm, Ceph; existing data
readers only, no new policy/runtime/physics dependencies.

**Spec:** .superpowers/sdd/2026-10-27-vega-autonomous-evaluation/DESIGN.md

## Global Constraints

- Only a370; preserve unrelated253dirty entries and frozen002/004/c548 evidence.
- Physical10/supervisor565bb430 unchanged. Index0job48670474 is QUALIFIED, neverreplay.
- Remaining own indices3,6,9,12,15, then exactly12science slots/1920originalstarts.
- One reviewed delegated writer; appautomationPAUSED; no retry or stale takeover.
- Honest Linux verifier, no /Users operative dependency or fakeDarwin.
- All18 caps167772160/201326592/92274688; shared671088640/67108864;
  immutable11844714496; zero-credit20887633920; no whole-slot caprelease.
- Per-slot reservation461373440 beforeissue accounts physicalraw+archive+controls;
  unverified raw is only reservationconsumption, never globalcredit.
- Preserve complete failed/partial evidence; any failed/ambiguous gate HOLDs futureissue.

## Review Focus

1. Laptop/SSH loss leaves controller and verifier running; actual child exit required.
2. Crash/lost return after SUBMIT_INTENT never leads to a second scheduler call.
3. Foreign/rebound sources or incomplete/full receipt flags cannot bypass provenance.
4. No Mac reference, missing first/own seal or pose artifact can reach next issue.
5. At free==R, reserved raw then archive growth remains bounded without quota laundering.

### Task 1: Linux preservation, reservation and independent verifier

Create scripts/vega_autonomous_successor005/{archive/,independent_verify.py,
control_preservation.py,tests/test_preservation.py,README.md,source manifest}.
Archive copy source is exact c548 package. Explicit delta: mode/host/root/transport,
Linux validation seam+pin, observed writer/verifier identities, in-flight reservation
budget, honest receipts and closed metadata publication. Retain all existing byte,
application, stage, source-stability and EOF gates; no old-source mutations.

**Interfaces:** preserve(request_path: str, output: str) -> full receipt ref, executed
as a distinct process and accepted only after actual exit0 by controller. Verify
generic small controls independently of CPU release. capacity reservation validates
slot/release/ledger R, G461373440 and raw/archive/provenance cap/consumption.

- [ ] RED rebound source/foreign authority/partial verifier/exit1/PID collision,
  bad CRC/trail/missing private metadata/source drift and Mac operative ref.
- [ ] RED exact free==R transaction: raw growth then archive growth; cross-slot
  consumption/credit/overlap and arbitrary-path credit rejected.
- [ ] Implement Linux profile and separate real child IO verification with honest
  process metadata; full normalized receipt passes actual unchanged sealer fixture.
- [ ] GREEN normal/-O pure tests; preserve RED/commands/output, commit ownpaths, review.

### Task 2: Owner lease, journal and durable scheduler coordinator

Create journal.py, coordinator.py, scheduler.py, tests/test_coordinator.py.
One Slurm8h/1CPU/2048MiB controller; capability check sbatch --test-only before
activation. Policy batch argv override selects one allowed index, no physicalchange.

**Interfaces:** journal.append(event)->{sequence,sha256}, acquire_new_lease(config)
refuses any existing lease; reconcile(config) read-only; run(config) dispatches
one slot through provided phase/execution adapter. Scheduler returns genuine raw
argv/return/stdout/stderr and materialized identity; no retry method.

- [ ] RED competing/stale owners, torn/hashbroken journal, duplicate intent,
  timeout/unknown/empty accounting and restart all deny resubmission.
- [ ] RED client disconnect survival, one failed fake slot blocks all later slots,
  inherited controller Slurm-step env removed and child policy index exact.
- [ ] RED resource/log/quota growth breaches cause exactly one owned cancellation,
  terminal drain and HOLD; completed slots are never required absent or reissued.
- [ ] RED <3630s refuses issue; >900s pending and <1530s active time trigger drain;
  USR1/TERM no late issue, unknown cancel remains HOLD without replay.
- [ ] Implement bounded/fsynced controls/logging, atomic lease and one-slot loop.
- [ ] GREEN pure normal/-O; source-only entrypoint, explicitpathcommit and review.

### Task 3: Remote closure, deterministic audits, seals and science gates

Create reviews.py, phases.py, tests/test_phase_closure.py, remote mirror/schema
templates; wire the Task2 adapter. Retain exact actual first seal/receipt.

**Interfaces:** validate_closure(config)->pinned inputs; review_own32(intent,raw)
produces measured identity-bound raw/runtime/parity evidence; prepare_phase(phase,
evidence)->independently backed release; process_terminal(slot,identity)->verified
backup/seal/measurement refs; science requires allsix seals and exact pose catalog.

- [ ] RED wrong first root/seal, any Mac executable ref or changed input/helper/
  release denies. Real unchanged gate accepts first48670474 anchor and forbids0replay.
- [ ] RED after first remaining slot completes, the next current slot passes its
  own absence check while earlier consumed roots/claims remain untouched; phase
  parent preparation is once-only, not repeated across the five.
- [ ] RED real NPZ contact/no-contact/missing/unsafe/801/native boundary cases,
  source/runtime/finite audit provenance errors and arbitrary PASS review denied.
- [ ] RED missing own seal/pose/tape blocks science; exactly12x160/allfive cases.
- [ ] Implement acyclic authority/control preservation, historical pose mirror with
  equal floats, audits for all actual cases, full measurement tables and reports.
- [ ] GREEN pure normal/-O; commit/review. No new physical probe or model execution.

### Task 4: Root complete proposal, native checks and sole-owner activation

- [ ] Complete whole-change review; preserve complete proposal and input controls.
- [ ] Fresh Vega quota/source/runtime/assets/interpreter/training/helper/remaining
  paths/claim checks; import unchanged first seal and preserve actual authority,
  final manifest/release/addendum; no operative Mac refs.
- [ ] Bounded native pure tests and controller capability/session lifecycle proof;
  no policy starts during tests. Root delegation and permanent lease activation.
- [ ] Submit controller once; verify actual source/lease/Slurm identity and
  disconnect independence. Observe first remaining policy RUNNING once.
- [ ] Monitor five qualifications and conditional full1920science without Mac
  intervention; preserve all receipts/reports and stop-on-failure evidence.

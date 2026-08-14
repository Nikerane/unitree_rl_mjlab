# Four-task CUDA instrumentation smoke — design

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer cap”
> wording and `[1.64, 3.28, ...]` below record the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The design body remains frozen;
> see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

**Status:** Revised after two independent design reviews; user review pending.
No commit, push, SSH, Slurm submission, GPU execution, training or policy
evaluation is authorized by this document.

## Purpose

Before the preregistered C/D-prime/F/E training matrix can run, prove that each
exact task package exposes live substep contact/impulse instrumentation on the
CUDA backend and that its fixed-impedance reward readers consume the expected
event state.

This is an instrumentation gate, not a policy-quality experiment and not a
miniature PPO run.

## Scope

One invocation validates one exact registered task on one device:

| Arm | Task |
|---|---|
| C | `Unitree-Z1-Hammer-CaT-Impulse` |
| D-prime | `Unitree-Z1-Hammer-CaT-Impulse-FirstStrike-Legacy` |
| F | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear` |
| E | `Unitree-Z1-Hammer-CaT-Impulse-Event` |

The prelaunch gate consists of four separate invocations. Each qualifying
process sees one GPU, one task and exactly **256 environments**, matching the
sampled evaluator's batch shape. The local implementation may use
`--device cpu` and a smaller explicit environment count only as an integration
test; CPU success is never represented as CUDA qualification.

Out of scope:

- PPO rollout/update or training;
- learned checkpoint loading;
- variable impedance or `set_gains`;
- changing reward weights, impulse caps, action scale or task semantics;
- statistical comparison between arms;
- computer-vision tracking or publication figures.

## Architecture

Add one thin task-aware smoke script. It reuses the live evaluator's task and
fixed-action contract validation instead of copying reward/cap/gain logic.

For the selected task the script:

1. Resolves and validates the exact registered training configuration and RL
   action clipping.
2. After validating the unmodified training configuration, applies only the
   diagnostic/campaign-safe overrides:
   `substep_impulse_rows.enabled=False` and deterministic reset/observations.
   Disabling row reconstruction matches the real 4×8 command and removes its
   per-substep host reads. Production `auto_reset=True` remains unchanged.
3. Immediately after environment construction and before reset or stepping,
   wraps each of the two resolved reward-term callables once to clone and latch
   its raw device tensor before RewardManager applies `nan_to_num`. This timing
   preserves mjlab's construction of stateful manager terms. The wrapper
   delegates exactly one call to the original function; the smoke never invokes
   a stateful reward reader a second time.
4. Runs `SingleStrikeReference` through the legal Differential IK action
   interface for its frozen sequence, then re-commands its final target for a
   six-step hold, matching the certified `playback_reference.py` semantics.
   The loop length is fixed in advance and does not branch on `done` or any
   device tensor. A controlled CPU probe rejected a zero-action hold: after
   the 10-step playback the head stopped at z≈0.1235 m above the ≈0.102 m nail
   top and never contacted; final-target hold contacted at step 11 and
   terminated successfully at step 13.
5. Accumulates contact, axial force, per-joint impulse, delivered impulse,
   joint velocity, nail depth, raw and manager reward activity, first-strike
   tracker activity and termination state entirely on device. A wrapper around
   `metrics_manager.compute` calls the original method, then device-latches
   each environment's first-terminal summary before mjlab autoresets it. Later
   episodes cannot overwrite that first-terminal summary.
6. After the final hold step, synchronizes once, copies the accumulated summary
   to the host, and evaluates tracker/sentinel and gate predicates.
7. Emits a provenance-bound JSON record outside the repository.
8. Exits nonzero if any required predicate fails.

The authoritative rollout must not insert `sim.forward()`, `.item()`,
`float(cuda_tensor)`, `int(cuda_tensor)`, `bool(cuda_tensor)`, CPU copies,
Torch synchronization or Warp synchronization inside the substep/control loop.
It accumulates device-side state and synchronizes only after the fixed rollout.
The implementation may reuse the timing model of `_PreIntegrationTracePhase`
for coherent pre/post joint velocity, but it must not reuse
`_SampledTraceCollector`, whose host conversions violate this qualification
contract.
If qualification fails, the existing CUDA diagnostic probe may be run in a
second fresh process; that synchronized diagnostic is never qualification
evidence.

## Configuration contract

Every invocation must verify:

- exact task-to-arm mapping;
- `impact_progress=8`, `delivered_impulse=2`;
- `imp_max_p=0`;
- manufacturer `IMP_J_LIMIT` unchanged;
- the complete fixed Differential IK action signature used by the 4×8
  campaign, including `delta_pos_scale=0.15`, clipping, `max_dq`, damping and
  all solver/action fields;
- RL `clip_actions=1.0`;
- fixed actuator gains/effort/armature and no gain-command action;
- the literal frozen manufacturer cap vector
  `(1.64, 3.28, 1.64, 1.64, 1.64, 1.64)` against both the live config and
  imported module constant;
- the frozen per-arm event reference impulse (`0.6094` for C/D-prime and
  `0.3088` for F/E);
- `substep_impulse_rows.enabled=False` in the effective smoke/training config;
- one assigned device and no multi-GPU process.

The deterministic strike may remove reset noise and actor-observation
corruption only after the unmodified training configuration has passed the
contract check. The output must state that this diagnostic reset is not the
stochastic sampled-evaluation distribution.

## Pass/fail contract

For CUDA instrumentation qualification, **all 256 environments** must satisfy
each per-env instrumentation predicate. The JSON reports pass/fail counts for
every predicate; a population maximum or “any environment” success is
insufficient. `hardware_qvel` is also all-environment evidence, but is reported
separately as a non-gating hardware-transport qualification.

### Common to C/D-prime/F/E

- all recorded observations, actions, reward components and physical signals
  are finite;
- the scripted strike makes hammer–nail contact;
- normal task success occurs before autoreset:
  `depth >= NAIL_SUCCESS_THRESHOLD` (currently 30 mm);
- the shipped per-joint impulse accumulator has a strictly positive maximum;
- the shipped delivered-impulse accumulator has a strictly positive maximum;
- the raw outputs of both configured reward readers are finite;
- both reward-manager contributions are strictly positive during the strike;
- `impossible_success_n == 0`;
- `lambda_dead_n == 0`;
- pre- and post-integration arm joint speeds are finite;
- finite exceedance of `3.1415 rad/s` is reported as the explicit
  `hardware_qvel` transport predicate and label, but does not fail
  integration or CUDA instrumentation qualification;
- code and asset repository revisions are known, clean before the run and
  unchanged after the run.

### D-prime/F/E only

- the production `FirstStrikeEventTracker` exists;
- `started` becomes true, which is the public accepted-onset predicate and
  logically proves prior arming;
- it finalizes as `productive` with `REASON_SUCCESS`;
- its delivered event impulse is strictly positive;
- within the first episode, each configured first-strike reward reader emits
  exactly one positive pulse, on its event-finalization/terminal control step,
  with no earlier positive pulse; the existing focused reward-reader tests
  separately prove that finalized tracker state cannot pay again;
- the task-specific first-strike reward readers are the exact configured
  legacy/event and linear/saturated variants.

C is deliberately exempt from tracker and one-pulse requirements because it
is the shipped repeated-credit legacy package; each of its two readers still
must produce at least one strictly positive manager contribution.

Both sentinel counts are derived through the existing evaluator
`_invariant_violations` logic after the fixed rollout, using the device-latched
first-terminal summary captured inside `metrics_manager.compute` before
autoreset. The smoke does not access private tracker arming state or invent a
separate sentinel formula.

## Timing semantics

Physical trace samples use the already reviewed explicit transition contract:

- head/site/contact/force are associated with pre-integration state `x_k`;
- cached nail depth and arm velocity provide the coherent pre-integration
  sample;
- post-integration depth and arm velocity are recorded separately.

The production tracker remains unchanged: its contact/force are associated
with `x_k`, while its endpoint depth is `x_(k+1)`. The smoke reports this
existing one-substep transition semantics and must not call it fully
synchronous.

## Output and provenance

JSON output is required and must contain:

- schema version, `integration_pass`, `cuda_qualification_pass`,
  `hardware_transport_qualified` and `hardware_transport_label`;
- arm, exact task ID, requested device and actual tensor/environment device;
- visible CUDA device count and physical device identity;
- environment count and per-predicate pass/fail counts;
- the full frozen task/action/reward contract digest;
- expected and observed code/asset full revisions plus pre/post dirty state;
- canonical sibling asset-repository path;
- scripted-reference identity;
- contact count, maximum force, nail depth;
- maximum per-joint impulse, delivered impulse and reward-reader totals;
- tracker state/reason where applicable;
- maximum pre/post arm joint speed and the non-gating hardware-transport
  predicate/label;
- sentinel counts;
- a list of failed predicates and diagnostic notes.

The caller supplies the expected full code revision, expected full asset
revision, canonical sibling asset-repository path and an output path outside
both repositories. The script rejects repository-local output paths and
refuses to overwrite an existing result. Writes use a fully flushed and
`fsync`ed temporary sibling, then atomically publish it with a same-filesystem
hard link. Link creation must fail with `EEXIST` for a regular path, symlink or
concurrent creator; the temporary link is then removed and the destination
directory is synchronized. `os.replace` is forbidden because it can overwrite
concurrently created evidence.

Output validation canonicalizes the parent directory but preserves the leaf
pathname; resolving the leaf itself would follow a dangling/existing symlink
and bypass the intended collision. Unsupported hard-link or directory-`fsync`
operations fail closed. A link failure leaves no final and cleans the
temporary; a directory-`fsync` failure is reported after publication, leaves
the already complete final intact, and still cleans the temporary.

CPU runs may set `integration_pass=true`, but
`cuda_qualification_pass` is always false. CUDA qualification additionally
requires exactly one visible GPU, real CUDA tensor/environment residency,
256 environments and a recorded physical device identity.

## Error handling

The smoke fails closed on:

- unknown or dirty provenance;
- observed code/asset revision differing from the expected deployment
  revisions before or after execution;
- task/config/action mismatch;
- missing task instrumentation;
- nonfinite values;
- no contact or insufficient depth advance;
- dead impulse/delivered paths;
- missing D-prime/F/E tracker activity;
- nonzero sentinels;
- nonfinite pre/post joint speed;
- output collision or provenance change during execution.

Failures still print a machine-readable diagnostic summary to stdout. A JSON
artifact is banked only if its provenance and output path remain valid.

## Tests

Use TDD.

1. Pure gate tests with literal records:
   - valid C and valid D-prime/F/E;
   - C tracker exemption;
   - missing tracker/onset/finalization for D-prime/F/E;
   - dead Λ/delivered/reward paths;
   - nonfinite data, sentinels, nonfinite pre/post qvel, and finite qvel
     transport-label behavior;
   - insufficient contact/depth;
   - dirty/changed provenance.
2. Contract tests bind the four exact task IDs to live reward classes,
   saturation modes, weights, caps, gains and complete DiffIK signature.
3. Output tests cover outside-repository enforcement, collision rejection,
   atomic write and schema completeness.
4. CPU integration runs the scripted strike once for each of the four tasks
   through the in-memory physics core and proves the harness reaches its
   physical/instrumentation checks while `cuda_qualification_pass` remains
   false. The provenance-enforcing CLI is tested separately against clean
   temporary repositories; there is no `--allow-dirty` qualification escape
   hatch.
5. Population tests prove that one failing environment out of 256 fails CUDA
   qualification for instrumentation predicates and that per-predicate counts
   are reported. A finite `hardware_qvel` exceedance changes only the
   hardware-transport qualification/label.
6. Timing tests fail if the authoritative rollout performs a substep/control
   host read, synchronization or tensor-dependent host branch; if
   `substep_impulse_rows` is enabled; if production autoreset is disabled; if a
   later episode overwrites the first-terminal latch; or if either raw reward
   callable is invoked more than once per RewardManager evaluation.
7. Independent reviewers inspect both code/runtime behavior and protocol
   compliance/YAGNI.

CPU integration cannot close the CUDA gate. The preregistration remains
`OPEN PRELAUNCH GATE` until the four clean Vega invocations pass.

## Planned command shape

The exact CLI is frozen by tests, with one task per process:

```text
.venv/bin/python scripts/smoke_first_strike_instrumentation.py \
  --task <exact-task-id> \
  --device cuda:0 \
  --num-envs 256 \
  --expected-code-revision <full-git-sha> \
  --expected-asset-revision <full-git-sha> \
  --asset-repo <canonical-sibling-path> \
  --out <external-attempt-directory>/<arm>.json
```

The four CUDA commands will be added to the preregistration after the local
implementation and reviews pass. They must be executed only from the same
clean code/asset revisions intended for the 4×8 training campaign.

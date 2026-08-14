# Four-task CUDA instrumentation smoke implementation plan

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer vector”
> wording and `[1.64, 3.28, ...]` below record the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The plan body remains frozen;
> see `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a task-aware, provenance-bound instrumentation smoke that must
pass for each exact C/D-prime/F/E task before the fixed-impedance 4×8 campaign
may be submitted.

**Architecture:** One focused script owns the frozen arm contract, pure gate,
runtime instrumentation and CLI. It validates the untouched training config
first, applies only deterministic diagnostic overrides, then runs a fixed
scripted strike with production autoreset and device-latches every
environment's first terminal state before reset. One focused test module covers
pure gates, live configuration drift, no-sync runtime semantics, provenance,
atomic output and four-task CPU integration.

**Tech Stack:** Python 3.10, PyTorch, mjlab 1.4.0, MuJoCo/MuJoCo Warp,
pytest, existing Z1 hammer task/evaluator utilities.

## Global constraints

- Work locally on the current branch and preserve the existing dirty worktree.
- Do not commit or push; the user's no-commit instruction overrides the skill's
  normal commit checkpoints.
- Do not use SSH, Slurm, Vega, CUDA, training or learned policy evaluation
  without new user authorization.
- Keep `IMP_J_LIMIT` at the literal manufacturer vector
  `(1.64, 3.28, 1.64, 1.64, 1.64, 1.64)`.
- Keep `imp_max_p=0`, `delta_pos_scale=0.15`, fixed actuator gains and
  `RL clip_actions=1.0`.
- Do not add variable impedance, `set_gains`, commanded stiffness, new reward
  terms or reward-weight changes.
- A qualifying invocation is exactly one registered task, one visible GPU and
  exactly 256 environments; all 256 must pass every per-environment predicate.
- The authoritative rollout performs no host read, synchronization or
  tensor-dependent Python branch inside its physics/control loop.
- CPU integration may set `integration_pass=true`, but always sets
  `cuda_qualification_pass=false`.
- Banked JSON must be outside both repositories, must not overwrite, and must
  bind clean pre/post code and asset revisions.

---

## File map

- Create `scripts/smoke_first_strike_instrumentation.py`
  - frozen arm semantics and schema;
  - pure pass/fail gate;
  - live configuration/provenance validation;
  - raw reward taps and device-side first-terminal recorder;
  - deterministic strike runner and CLI/atomic JSON writer.
- Create `tests/test_smoke_first_strike_instrumentation.py`
  - pure, configuration, provenance, timing and CPU integration tests.
- Modify `docs/results/2026-07-25_first_strike_reward_experiment.md`
  - add the reviewed command contract after local implementation passes;
  - leave the CUDA gate explicitly open until four clean Vega JSONs exist.
- Do not modify installed `mjlab`, `rsl_rl` or `mujoco_warp` packages.

---

### Task 1: Frozen schema and pure all-environment gate

**Files:**
- Create: `scripts/smoke_first_strike_instrumentation.py`
- Create: `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION: str = "four-task-cuda-smoke-v1"`
  - `ARM_CONTRACTS: dict[str, ArmContract]`
  - `ArmContract` frozen dataclass.
  - `PredicateCount` frozen dataclass with `passed`, `total` and `failed`.
  - `evaluate_gate(record: Mapping[str, Any]) -> dict[str, Any]`.
- `evaluate_gate` consumes only JSON-like values and performs no simulator or
  filesystem access.

- [ ] **Step 1: Write failing pure-gate tests**

Add literal-record helpers and tests proving:

```python
def test_valid_c_record_passes_without_tracker():
    record = valid_record(arm="C", num_envs=256, device_type="cuda")
    result = smoke.evaluate_gate(record)
    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["failed_predicates"] == []


@pytest.mark.parametrize("arm", ("D-prime", "F", "E"))
def test_first_strike_arm_requires_productive_success_and_one_terminal_pulse(arm):
    record = valid_record(arm=arm, num_envs=256, device_type="cuda")
    record["predicate_counts"]["productive_success"] = {"passed": 255, "total": 256}
    result = smoke.evaluate_gate(record)
    assert result["cuda_qualification_pass"] is False
    assert "productive_success" in result["failed_predicates"]


def test_finite_qvel_exceedance_is_transport_label_not_smoke_failure():
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["predicate_counts"]["hardware_qvel"] = {"passed": 255, "total": 256}
    result = smoke.evaluate_gate(record)
    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is True
    assert result["hardware_transport_qualified"] is False


def test_cpu_can_pass_integration_but_never_cuda_qualification():
    record = valid_record(arm="E", num_envs=8, device_type="cpu")
    result = smoke.evaluate_gate(record)
    assert result["integration_pass"] is True
    assert result["cuda_qualification_pass"] is False
```

Also cover zero Λ, zero delivered impulse, nonfinite raw reward, zero manager
contribution, `impossible_success_n > 0`, `lambda_dead_n > 0`, no 30 mm success,
wrong terminal reason, pulse before finalization, missing terminal pulse,
nonfinite post/pre qvel, finite qvel transport labelling above 3.1415, dirty
provenance, changed revision, wrong task and wrong environment count.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py -q
```

Expected: collection/import failure because the smoke module does not exist.

- [ ] **Step 3: Implement the minimal frozen contract and pure gate**

Use these exact arm semantics:

```python
@dataclass(frozen=True)
class ArmContract:
    arm: str
    task: str
    impact_reader: str
    delivered_reader: str
    tracker_required: bool
    event_i_ref_n_s: float
    delivered_saturate: bool | None


ARM_CONTRACTS = {
    "C": ArmContract(
        "C", ARM_TASKS["C"], "ImpactProgressTerm", "DeliveredImpulseTerm",
        False, 0.6094, None,
    ),
    "D-prime": ArmContract(
        "D-prime", ARM_TASKS["D-prime"],
        "FirstStrikeLegacyImpactRewardTerm",
        "FirstStrikeLegacyDeliveredRewardTerm", True, 0.6094, None,
    ),
    "F": ArmContract(
        "F", ARM_TASKS["F"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, False,
    ),
    "E": ArmContract(
        "E", ARM_TASKS["E"], "FirstStrikeImpactRewardTerm",
        "FirstStrikeDeliveredRewardTerm", True, 0.3088, True,
    ),
}
TASK_CONTRACTS = {contract.task: contract for contract in ARM_CONTRACTS.values()}
```

`evaluate_gate` must:

1. validate the schema/task/arm pairing;
2. require every common predicate count to equal `num_envs`;
3. require D-prime/F/E tracker predicates and exact-one terminal-pulse counts;
4. require clean unchanged provenance for integration pass;
5. set CUDA qualification only when integration passes, device type is CUDA,
   exactly one GPU is visible, actual tensor/environment devices are CUDA,
   physical device identity is nonempty and `num_envs == 256`;
6. return sorted `failed_predicates` and preserve every count.

- [ ] **Step 4: Run pure-gate tests and confirm GREEN**

Run the Task 1 tests only:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py \
  -k "gate or population or cpu_can_pass" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Review checkpoint**

Run `git diff --check` on the two named files. Do not commit. Have a reviewer
check that no “any environment” or population-maximum shortcut can pass.

---

### Task 2: Live task contract, provenance and atomic output

**Files:**
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Produces:
  - `validate_live_contract(task: str) -> tuple[Any, Any, dict[str, Any]]`,
    returning the untouched training env config, RL config and frozen digest.
  - `make_diagnostic_cfg(training_cfg: Any, num_envs: int) -> Any`.
  - `read_repo_provenance(path: Path, expected_revision: str) -> dict[str, Any]`.
  - `validate_output_path(out: Path, code_repo: Path, asset_repo: Path) -> Path`.
  - `write_json_atomic(out: Path, payload: Mapping[str, Any]) -> None`.

- [ ] **Step 1: Write failing live-contract drift tests**

Parameterize all four task IDs and assert the live contract includes:

```python
assert digest["impact_weight"] == 8.0
assert digest["delivered_weight"] == 2.0
assert digest["imp_max_p"] == 0.0
assert tuple(digest["impulse_limits_n_m_s"]) == (
    1.64, 3.28, 1.64, 1.64, 1.64, 1.64,
)
assert digest["rl_clip_actions"] == 1.0
assert digest["registered_substep_impulse_rows_enabled"] is True
assert digest["event_i_ref_n_s"] == smoke.TASK_CONTRACTS[task].event_i_ref_n_s
diagnostic_cfg = smoke.make_diagnostic_cfg(training_cfg, num_envs=8)
assert diagnostic_cfg.metrics["substep_impulse_rows"].params["enabled"] is False
```

Add mutation tests that fail independently for:

- swapped F/E `saturate`;
- wrong reward-reader class;
- missing/wrong `FirstStrikeEventTracker` on D-prime/F/E;
- wrong `i_ref`, impact/delivered weights or `imp_max_p`;
- drift in any complete DiffIK action field;
- drift in actuator stiffness/damping/effort/armature;
- live config cap drift;
- imported module `IMP_J_LIMIT` drift relative to the literal vector;
- `clip_actions != 1.0`;
- enabled `substep_impulse_rows` in the effective diagnostic config.

- [ ] **Step 2: Write failing provenance/output tests**

Use clean temporary Git repositories to prove:

```python
def test_atomic_writer_rejects_repo_local_and_existing_paths(tmp_path):
    code_repo = init_clean_repo(tmp_path / "code")
    asset_repo = init_clean_repo(tmp_path / "assets")
    with pytest.raises(ValueError, match="outside both repositories"):
        smoke.validate_output_path(
            code_repo / "result.json", code_repo, asset_repo
        )
    out = tmp_path / "external" / "result.json"
    out.parent.mkdir()
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})


def test_provenance_rejects_dirty_repo(tmp_path):
    repo = init_clean_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        smoke.read_repo_provenance(repo, git_head(repo))


def test_provenance_rejects_wrong_expected_revision(tmp_path):
    repo = init_clean_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="expected revision"):
        smoke.read_repo_provenance(repo, "0" * 40)


def test_post_run_revision_or_dirty_state_invalidates_payload():
    record = valid_record(arm="E", num_envs=256, device_type="cuda")
    record["provenance"]["code"]["post_dirty"] = True
    result = smoke.evaluate_gate(record)
    assert result["cuda_qualification_pass"] is False
    assert "code_provenance" in result["failed_predicates"]


def test_atomic_writer_leaves_no_partial_file_on_link_failure(
    tmp_path, monkeypatch
):
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        smoke.os, "link",
        lambda source, target: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        smoke.write_json_atomic(out, {"schema_version": smoke.SCHEMA_VERSION})
    assert not out.exists()
    assert list(tmp_path.glob(".result.json.*.tmp")) == []
```

The output path test must resolve the output parent (including parent symlinks)
and reject descendants of either canonical repository, but must preserve the
leaf name so an existing or dangling leaf symlink remains an `EEXIST`
collision. The writer must use `tempfile.NamedTemporaryFile` in the destination
directory, `flush()`, and `os.fsync()`, then atomically publish with
`os.link(temp_path, final_path)`. It must treat `FileExistsError` as a collision
for regular files, symlinks (including dangling symlinks), and a concurrent
creator; it must remove the temporary link and `fsync` the destination
directory. Tests identify the directory descriptor with `os.fstat` at the
`fsync` call, cover generic hard-link failure, and require a directory-`fsync`
failure to propagate while retaining the already published complete final and
cleaning the temporary name. Never use `os.replace`, because it permits
overwrite.

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py \
  -k "contract or provenance or output or atomic" -q
```

Expected: failures for the missing validation/output functions.

- [ ] **Step 4: Implement exact live validation**

`validate_live_contract` must first call the existing
`scripts.eval_impulse._validate_sampled_env_contract(training_cfg, task)`, then
add the missing task-specific checks:

```python
impact = training_cfg.rewards["impact_progress"]
delivered = training_cfg.rewards["delivered_impulse"]
assert type(impact.func).__name__ == contract.impact_reader
assert type(delivered.func).__name__ == contract.delivered_reader
assert float(delivered.params["i_ref"]) == contract.event_i_ref_n_s
if contract.delivered_saturate is not None:
    assert delivered.params["saturate"] is contract.delivered_saturate
```

For function/class-valued configs, normalize with
`func.__name__` when `inspect.isclass(func)`, otherwise
`type(func).__name__`. D-prime/F/E must have
`FirstStrikeEventTracker`, `per_substep=True`, `reduce="last"`; C must retain
its exact legacy reward readers and is not required to add a tracker.

Compare both `training_cfg.metrics["cat_soft"].params["imp_limit"]` and the
imported `IMP_J_LIMIT` to the literal tuple. Load the task RL config and require
`float(agent_cfg.clip_actions) == 1.0`.

The live digest records that the registered task contains the row diagnostic
and the effective runtime digest records that the campaign override disables
it. `make_diagnostic_cfg` must deep-copy only after validation, then set:

```python
cfg.scene.num_envs = num_envs
cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
cfg.observations["actor"].enable_corruption = False
cfg.observations["critic"].enable_corruption = False
cfg.metrics["substep_impulse_rows"].params["enabled"] = False
```

It must not change autoreset, rewards, terminations, gains, caps, action scale,
physics timestep or decimation.

- [ ] **Step 5: Implement provenance and output**

Require full 40-character lowercase hexadecimal revisions, canonical paths,
clean `git status --porcelain`, and exact `git rev-parse HEAD` matches. Capture
the same values before and after runtime and make them gate predicates.

Do not add `--allow-dirty` or a repository-local output fallback.

- [ ] **Step 6: Run Task 2 tests and confirm GREEN**

Run the Task 2 selection plus Task 1:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py \
  -k "not cpu_integration and not runtime" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Review checkpoint**

Do not commit. Ask a protocol reviewer to compare the digest against
`FROZEN_CAMPAIGN_MATRIX`, `EXPECTED_FIXED_ACTION_SIGNATURE`,
`EXPECTED_FIXED_ACTUATOR_SIGNATURE`, `ARM_EVENT_I_REF_N_S` and the four
registered configs.

---

### Task 3: Raw reward taps and device-side first-terminal recorder

**Files:**
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Produces:
  - `RawRewardTap`, which delegates one call and latches raw device tensors.
  - `DeviceSmokeRecorder.install(env) -> None`.
  - `DeviceSmokeRecorder.finalize() -> dict[str, Any]`, called only after the
    fixed rollout and its single synchronization.

- [ ] **Step 1: Write failing exactly-once reward-tap tests**

Use a fake callable with a call counter:

```python
def test_raw_reward_tap_delegates_exactly_once_and_preserves_nonfinite_raw():
    original = CountingReward(torch.tensor([float("nan"), 2.0]))
    tap = smoke.RawRewardTap(original, num_envs=2, device=torch.device("cpu"))
    returned = tap(fake_env)
    assert original.calls == 1
    assert torch.isnan(tap.last_raw[0])
    torch.testing.assert_close(returned[1], torch.tensor(2.0))
```

Also prove `reset(env_ids)` delegates once for stateful reward terms and clears
only the tapped episode state for those IDs.

- [ ] **Step 2: Write failing first-terminal latch tests**

With tensor-only fake environment state, simulate staggered termination:

```python
first_payload = terminal_payload(
    reset_buf=torch.tensor([True, False]),
    terminal_depth=torch.tensor([0.031, 0.010]),
)
recorder.capture_control_step(**first_payload)
first = recorder.terminal_depth.clone()
second_payload = terminal_payload(
    reset_buf=torch.tensor([True, True]),
    terminal_depth=torch.tensor([0.001, 0.031]),
)
recorder.capture_control_step(**second_payload)
assert recorder.terminal_depth[0] == first[0]  # no overwrite
assert recorder.terminal_seen.tolist() == [True, True]
```

Prove the latch captures:

- terminal depth and normal-success flag;
- episode-peak six-joint Λ and delivered impulse;
- tracker `started/finalized/productive/reason/delivered` for D-prime/F/E;
- raw reward finiteness before RewardManager sanitization;
- weighted manager contribution and positive-pulse count/step;
- `contact_seen`, force maximum and pre/post qvel maxima accumulated before
  terminal;
- every field independently per environment.

- [ ] **Step 3: Write failing no-host-action tests**

Instrument/monkeypatch forbidden operations so they raise while the recorder's
substep and control callbacks execute:

- `Tensor.item`, `Tensor.cpu`, `Tensor.numpy`;
- `torch.cuda.synchronize`;
- Warp synchronization;
- tensor `__bool__`;
- `env.sim.forward`.

The callbacks must pass without invoking any of them. A negative-control
callback containing `.item()` must fail so the guard is proven live.

- [ ] **Step 4: Run Task 3 tests and confirm RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py \
  -k "reward_tap or terminal_latch or no_host" -q
```

Expected: failures for missing runtime classes.

- [ ] **Step 5: Implement `RawRewardTap`**

After constructing the environment, replace only the two selected
`reward_manager` term-config callables:

```python
impact_cfg = env.reward_manager.get_term_cfg("impact_progress")
delivered_cfg = env.reward_manager.get_term_cfg("delivered_impulse")
impact_tap = RawRewardTap(impact_cfg.func, env.num_envs, env.device)
delivered_tap = RawRewardTap(delivered_cfg.func, env.num_envs, env.device)
impact_cfg.func = impact_tap
delivered_cfg.func = delivered_tap
```

The tap calls `original(env, **params)` exactly once, clones its raw result
before RewardManager's `nan_to_num`, updates device-side finite/positive/count
tensors, returns the unmodified result, and delegates `reset` when the original
has it.

- [ ] **Step 6: Implement `DeviceSmokeRecorder` callbacks**

Install callbacks in this order:

1. wrap `env.sim.step` only to cache pre-integration arm qvel;
2. wrap `metrics_manager.compute_substep`, call the original first, then
   accumulate contact/force and coherent pre/post qvel/depth on device;
3. wrap `metrics_manager.compute`, call the original first, update current
   first-episode reward data, then latch `env.reset_buf & ~terminal_seen`.

Use `torch.where`, masked assignment and `torch.maximum`; never use a Python
condition on a tensor. Use the shipped accumulator objects and the public
tracker properties. Preserve original methods for teardown.

`finalize()` is the only method allowed to synchronize/copy to CPU. It converts
the per-env tensors into predicate counts and then calls
`scripts.eval_impulse._invariant_violations` on the first-terminal records.

- [ ] **Step 7: Run Task 3 tests and confirm GREEN**

Run the Task 3 selection, then all non-integration smoke tests.

- [ ] **Step 8: Review checkpoint**

Do not commit. Have a runtime reviewer inspect callback ordering against
`ManagerBasedRlEnv.step`: termination → reward → metrics compute → autoreset.
Have a second reviewer check that raw rewards are not evaluated twice.

---

### Task 4: Deterministic runner, CPU integration and CLI

**Files:**
- Modify: `scripts/smoke_first_strike_instrumentation.py`
- Modify: `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Produces:
  - `run_smoke(task: str, device: str, num_envs: int) -> dict[str, Any]`.
  - `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`.
  - `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing CLI-shape tests**

Require exactly:

```text
--task
--device
--num-envs
--expected-code-revision
--expected-asset-revision
--asset-repo
--out
```

Reject unknown task IDs, `num_envs < 1`, CUDA qualification with anything
other than 256, output collisions, noncanonical asset repositories and missing
expected revisions.

- [ ] **Step 2: Write failing four-task CPU integration tests**

Parameterize the exact task IDs. Use one or a small explicit CPU environment
count and assert:

```python
result = smoke.run_smoke(task=task, device="cpu", num_envs=1)
assert result["integration_pass"] is True
assert result["cuda_qualification_pass"] is False
assert result["predicate_counts"]["normal_success"]["passed"] == 1
assert result["predicate_counts"]["lambda_live"]["passed"] == 1
assert result["predicate_counts"]["delivered_live"]["passed"] == 1
assert math.isfinite(result["max_pre_arm_qvel_rad_s"])
assert math.isfinite(result["max_post_arm_qvel_rad_s"])
# Finite overspeed is retained as hardware-transport evidence, not a smoke
# integration/CUDA-instrumentation failure.
assert "hardware_transport_qualified" in result
```

For D-prime/F/E additionally require productive `REASON_SUCCESS` and exactly
one terminal pulse for both configured readers. For C require at least one
positive contribution from both repeated-credit readers.

- [ ] **Step 3: Run CLI/integration tests and confirm RED**

Run the CLI unit subset first, then one CPU arm to size runtime:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py \
  -k "cli or cpu_integration" -q
```

- [ ] **Step 4: Implement the fixed runner**

After config validation/diagnostic override:

1. construct raw `ManagerBasedRlEnv`;
2. install taps and recorder before reset;
3. wrap with `RslRlVecEnvWrapper(env, clip_actions=1.0)`;
4. reset once;
5. build `SingleStrikeReference(num_envs, env.device, approach_height=0.10)`;
6. compute `playback_length` once as a Python integer before the loop;
7. run exactly `playback_length + 6` control steps;
8. use reference targets during the frozen playback prefix and re-command the
   final reference target during the six-step hold, matching
   `playback_reference.py` (the rejected zero-action variant stopped the head
   above the nail and produced no contact);
9. never inspect `done` or device tensors inside the loop;
10. synchronize once after the loop, finalize, close in `finally`.

The scripted action remains:

```python
action = ((target - head_position) / 0.15).clamp(-1.0, 1.0)
```

and is sent through the same `RslRlVecEnvWrapper` clipping layer used by the
campaign. Do not call `sim.forward()` inside the loop.

- [ ] **Step 5: Implement device qualification and CLI lifecycle**

For CUDA, require:

- requested and actual environment/tensor device `cuda:0`;
- `torch.cuda.device_count() == 1`;
- nonempty `torch.cuda.get_device_name(0)`;
- recorded device UUID/PCI identity when exposed by the runtime;
- `num_envs == 256`.

Require finite pre/post qvel for integration and CUDA instrumentation
qualification. Report the all-environment `hardware_qvel` predicate plus
`hardware_transport_qualified`/`hardware_transport_label` separately; finite
rail exceedance must not change either instrumentation pass bit.

CLI order:

1. validate arguments/output path;
2. capture/validate pre-run code and asset provenance;
3. run the smoke;
4. capture/validate post-run provenance;
5. evaluate the pure gate;
6. print a compact JSON summary to stdout;
7. atomically write the full JSON only when provenance is still valid;
8. return zero only for CUDA qualification on CUDA requests, or integration
   pass on explicit CPU requests.

- [ ] **Step 6: Run all focused smoke tests**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  -m pytest tests/test_smoke_first_strike_instrumentation.py -q
```

Expected: all focused tests pass. Record test count and wall time.

- [ ] **Step 7: Run the four CPU integrations as separate processes**

Use an external temporary directory and one task per process. These are
integration evidence only; they must report
`cuda_qualification_pass=false`. Because the current working repository is
dirty by instruction, exercise `run_smoke` in tests rather than inventing a
dirty-provenance CLI bypass.

- [ ] **Step 8: Review checkpoint**

Do not commit. Two independent reviewers must approve:

- runtime/code correctness, callback ordering, CUDA no-sync semantics;
- protocol/YAGNI, all-256 logic, frozen task identities and provenance.

Fix load-bearing findings with new failing tests before changing production
code.

---

### Task 5: Cross-module verification and preregistration update

**Files:**
- Modify: `docs/results/2026-07-25_first_strike_reward_experiment.md`
- Modify only if a failing regression proves necessary:
  `scripts/smoke_first_strike_instrumentation.py`
- Modify only if a failing regression proves necessary:
  `tests/test_smoke_first_strike_instrumentation.py`

**Interfaces:**
- Produces a locally verified smoke implementation and a still-open,
  executable CUDA preregistration gate.

- [ ] **Step 1: Run focused and cross-module tests**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_smoke_first_strike_instrumentation.py \
  tests/test_first_strike_campaign.py \
  tests/test_eval_impulse_hook.py \
  tests/test_first_strike_event.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_cat_soft_hook.py -q
```

Any failure blocks documentation changes. Diagnose before editing.

- [ ] **Step 2: Run the repository's pre-training gates locally**

Run all phases A–M and the contact sensor verifier:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  validate_rewards.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  verify_contact_sensor.py
```

Also run `verify_reward_setup.py`; if its unseeded random sweep misses a sparse
term, report that exact nondeterminism and rerun unchanged once, without
weakening its gate.

- [ ] **Step 3: Update the preregistration without claiming CUDA success**

Replace the “no runnable smoke command exists” wording with the reviewed CLI
shape and four exact task commands. Keep:

```text
OPEN PRELAUNCH GATE — local CPU integration passed; four clean 256-env CUDA
qualification JSONs are still required before any 4×8 submission.
```

Do not add fabricated revisions, Vega output, pass rows or result numbers.

- [ ] **Step 4: Final static verification**

Run:

```bash
git diff --check -- \
  scripts/smoke_first_strike_instrumentation.py \
  tests/test_smoke_first_strike_instrumentation.py \
  docs/results/2026-07-25_first_strike_reward_experiment.md \
  docs/superpowers/specs/2026-07-25-four-task-cuda-smoke-design.md \
  docs/superpowers/plans/2026-07-25-four-task-cuda-smoke.md
rg -n "TBD|TODO|FIXME|allow-dirty|cuda_qualification_pass.*true" \
  scripts/smoke_first_strike_instrumentation.py \
  tests/test_smoke_first_strike_instrumentation.py \
  docs/results/2026-07-25_first_strike_reward_experiment.md
```

Inspect `git status --short` and name only the files created/modified by this
work. Preserve all unrelated user changes.

- [ ] **Step 5: Stop at the external-action boundary**

Report:

- focused and cross-module test counts;
- CPU integration outcomes for C/D-prime/F/E;
- both reviewer verdicts and resolved findings;
- that CUDA qualification, Vega submission, training and evaluation remain
  unrun;
- exact external next step, requiring fresh user authorization:
  clean commit/push/pull followed by four one-task, 256-environment Vega smoke
  invocations.

Do not commit, push, SSH or submit the campaign.

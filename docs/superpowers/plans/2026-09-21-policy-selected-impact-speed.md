# Policy-Selected Impact Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the smallest experimental extension in which one actor chooses a stroke speed and continues controlling the existing joint targets and impedance.

**Architecture:** Append one latched speed action to the existing twelve motor/gain actions. A masked actor distribution and two fixed-scale observation features support the once-per-stroke decision without replacing PPO, CaT or the controller. Qualification and evaluation tools distinguish a usable command from an ignored or merely predictive output.

**Tech Stack:** Existing Python/torch/tensordict, mjlab 1.4.0, installed rsl_rl, MuJoCo/mujoco_warp 3.8.1, pytest. No new dependency.

**Spec:** [Approved design](../specs/2026-09-21-policy-selected-impact-speed-design.md), including section 10's post-approval code-review corrections.

## Global Constraints

- Start from clean pilot commit `3ca89a297a58d8e59d74c4797caeb8629499f1aa`, not the dirty main checkout. Execution uses an isolated worktree through the worktree skill. Preserve all unrelated changes.
- Preserve `[joint_position(6), joint_stiffness(6)]` meanings and ordering; append `strike_speed(1)`. Keep six-position action history, action-rate penalty, stiffness mapping `C=1.25`, damping coupling, quintic geometry and 20 mm followthrough unchanged.
- One online actor, not two learned policies; no reference-timing generator, fixed acceleration profile or actuator rewrite.
- Existing tasks/checkpoints/weights/soft-CaT mathematics remain unchanged. The new negative cost is opt-in and classified explicitly.
- Experimental actor/critic observations are exactly 42 coordinates: existing40, normalized speed, pending flag. Normalize only the original40; initialize pending-input columns to zero.
- First self-selected step executes all13 outputs; subsequent steps execute only12. External-goal episodes have pending0 from their first step. No added physical hold step.
- This plan delivers software and qualification tools. It does **not** choose robot-specific speed bounds, cost weights, checkpoint identities or authorize a GPU campaign. Those need a frozen evidence/run record; required values fail closed when absent.
- No broad refactor, dependency patch, historical evaluator rewrite or removal of old evidence/tests. Keep PT and ONNX compatibility claims separate from successful file creation.

## Review Focus

1. External goal reset occurs before/after other reset components: action-term ownership must prevent accidental erasure (Task 1).
2. Pending flag first becomes one after qualification: actor/critic inputs must remain bounded and export-identical (Task 2).
3. Infinity is hidden by clipping: reject raw non-finite values before the wrapper clips them (Task 4).
4. Finalization coincides with timeout/reset: charge one correct event cost, never a duplicate miss (Task 3).
5. Selection is rare or absent in a rollout: zero-selection telemetry must remain finite and not count ignored outputs as chosen goals (Task 5).

## File map and execution rules

All implementation paths below are relative to the isolated pilot-derived worktree. Use `/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python` for local commands; if unavailable, use `conda run -n unitree_mjlab python` explicitly. Run commands from the worktree, not the main checkout.

New files:

- `src/tasks/hammer/mdp/impact_speed.py`: bounded goal state, action term, observations and one-shot event cost; consumes the existing tracker.
- `src/tasks/hammer/rl/impact_speed_policy.py`: fixed-tail normalization, masked Gaussian/model and qualification-to-selection initialization.
- `src/tasks/hammer/rl/impact_speed_wrapper.py`: pre-clipping finite validation and experimental wrapper selection.
- `src/tasks/hammer/rl/impact_speed_contract.py`: experimental metadata and telemetry, avoiding further large additions to the legacy runner.
- `scripts/evaluate_impact_speed.py`: bounded evaluator and integration smoke, using the native safety-on task.
- `tests/test_impact_speed.py`, `tests/test_impact_speed_policy.py`, `tests/test_impact_speed_contract.py`, `tests/test_impact_speed_evaluation.py`.

Modify only the task/config factories, `mdp/trackability.py`, `cat/hook.py`, narrow runner dispatch points, `scripts/train.py` and `scripts/play.py`, plus directly affected tests/docs. No modifications to `CatPPO`, `CatRolloutStorage` or installed dependencies are expected.

Each task: a fresh implementation subagent, an independent specification review and a code-quality review before proceeding. Do not parallelize overlapping edits. Parallel review is appropriate. Commit each accepted task using explicit file lists, not `git add .`.

## Task 1: Goal-state and action contract

**Files:** create `mdp/impact_speed.py` and `tests/test_impact_speed.py` under the paths above.

**Interfaces:**

- `ImpactSpeedState(num_envs: int, device: str | torch.device, bounds: tuple[float, float], mode: Literal["external", "policy"])` owns immutable `.bounds`, float tensors `goal_mps[B]`, `pending[B]`, and boolean `cost_consumed[B]`.
- `reset(env_ids: torch.Tensor | slice | None) -> None`: subset-only; external mode samples uniform normalized goals with torch's seeded RNG, policy mode clears goal to midpoint/sentinel and sets pending1. No reset event owns these fields.
- `select(raw: torch.Tensor) -> None`: accepts `[B,1]`, validates finite values, clips±1, updates only pending worlds, then clears their pending flags.
- `observation() -> torch.Tensor`: `[B,2]` normalized goal and pending. `normalized_goal` sentinel is zero before policy selection.
- `ImpactSpeedActionCfg(ActionTermCfg)` has `bounds: tuple[float,float] | None`, `mode`, and `build(env)`; `ImpactSpeedAction` owns `.state`, exposes `action_dim=1`, `raw_actions` and `processed_actions`, latches in `process_actions`, does nothing in `apply_actions`, resets in `reset(env_ids)`.
- Observation functions `impact_speed_goal(env)` and `impact_speed_pending(env)` return `[B,1]` from `env.action_manager.get_term("strike_speed").state`.

- [ ] Write tests including the following literal contract (test bounds are arithmetic fixtures, not approved robot bounds):

```python
def test_policy_goal_latches_once_and_resets_only_subset():
    s = ImpactSpeedState(2, "cpu", (0.2, 0.8), "policy")
    s.reset(None)
    s.select(torch.tensor([[-1.0], [1.0]]))
    torch.testing.assert_close(s.goal_mps, torch.tensor([0.2, 0.8]))
    s.select(torch.zeros(2, 1))
    torch.testing.assert_close(s.goal_mps, torch.tensor([0.2, 0.8]))
    s.reset(torch.tensor([0]))
    assert s.pending.tolist() == [1.0, 0.0]
    assert s.goal_mps[1].item() == pytest.approx(0.8)

def test_external_goal_cannot_be_overwritten():
    s = ImpactSpeedState(3, "cpu", (0.2, 0.8), "external")
    s.reset(None)
    before = s.goal_mps.clone()
    s.select(torch.ones(3, 1))
    torch.testing.assert_close(s.goal_mps, before)
    assert not s.pending.any()
```

- [ ] Run `python -m pytest tests/test_impact_speed.py -q`; confirm failure is missing new implementation, not missing runtime dependencies.
- [ ] Implement the state mapping with shape/domain validation (`0 < lower < upper`, both finite, reject bools), vectorized subset ownership, and action adapter following `JointStiffnessActionCfg`/`ActionTerm` APIs. Core mapping:

```python
lo, hi = self.bounds
candidate = lo + 0.5 * (raw[:, 0].clamp(-1, 1) + 1) * (hi - lo)
selected = self.pending.bool()
self.goal_mps.copy_(torch.where(selected, candidate, self.goal_mps))
self.pending.zero_()
```

- [ ] Add tests for NaN/±inf, invalid widths/bounds, repeated substep `apply_actions` not relatching, empty/subset resets, and external seeded reproducibility. Test actual manager reset ordering using the project's existing fake-environment patterns.
- [ ] Run the file green; review that none of the first12 target/gain logic was reimplemented. Commit only the new module and tests.

## Task 2: Masked learning and fixed-scale inputs

**Files:** create `rl/impact_speed_policy.py`, `tests/test_impact_speed_policy.py`.

**Interfaces:** `ImpactSpeedNormalization(nn.Module)`, `ImpactSpeedGaussian(GaussianDistribution)`, `ImpactSpeedModel(MLPModel)`. The same model class supports stochastic13-output actor and deterministic1-output critic. Only stochastic actor uses action masking.

- [ ] Write tests for inactive last-coordinate log-probability/entropy/KL, all13 active at pending1, stored-mask roundtrip through actual RolloutStorage, invalid old/new masks, and unchanged raw actions under wrapper clipping. Include a normalization transition test:

```python
def test_pending_does_not_become_100_after_qualification():
    norm = ImpactSpeedNormalization()
    external = torch.zeros(16, 42)
    norm.update(external)
    autonomous = external[:1].clone()
    autonomous[:, -1] = 1
    assert norm(autonomous)[0, -1].item() == 1
```

- [ ] Run `python -m pytest tests/test_impact_speed_policy.py -q`; confirm red.
- [ ] Implement the fixed-tail normalizer using the installed empirical normalizer only on the original40 coordinates:

```python
class ImpactSpeedNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.physical = EmpiricalNormalization(40)

    def forward(self, x):
        return torch.cat((self.physical(x[..., :40]), x[..., 40:]), dim=-1)

    @torch.jit.unused
    def update(self, x):
        self.physical.update(x[..., :40])
```

- [ ] In the model constructor, require `obs_dim==42`, retain empirical normalization enabled but replace its module above, and set the last input column of the first `nn.Linear` weight to zero under `torch.no_grad()`. Leave it trainable. In `get_latent`, collect raw observations using `self.obs_groups`, verify binary raw pending, set `distribution.activity_mask = cat(ones12, pending)`, then return `super().get_latent(...)`. Parent `forward` handles trajectory unpadding; do not duplicate it.
- [ ] Implement distribution reductions. `params` returns `(mean, std, activity_mask.clone())`. Use the installed distribution's per-coordinate Normal statistics and Gaussian KL formula:

```python
def log_prob(self, outputs):
    return (self._distribution.log_prob(outputs) * self.activity_mask).sum(-1)

@property
def entropy(self):
    return (self._distribution.entropy() * self.activity_mask).sum(-1)

def kl_divergence(self, old_params, new_params):
    om, os, mask = old_params
    nm, ns, new_mask = new_params
    if not torch.equal(mask, new_mask):
        raise ValueError("recorded action masks disagree")
    elementwise = torch.log(ns / os) + (os.square() + (om-nm).square()) / (2*ns.square()) - 0.5
    return (elementwise * mask).sum(-1)
```

- [ ] Test mixture minibatches, a rollout with no selections, inherited deterministic export modules, normalizer state reload, zero inactive selector gradients and nonzero active gradients. Compare first12 actor outputs/critic values at pending0 versus1 immediately before autonomous training with zero pending columns and identical remaining inputs. This tests unintended flag jumps, not future trained invariance.
- [ ] Run new tests plus `tests/test_cat_ppo_gae.py tests/test_cat_storage.py`; inspect that stock PPO and CaT return code remain untouched. Commit the module/tests.

## Task 3: Exactly-once speed-consistency cost

**Files:** extend `mdp/impact_speed.py`, `tests/test_impact_speed.py`; narrowly edit `cat/hook.py` and `tests/test_cat_soft_hook.py`.

**Interfaces:** `consume_speed_cost(state, *, finalized: Tensor, started: Tensor, measured_mps: Tensor, done: Tensor) -> Tensor[B]` computes nonnegative dimensionless cost and marks consumed worlds. `ImpactSpeedCost(ManagerTermBase)` calls it using the existing tracker and `env.reset_buf`, returning `cost / env.step_dt`. Config reward term name is `speed_goal`, weight is negative event coefficient.

- [ ] Write literal outcome tests: finalized+done must use measured error; started+unfinalized+done costs1; no accepted contact+done costs1; neither finalized nor done costs0; a second read always costs0. A finalized nonproductive event still consumes its speed error. A missing/nonfinite finalized measurement raises; no synthetic zero speed.

```python
def test_event_charge_is_independent_of_control_dt():
    for dt in (0.01, 0.02, 0.04):
        raw = torch.tensor([0.25]) / dt
        coefficient = 0.4
        torch.testing.assert_close(raw * (-coefficient) * dt, torch.tensor([-0.1]))
```

- [ ] Run `python -m pytest tests/test_impact_speed.py tests/test_cat_soft_hook.py -q`; confirm new-case failures.
- [ ] Implement the tensor-level consume function with finalized precedence and one consumed latch:

```python
lo, hi = state.bounds
eligible = ~state.cost_consumed
event = eligible & finalized
failure = eligible & ~finalized & done
error = ((measured_mps - state.goal_mps) / (hi - lo)).square().clamp_max(1)
cost = torch.where(event, error, torch.zeros_like(error))
cost = torch.where(failure, torch.ones_like(cost), cost)
state.cost_consumed |= event | failure
```

Use validity checks only on event measurements; missing measurements outside events are not failures. `started` determines failure labels (miss versus incomplete), not cost magnitude. Copy event outcome/goal/speed evidence into per-transition extras before reset for the evaluator; never read reset-cleared state as terminal evidence.

- [ ] Add `speed_goal` explicitly to the known negative-term allowlist. Test that unknown negative terms still fail, old task `r_pos` is unchanged, and `speed_goal` is excluded from positive return. Test real reward manager dt scaling, not just the algebra fixture.
- [ ] Run affected tests and `tests/test_first_strike_event.py tests/test_delivered_impulse_reward.py`; review no productive-only helper reuse or callback mutation. Commit explicit files.

## Task 4: Isolated task registration and raw-action guard

**Files:** modify `config/z1/__init__.py`, `config/z1/rl_cfg.py`, `mdp/trackability.py`, `scripts/train.py`, `scripts/play.py`; create `rl/impact_speed_wrapper.py`; extend new tests and `tests/test_joint_position_config.py`, `tests/test_joint_trackability.py`.

**Interfaces:** experimental task ID `Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed`; `_impact_speed_env_cfg(play: bool)` derives `_quintic_pilot_env_cfg` and appends the new term/observations/cost. `z1_impact_speed_ppo_runner_cfg()` derives the existing RL config and changes only model/distribution identities. `wrap_impact_speed_env(env, clip_actions)` selects the guarded wrapper only for the exact experimental action signature; otherwise returns the existing wrapper.

- [ ] Write configuration comparison tests: unchanged first12 action classes/parameters, goal observations appended in both groups, original action history/penalty still position-only, original rewards and safety parameters identical except the explicit new term. Legacy factory outputs must remain identical.
- [ ] Write pre-clip rejection tests with a fake underlying env that counts calls:

```python
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_never_reaches_environment(bad, guarded_wrapper):
    action = torch.zeros(1, 13)
    action[0, 12] = bad
    with pytest.raises(ValueError, match="finite"):
        guarded_wrapper.step(action)
    assert guarded_wrapper.env.step_calls == 0
```

- [ ] Run configuration/wrapper tests red, then implement the wrapper with `torch.isfinite(actions).all()` before `super().step(actions)`. Initially prefer explicit correctness over eliminating its reduction; do not claim it is cost-free on CUDA. Use this wrapper in train, play and new evaluation. Direct action term finite checks remain in place.
- [ ] Register only the experimental task. Its `ImpactSpeedActionCfg.bounds=None` deliberately cannot build until the caller supplies a reviewed range; the factory can be inspected without constructing an environment. Likewise require an explicit negative `speed_goal` weight at environment construction (default0 is not a valid training setting). CLI/config and evaluator must reject missing bounds/coefficient, not silently use arithmetic test values. Append the new named negative reward only in this task.
- [ ] Set actor and critic `class_name` to the new model, actor distribution to the new Gaussian. Admit the exact three-term signature in trackability without accepting arbitrary extra terms. Use the new term's params for mode (`external`/`policy`) rather than creating two near-identical task registrations.
- [ ] Run legacy config tests plus new ones. Review the native action mapping diff is empty. Commit the focused changes.

## Task 5: Checkpoint, telemetry and export contracts

**Files:** create `rl/impact_speed_contract.py`, `tests/test_impact_speed_contract.py`; narrowly modify `rl/runner.py`; preserve existing `tests/test_vic_rollout_telemetry.py` expectations.

**Interfaces:** `build_impact_speed_metadata(env, raw_policy_clip: float) -> dict`, `summarize_impact_speed_rollout(storage, bounds: tuple[float,float]) -> dict | None`, `validate_impact_speed_metadata(saved: dict, expected: dict, *, allow_phase_transfer: bool=False) -> None`. Schema identity `impact_speed_v1`, not a reused legacy integer meaning.

- [ ] Write tests rejecting wrong `[6,6,1]` dimensions/order, missing bounds, wrong goal axis/observation order/normalizer, stale gain mapping and absent checkpoint metadata. Unchanged legacy schemas must still reject 13D under their old identities.
- [ ] Test zero-selection rollouts: `selection_count=0`, selected-speed quantiles are `None` rather than NaN, gain telemetry still present. Active selection statistics use saved raw actions clipped/mapped only where saved pending mask=1. Gain slices are `6:12`, never `6:`.
- [ ] Run red, then implement explicit new-schema dispatch before legacy metadata/telemetry branches. Extract reusable legacy validation into a narrowly parameterized helper only if needed; do not spoof a12D environment or weaken validators globally.

```python
selected = storage.distribution_params[2][..., 12].bool()
sampled = storage.actions[..., 12][selected].clamp(-1, 1)
physical = lo + (sampled + 1) * 0.5 * (hi - lo)
selection_count = int(selected.sum())
```

- [ ] Add contract metadata to PT `infos` **before** save and validate before normal resume. Copy existing schema fields for actuator/controller/safety provenance; add action names/widths, obs names/widths, fixed-tail normalization, bounds, axis, mode and cost units. Phase-transfer is an explicit path, not ordinary resume bypass.
- [ ] Export using inherited deterministic modules with the replacement normalizer. Test eager/PT/ONNX output parity if runtimes are installed. Exercise reset/latch/feedback around exported inference, not only one stateless forward. An ONNX warning is failure of the export gate, not training failure or a parity pass.
- [ ] Run new tests and unchanged VIC telemetry tests. Review zero selections, post-reset transition evidence and deterministic playback. Commit explicit files.

## Task 6: Qualification-to-selection initialization

**Files:** extend `rl/impact_speed_policy.py`, `tests/test_impact_speed_policy.py`; integrate explicit loading path in the experimental runner dispatch.

**Interface:** `initialize_selector_for_autonomy(actor: ImpactSpeedModel, *, initial_std: float=1.0) -> None`. It resets only the last output row/bias and selector std coordinate. A newly constructed runner/optimizer loads qualified actor/critic/normalizers without optimizer state; ordinary resume remains unchanged. The phase change must be recorded in checkpoint metadata.

- [ ] Snapshot every tensor, initialize, and test first12 mean outputs and critic/normalizers are preserved for identical observations. Assert last output row/bias zero and selector std1. Pending input columns must already be zero at end of qualification; fail if that invariant does not hold rather than overwriting learned columns silently.
- [ ] Run tests red, implement using the model's actual final/first `nn.Linear` modules, selected by inspected layer type and validated dimensions:

```python
with torch.no_grad():
    final.weight[12].zero_()
    final.bias[12].zero_()
    actor.distribution.std_param[12].fill_(initial_std)
```

Reject unsupported distribution parametrizations instead of assuming a `std_param` exists. Use a newly constructed optimizer; no manual partial Adam-state surgery.

- [ ] Test no old optimizer moments leak, normalizer physical statistics survive, pending0→1 remains in fixed scale, metadata mode changes explicitly, and the first autonomous selector participates in log-probability/gradient. Record that midpoint-mean/random exploration is initialization, not an already learned speed choice.
- [ ] Run the policy/contract tests, review no second policy exists, and commit.

## Task 7: Bounded evaluation and experimental smoke

**Files:** create `scripts/evaluate_impact_speed.py`, `tests/test_impact_speed_evaluation.py`; use existing numerical/reset helpers only where their contracts fit. Do not invoke historical evaluators that force impulse CaT log-only or pin old checkpoint hashes.

**Interfaces:** CLI requires task/checkpoint, explicit two-value `--speed-bounds`, `--speed-cost`, `--seed`, `--num-envs`, and `--mode external|policy|intervention|smoke`. `--goals` is required for external/intervention modes and must lie inside bounds. `--output` names a new result directory; reject overwrite. Smoke uses configured task with64 transitions and no learning. The evaluator uses the same action wrapper and contract validation as training/play.

- [ ] Write fixtures for paired misses, mismatched reset hashes, finalized data followed by reset, no selections, an ignored goal, and genuinely increasing achieved speed. The ignored-goal fixture must not pass merely because one goal has more hits.

```python
def test_ignored_goal_does_not_pass_responsiveness():
    low = {"hit": [True]*4, "speed": [0.5]*4}
    high = {"hit": [True]*4, "speed": [0.5]*4}
    report = compare_goal_responses(low, high, seed=7)
    assert report["paired_speed_difference"] == 0
    assert report["responsive"] is False
```

- [ ] Run red; implement `compare_goal_responses(low: dict, high: dict, seed: int) -> dict` with identical episode IDs and paired bootstrap on common-valid pairs, plus unconditional hit/availability counts and bin-wise misses. Never interpret the common-hit result without coverage. Preserve paired ordering and use deterministic RNG seeded in the report.
- [ ] In intervention mode, run identical resets and identical first self-selected action, then override only the latched goal before the second observation/decision; recompute actor and critic observations. Reject intervention after contact has started. Let later gains respond, and report their change; do not freeze them. Run low/middle/high interventions independently from the same state/RNG evidence.
- [ ] Emit per-episode JSONL/compact summary with goal, measured onset speed, hit/productive/finalized status, first-contact time, actual delivered impulse, tracking RMS/max in mm, joint-target error, per-joint impulse, CaT statistics, clipping and gains. Capture terminal outcomes before resets using Task3 extras. Include discounted cost and timeout count; do not call a late miss penalty a hard deadline.
- [ ] Smoke checks finite13D outputs, pending lifecycle, safety configuration unchanged, one-shot event accounting,42D observations, checkpoint contracts and export parity where available. Only simulation-capable runtime can pass the native smoke; fake-env tests alone do not.
- [ ] Keep rendering through existing play/reference-line support. During later approved qualification produce red-reference videos and white-background trace plots; no new viewer or general plotting framework. Commit evaluator/tests after review.

## Task 8: Regression, evidence freeze and handoff

**Files:** update experimental usage in `docs/README.md` and add a dated result record only for checks actually run. No GPU launch script in this task.

- [x] Run the new suite and unchanged regression set (115 new + 409 legacy/safety tests; exact commands/results in `docs/results/2026-09-21_z1_impact_speed_plumbing.md`):

```sh
python -m pytest tests/test_impact_speed.py tests/test_impact_speed_policy.py tests/test_impact_speed_contract.py tests/test_impact_speed_evaluation.py -q
python -m pytest tests/test_joint_position_config.py tests/test_joint_trackability.py tests/test_smoke_joint_position_fixed.py tests/test_vic_rollout_telemetry.py tests/test_first_strike_event.py tests/test_cat_ppo_gae.py tests/test_cat_storage.py tests/test_variable_impedance.py tests/test_quintic_reference.py -q
python -m pytest tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py -q
python docs/research/reward-design/validate_rewards.py
python docs/research/reward-design/verify_contact_sensor.py
python docs/research/reward-design/verify_reward_setup.py
```

- [x] Confirm validation phases A–M pass. These scripts validate existing baseline contracts; also run the new experimental smoke with an explicitly recorded arithmetic smoke range/cost, clearly **not** a physical reachability qualification. Record commands, runtime, failures and export status. Never silently skip unavailable GPU/native simulation gates. Dated evidence records all three baseline gates, native CPU smoke, PT/ONNX ReferenceEvaluator export and absent onnxruntime; GPU qualification was not attempted.
- [ ] Have fresh reviewers inspect the whole branch for semantic drift, lost guards and excess scope. Resolve findings with targeted tests. Update plan checkboxes only from evidence.
- [ ] Freeze a separate proposed qualification run record containing actual baseline/checkpoint hash, reset library hash, candidate physical range and derivation, coefficient sweep in event-return units, active safety caps/dose, seeds, budget and go/no-go criteria from the approved spec. Do not fill this record with invented values. Its preparation is the next task after plumbing, not a reason to default silently.
- [x] Stop before GPU qualification for review of that run record. Proposed later sequence remains E1 externally sampled goals → E2 policy-selected goals → causal interventions → three curves → matched-seed comparison. Failure of responsiveness stops progression; no automatic planner switch, timing constraint, extra dense reward or larger training budget. The separate 2026-09-21 proposal is explicitly NO-GO with unresolved fields; this checkbox records the stop, not launch approval.
- [ ] Final handoff includes implemented commit, tests actually passed, remaining blockers and the smallest next experiment. No claim of optimal impedance, exact velocity control, three-curve generalization or hardware safety follows from plumbing tests.

## Plan self-review and user handoff

The tasks cover the approved design's action/state, PPO, objective, compatibility, transfer and evaluation requirements. Physical range/weight selection and training remain deliberately evidence-gated, not unspecified implementation defaults. New names/interfaces above are the contract across subagents; any required change is reconciled in this plan before adjacent tasks implement it.

Execution method is already subagent-driven. User reviews this written plan before implementation, as required by the writing-plans skill. The next approval is for the first software tasks, not blanket permission for every later training or architecture change.

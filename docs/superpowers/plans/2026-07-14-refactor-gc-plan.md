# Refactor + garbage-collection plan (2026-07-14)

**Source:** five-agent deep-dive audit @ `528212b` (launch-readiness cross-check, machinery-semantics
cross-check, `src/` duplication, scripts/gates duplication, tests hygiene). Every finding below was
returned with file:line evidence; the load-bearing ones were independently re-verified.

**Cross-check verdict first (context for sequencing):** the impulse machinery has **no code bugs at
blocker/major level** — ring/reset interplay, in-step auto-reset ordering, debounce boundaries, gate
mirrors, R2 preview purity all verified clean. The one BLOCKER is operational, not code: the
**zero-Λ CUDA anomaly** (T4 smoke: Λ ≡ 0, delivered ≡ 0 on substep-rate reads; `528212b` shipped
`diag_cuda_substep_probe.py` and its message forbids the pair run until the probe verdict is in).

**Prime directive:** the gates are sacred — pytest (312), `validate_rewards` A–M, the C0 derive
gate, playback. Every phase below ends with all of them green, one commit per phase. **Nothing here
touches numerics**: the window-sum recompute-vs-incremental choice, the three deliberately different
EMA policies, reward weights/formulas, and the `_i`-clock reset semantics are explicit non-goals.

**Sequencing rule:** Phases R1+ start only **after the Lightning pair pilot is launched and its
pins are immutable** (refactoring under a staged experiment invites drift between what was audited
and what trains). Phase 0 is the exception — it is the launch path itself.

---

## Phase 0 — pre-launch (operational, not refactoring; do NOW, before Lightning)

| # | Item | Evidence |
|---|---|---|
| 0.1 | Run `diag_cuda_substep_probe.py --device cuda:0` (+ `--device cpu` control) on the GPU box; fix the root cause; re-probe. **Hard gate** — if substep reads are dead on CUDA, the pilot trains a different effective reward (delivered_impulse ≡ 0) and measures nothing. | `528212b` message; probe verdict logic `scripts/diag_cuda_substep_probe.py:147-161` |
| 0.2 | Auto-eval device: `eval_impulse.sh:56` defaults `DEV=cuda:0` — the anomaly surface. Pass `DEV=cpu` through `lightning_pair.sh` (env inheritance works) or default it after 0.1 resolves. | launch audit F1.2 |
| 0.3 | Decide `substep_impulse_rows` for the pair: per-substep Python contact loop + unconditional host sync (`contact_row_impulse.py:180,206-219`), untested at 4096 envs, diagnostic-only → recommend `--env.metrics.substep-impulse-rows.params.enabled False` for the paid run, or profile in the first minutes. | launch audit F7; src audit §3-1 |
| 0.4 | Eval CSV out of `/tmp`: `eval_impulse.py:243` + `lightning_pair.sh:54-56` default the thesis-bound summary.csv to `/tmp` — Lightning studio restarts wipe it. Set `OUT` to a durable path (e.g. `logs/eval/`). | scripts audit §4 |
| 0.5 | Pin hygiene on the box: soft-cat pin is now **528212b** (memory's bfe3a30 is stale); sibling repo `safe_impact_manipulation` hammer-z1 @ **0b027cc** must be a sibling dir (repo-relative asset resolution, `nail_block.py:15-19`). | launch audit F1.4/F1.5 |

Everything else on the launch path verified green: task registration, every tyro flag (parse-replicated),
checkpoint naming vs rsl_rl, log-only invariant (env_cfgs default + hook short-circuit + explicit pin),
Track twin wires the current (fixed) prior with anneal completing at iter 250, IMP_J_LIMIT/i_ref values,
eval FIELDNAMES/schema guard.

## Phase R0 — correctness hardening (small, high-value; can precede or follow launch)

| # | Item | Risk |
|---|---|---|
| R0.1 | **Curriculum vs `_NEG_TERMS` guard gap**: the negative-weight (1−δ)-discount guard runs once, lazily (`hook.py:121-147`); `vel_penalty`'s `vel_excess` starts at weight 0.0 and ramps negative at step 1500 (`env_cfgs.py:139-153`). Combining `vel_penalty`+`cat_soft` would silently re-open the documented penalty-evasion exploit. Fix: factory assert that they are mutually exclusive (cheapest), or re-check the guard on curriculum mutation. | LOW |
| R0.2 | **Placeholder-cap guard covers only scalar `imp_limit`** (`hook.py:67-79`, `isinstance(il, (int,float))`): a per-joint `[0.1]*6` with `imp_max_p>0` would enforce ~23-69× too tight silently. Extend the guard to list/tensor forms. | LOW |
| R0.3 | **Test-stub loud-failure helper (Layer B)**: a ~25-line `stub(cls, **attrs)` in `tests/helpers.py` that AST-extracts `self.X = …` targets from `cls.__init__` and raises on missing attrs. Retrofit the 14 `object.__new__` sites; this immediately forces `test_cat_soft_hook.py`'s `_hook` to set the 5 `_imp_*` fields it currently omits. Kills the silent-break class that has fired twice (`_window`, `_enabled`). | LOW |
| R0.4 | **Cover `enabled=False`**: one unit test for the `ContactRowImpulseAccumulator` short-circuit (currently untested — a regression would go uncaught exactly when it matters, on the GPU box). | LOW |
| R0.5 | **Shared-mutable `SceneEntityCfg`**: the hook resolves the PASSED cfg in place (`hook.py:88-89`) while the accumulators copy-then-resolve; one shared `vb_robot_cfg` instance feeds 5 consumers. Extract `resolve_fresh(cfg, scene)` and use it in all three places. Latent (identical joint sets today), bites at G1/multi-scene. | LOW |
| R0.6 | **Metric-ordering asserts**: `cat_delta_peak` after `cat_soft`, `imp_peak_*`/`delivered_total` after their accumulators — today guaranteed only by dict-insertion order in one function. 3-line assert in the factory converts comment-contract → guard. | LOW |
| R0.7 | **Test markers**: `test_nail_physics.py` + `test_hammer_physics.py` (4 of the 5 slowest, full env builds) are missing `pytest.mark.integration`; conversely 10 pure unit tests in `test_contact_row_impulse.py` inherit the marker wrongly (split them out). Restores the documented fast path. | LOW |

## Phase R1 — garbage out (deletions/archives; zero behavior change)

**Archive (git mv to `docs/archive/tooling/` or delete; each with its doc-line edit):**
- The **Vega layer**: `scripts/slurm/{setup_vega.sh,sanity.sbatch,train_array.sbatch}` + `scripts/eval_peak_qv.sh` (frozen June protocol, hardcoded dead checkpoints). **Prerequisite:** relocate `train_array.sbatch`'s run-name contract statement — `compare_runs.py:37-49` and `eval_impulse.sh:83-86` parse `<timestamp>_<run>_seed<N>` and cite the sbatch as its source. Banner `docs/VEGA_TRAINING_PLAN.md` as superseded-by-Lightning (its V1/b_strike RESULT sections stay — they are cited evidence). The repo currently has **no written record that the GPU route changed**; add 3 lines to docs/README.md.
- **Zero-consumer set**: `scripts/record_reference_trajectory.py` + its orphan `src/tasks/hammer/data/reference_strike_qtraj.npz` (loaded nowhere; recorded with the pre-117e070 degenerate reference; its docstring's "the tracking reward consumes this" is false), `scripts/diag_grip_choice.py` (gripper-era; the gripper EE no longer exists), `scripts/diag_strike_trace.py` (June site-bug one-off; `OVERSHOOT` default two generations stale), `scripts/_play_sanity.py`, `scripts/_play_checkpoint_sanity.py`, `scripts/retest_video.sh` (all 2026-05-22 audit one-offs, referenced only from archived docs), `scripts/visualize_terrain.py` (vendored upstream, non-thesis robots).
- `docs/research/reward-design/test_single_strike.py`: retired 2026-07-10 but carries **no banner**. Either archive (edit CLAUDE.md + docs/README.md:81 sentences) or add a RETIRED banner to its docstring. Minimum: the banner.
- `scripts/view_pose.py`: delete the `VERTICAL_POSE` mode (stale-rejected 2026-07-06 — vertical-at-nail infeasible at the L6 grasp) or archive the script.
- `scripts/render_reference_path.py`: fold into `render_reference.py --trail` or archive.
- Local cruft: 9 orphan `.pyc` in `scripts/__pycache__/` (already-deleted June scripts).

**Dead code in src/ (small deletions):**
- `from mjlab.envs.mdp import *` in `mdp/__init__.py:1` — unused re-export, whole-namespace shadowing surface (verify with import + collect, then delete).
- Unused `EventTermCfg` import (`env_cfgs.py:7`); unused actuator imports in 3 asset constants files.
- `ContactRowImpulseAccumulator`'s ignored `robot_cfg` param at `env_cfgs.py:253` (drop at call site — a reader tuning it changes nothing).
- `SingleStrikeReference.peek()` — production-dead since R2-F1 (preview replaced it); delete with its test or keep 3 lines as history. Decide once.
- Inverse-fix: `NAIL_SLIDE_JOINT_NAME`/`NAIL_TOP_SITE_NAME` (`nail_block.py:43-44`) have zero consumers while their literals are hardcoded at 15+ sites — USE them, don't delete.

**Test garbage:** 3 vacuous tests in `test_env.py` (tautology/subsumed); fix `test_windup_phase_grows_with_step_count` (name promises a growth law; body checks one midpoint interval); dead `import pytest` (`test_strike_reference.py:13`); `__main__` self-runner in `test_hammer_physics.py:138-144`; relative `sys.path.insert(0,"src")` shims.

## Phase R2 — single source of truth (constants + δ formula)

| # | Item | Payoff |
|---|---|---|
| R2.1 | **δ formula**: 3 implementations (`constraint_manager.py:57-72` canonical; `hook.py:191-206`; `velocity_bound.py:156-162`). Extract `CaT.ema(...)` + `CaT.delta_map(...)` statics; add `CaT.add_precomputed(name, raw, probs)` so the hook stops writing `_cat.probs`/`_cat.raw_constraints` directly. Do NOT unify the EMA *policies* (documented, deliberate divergences). | HIGH — δ is the enforcement primitive; a fix currently must be remembered in 3 places |
| R2.2 | **Arm-joint tuple** (order-sensitive — IMP_J_LIMIT keys on it positionally): 4 definitions in src (`z1_constants.py:213` canonical, `velocity_bound.py:44`, `contact_row_impulse.py:98`, `env_cfgs.py:299` literal) + 6 copies in scripts/gates/tests. Import the canonical everywhere; the existing `TestJointOrderPin` then pins declared-source-order == resolved-order. | HIGH |
| R2.3 | **Window/rearm 25**: named `IMPULSE_WINDOW_SUBSTEPS` in impulse_bound.py, referenced by class defaults AND env_cfgs explicit params (the gate reads `_window` live, so the class defaults are the redundant copy today). | MED |
| R2.4 | **Velocity limit 3.1415**: two canonicals (`velocity_bound.py:41` public, `z1_constants.py:134` private) + 3 hardcoded script sites. Hardware truth → z1_constants owns it; velocity_bound imports. | MED |
| R2.5 | **`TAU_RATED [30,60,30,30,30,30]`** in `derive_impulse_thresholds.py:55` duplicates what z1_constants actuator cfgs encode as effort_limit. Add `Z1_ARM_EFFORT_LIMITS` to z1_constants; both consume it. | MED |
| R2.6 | **Sensor/site names + strike axis + depth-eps + CaT tau/max_p**: `"hammer_nail_contact"` ×7, axis `(0,0,-1)` ×4 in src (+2 scripts), eps 5e-4 ×4, tau/max_p dicts ×3. One `src/tasks/hammer/constants.py` consumed by factories and class defaults. | MED-HIGH collectively |
| R2.7 | **Tests constants**: DT/DEC/HOLD_STEPS/APPROACH_HEIGHT → `tests/helpers.py` (keep `test_configs.py`'s literal pins — deliberate). | LOW-MED |

## Phase R3 — the playback extraction (biggest duplication)

The open-loop reference-strike driver exists at **12 live sites** (2 gates, 5 scripts, 3 test
fixtures + 1 test inline + 1 viewer) with per-site variations (auto_reset handling, hold steps 6/10/14,
stop condition, step indexing, instrumentation hooks) + 17 frozen copies in dated results assets
(excluded — evidence). Extract to `src/tasks/hammer/mdp/playback.py`:

- `resolve_hammer_scene(env) -> HammerScene` (kills the 4-line site-resolution block + `head()`/`nail_top()` closures everywhere);
- `run_reference_playback(env, *, hold_steps=6, stop=..., step_index=..., on_step=..., **ref_kwargs)`
  with `stop ∈ {reset_terminated, episode_reset, never}` and `step_index ∈ {loop, episode_buf}` —
  both switches are load-bearing (multi-episode replay vs single-strike), not cosmetic.

Constraints: the two gate scripts keep their independent *measurement* sums (their independence
argument concerns the measured quantities, not the driver — state this in their docstrings);
`test_eval_impulse_hook.py` keeps `auto_reset=True` (it tests exactly that path). This phase also
dissolves the HOLD_STEPS mirror-comment web (5 "mirrors derive_impulse_thresholds.py" comments) and
most sensor-name string spread. Payoff: the next reference-degeneracy-class bug becomes a one-file fix.

## Phase R4 — structural refactors (src/)

| # | Item | Risk |
|---|---|---|
| R4.1 | **Factory decomposition**: `z1_hammer_env_cfg` is 300 lines / 9 bool kwargs / 8 conditional blocks (the impulse block alone is 115 lines). Split into `_wire_sites`, `_add_velocity_arms` (→ `legacy_velocity_arms.py` — the 5 orphaned A1–A4 ablation flags, kept re-runnable, out of the live path), `_add_impulse_stack`, `_add_cat_soft_hook` (kills the duplicated tau/max_p param dicts ×2), `_apply_play_overrides`. Same public signature; test_configs (70 tests) pins the product. The C5 composition bug happened precisely because two blocks both wrote `cfg.metrics["cat_soft"]`. | LOW |
| R4.2 | **Substep accumulator base class** (`mdp/substep_base.py`): 4 classes hand-copy the decimation clock (`_i % _dec` clear), episode-peak monotone max, contact gate, env-stash + 6 copies of the getattr-guard, and two different reset idioms (~10 sites). Template-method base with declarative episode buffers; keep all existing private attr names (test stubs poke them); keep the exact clear→accumulate→latch→peak order (semantically load-bearing). | LOW-MED |
| R4.3 | **`DepthRatchet` helper** in rewards.py: the depth-advance gate state machine is triplicated (NailDepthDelta / ImpactProgress / DeliveredImpulse; per-term fill values stay). These are the anti-farm guards — validate_rewards C-I + 20 unit tests pin them; mechanical but careful. | MED |
| R4.4 | **Hook params dataclass**: `CatHookParams.from_dict` — `_validate_params` and `__init__` currently duplicate 4 defaults that can silently desync. | LOW |
| R4.5 | **Stale comment fixes**: `hammer_env_cfg.py:10-11` docstring describes a 3-term reward (code has 7+2 — the most dangerous stale text found); `z1_constants.py:227-228` claims a max_dq rail exists (env_cfgs documents none is set); `cat/constraint_manager.py:5-6` + `cat/__init__.py:3` reference a nonexistent "ConstraintManager" class; `velocity_bound.py:22-24` archaeology parenthetical; `rewards.py:31` raw line-number cross-file pin → function name. | none |
| R4.6 | **Allocation hygiene** (with R4.2, base class owns buffers): preallocate + `out=` for `_rolling`/`amax`/`where` in the 500 Hz paths (~9 fresh allocs/substep in SIA, ~12 in SDI); cache `ImpactProgressTerm`'s axis tensor (SDI already does). Numerics bit-identical. | LOW |
| R4.7 | **CRIA vectorization** (`contact_row_qfrc`): cache geom-ids in `__init__`, hoist the unconditional `nacon` host sync, replace the per-contact Python loop with `torch.isin` masks + `index_add_`. **Gate on the Phase-0 profile**: if the metric stays disabled for GPU training, this is LOW priority; if it must run (it is the enforced-Λ validation channel), it is the top perf item. | MED |

## Phase R5 — tests + eval tooling consolidation

- `tests/helpers.py` **Layer A**: construct the 3 accumulators via their real `__init__` with a ~40-line fake env (verified feasible — `ManagerTermBase.__init__` is trivial, `SceneEntityCfg.resolve` needs 3 entity attrs). Migrate one class per commit; delete old factories only after green. Also gains coverage of the `__init__` validation branches (currently zero unit coverage).
- Merge `strike_trace_1env` + `strike_trace_1env_rows_vs_shipped` (identical cfg + identical strike; one drive records both). Keep the 2-env fixture untouched (its loop encodes hard-won subtleties).
- Shared structural asserts for the 10 facts pinned twice in `test_assets.py` vs `test_mjcf_spec.py`.
- `test_nail_physics` / `test_hammer_physics`: share the aimed `drive_toward_nail` helper — two tests still use the blind `[0,0,-1]` action the repo already adjudicated as flaky.
- Collapse the `eval_impulse.py` ↔ `eval_impulse.sh` duplicated ~25-line protocol headers into one canonical block (py side), fix `PY` default (`.venv/bin/python` is Vega layout), and the /tmp default (Phase 0.4 makes it durable; make it permanent here).
- `pytest.ini`: drop forced `-v`; add `filterwarnings` for the 35 identical torch.jit deprecation warnings.
- Do NOT unify the `_rows_*` test assertions with the `_acc` ones — the two classes deliberately have different semantics (pulse vs sliding window); only the scaffolding is shared.

## Explicit non-goals

Window-sum exactness choice (recompute, not incremental); the three EMA policy variants; reward
weights/formulas; `_i`-clock global-substep semantics; `test_configs.py` literal pins; the 17
playback copies inside dated results assets; `src/assets` non-Z1 robot dirs (out-of-thesis freight —
prune only if the inherited velocity/tracking tasks are ever dropped).

## Order + effort

0 (pre-launch, ~an hour on the GPU box) → R0 (half a day) → **launch + bank the pilot** → R1 (half a
day, mostly `git mv` + doc-line edits) → R2 (1 day) → R3 (1 day) → R4 (1-2 days) → R5 (1 day).
Each phase = one commit, full gate battery after each. R2.1/R2.2 (δ + joint order) are the two
highest drift-risk items if the plan is cut short — do those even if nothing else happens.

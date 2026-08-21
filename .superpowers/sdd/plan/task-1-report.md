# Task 1 recovery report — horizontal reference family and registered task pair

Date: 2026-08-21

Branch: `codex/z1-horizontal-trajectory-pilot`

Base/starting revision: `10a9e85c36844be7d261cc61b6bedad3c1e2ac91`

## Recovery and RED evidence

This task resumed an interrupted, uncommitted worktree after a laptop shutdown. The
existing WIP was preserved and audited rather than restarted.

Recovered controller-message evidence, **not directly observed by this recovery
agent**:

- The prior implementer had witnessed RED→GREEN for world-X route geometry and
  zero-amplitude identity, subset-only reset sampling and episode stability, the
  shared cached observation/imitation reader, exactly two registrations, and
  playback geometry/orchestration.
- The last reported state before shutdown was 14/16 passing scientific-gate tests.
  The remaining incompatibility was the historical direct-reference constructor
  provenance guard after `horizontal_detour_m` was added.

Fresh recovery evidence began from a tree in which the provenance compatibility
patch and its regression test were already present. Therefore no fresh RED is
claimed for that change. A fresh isolated run of
`tests/test_direct_reference_scientific_gates.py` passed 17/17. The audited guard:

- accepts exactly the three historical positional defaults plus the new
  keyword-only `horizontal_detour_m` parameter;
- requires the new default to remain exactly `0.0`; and
- deliberately excludes the zero-default extension from the serialized historical
  direct-controller spec, preserving the banked direct-reference digest.

## Implemented Task 1 contract

- `SingleStrikeReference` owns one reset-stable route sign per environment.
- The frozen detour is world-X only:
  `sign * 0.020 * sin(2*pi*phi)^2` for the first half of the route, and zero
  afterward.
- The zero-amplitude path takes the historical direct interpolation branch, with
  bit-identical waypoint and playback targets.
- One reset event resets and resamples only the requested environment IDs.
- Strike-phase observations, reference-error observations, and the imitation
  reader all request the same cached reference instance and matching detour
  parameter.
- The historical VIC-TT registration is unchanged. Exactly two new route tasks are
  registered:
  - `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-Annealed`
  - `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-Persistent`
- Both route tasks retain VIC-TT and the same RL config, use impulse CaT `p=0.2`,
  and use caps `[0.369, 0.246, 0.738, 0.369, 0.246, 0.0164]` N.m.s.
- The annealed arm retains `r_imit=0.1`, `sigma=0.05`, and the historical
  step-6000 endpoint (`250 iterations * 24 steps/iteration`).
- The persistent arm uses `r_imit=0.2`, `sigma=0.05`, and has no imitation
  curriculum. A full config-tree comparison proves the registered pair differs
  only in imitation weight and curriculum.

## Fresh GREEN verification

### Actual MuJoCo R-/R0/R+ playback

Command:

```text
PYTHONPATH=. MPLCONFIGDIR=/tmp/unitree-mpl-cache conda run -n unitree_mjlab python docs/research/reward-design/playback_reference.py
```

Result: exit 0 and `PHASE M CONTROL-RATE FEASIBILITY: PASS`.

| Route | Max nail depth | Contact | Contact speed | Sampled qvel peak | Finite |
|---|---:|---:|---:|---:|---|
| R- | >= 32.6 mm | step 7 (allowed <= 8) | 1.37 m/s | 2.4290 rad/s | true |
| R0 | >= 32.6 mm | step 7 (allowed <= 8) | 1.36 m/s | 2.3787 rad/s | true |
| R+ | >= 32.2 mm | step 7 (allowed <= 8) | 1.36 m/s | 2.3363 rad/s | true |

The script also reported `identical_z_start_endpoint=True`. Its verdict is explicitly
limited to control-rate sampled feasibility; authoritative 500 Hz certification
remains delegated to `evaluation/guideline/qualify_reference.py`.

The first bare nested-script invocation failed before environment construction with
`ModuleNotFoundError: evaluation`; adding the repository root to `PYTHONPATH` fixed
the invocation. This was not a scientific-gate failure.

### Focused and historical regressions

Command:

```text
PYTHONPATH=. MPLCONFIGDIR=/tmp/unitree-mpl-cache conda run -n unitree_mjlab pytest -q \
  tests/test_strike_reference.py \
  tests/test_imitation_reward.py \
  tests/test_configs.py \
  tests/test_joint_position_config.py \
  tests/test_variable_impedance.py \
  tests/test_direct_reference_scientific_gates.py
```

Result: **243 passed, 0 failed** in 7.41 seconds. Warnings were upstream Torch JIT
deprecations, one profiler warning, and a non-writable pytest cache warning in the
host-managed worktree.

### Registration/config probe

The live registry probe reported exactly two task IDs under the frozen horizontal
route prefix. It reported:

- annealed: weight `0.1`, curriculum `('r_imit_anneal',)`, `imp_max_p=0.2`;
- persistent: weight `0.2`, no curriculum, `imp_max_p=0.2`;
- both: the exact six frozen caps and the historical `RslRlOnPolicyRunnerCfg`.

### Static and scope checks

- `python -m py_compile` passed for all 11 modified Python files.
- `git diff --check` passed with no whitespace errors.
- The diff is limited to the route reference/event/readers, two registrations,
  historical provenance compatibility, playback qualification, and focused tests.
- No launcher, Vega operation, push, training, evaluator extension, result banking,
  model artifact, or documentation banking was performed.

## Residual concerns

- Playback is intentionally a sampled 50 Hz feasibility gate, not the later 500 Hz
  evaluator/certification gate.
- The existing slow-press diagnostic still reports that press-exploit pressure
  exists; it is non-gating and not introduced by this route change.

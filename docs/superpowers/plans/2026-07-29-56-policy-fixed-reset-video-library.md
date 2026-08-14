# 56-Policy Fixed-Reset Video Library Implementation Plan

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer values”
> below refers only to the historical registered-task boundary, not a validated Z1
> reaction-impulse or damage limit. The plan body remains frozen; see
> `../../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an apples-to-apples video, montage, and hammer-head trajectory library for all 56 accepted FQ4x8 and FQ3x8 policies from one frozen realized reset.

**Architecture:** Extend the existing single-policy renderer with a small deterministic-reset and trace-output contract, while keeping pure validation/plotting/index logic in one focused analysis module. Build a manifest from the accepted campaign evidence, render one online-inference episode per checkpoint, then fail closed unless all 56 artifacts share the same reset digest and expected arm/seed membership.

**Tech Stack:** Python 3.10, mjlab/MuJoCo, PyTorch, NumPy, imageio/ffmpeg, Matplotlib, pytest.

## Global Constraints

- Fixed impedance only; never call `set_gains`.
- Keep `IMP_J_LIMIT` at manufacturer values and `imp_max_p=0.0`.
- Render one environment, deterministic online policy inference, and the same realized reset for every checkpoint.
- Stop before any automatic reset can contaminate the first episode.
- Use camera 960×720, distance 0.85, elevation −25°, fixed 10 fps playback, and a common rollout horizon.
- Require exactly 56 unique checkpoint identities: FQ4x8 arms F8/F0/D0/FQ-min seeds 8–15 and FQ3x8 arms F8/B8/FQ seeds 16–23.
- Deploy tracked source to Vega only through commit/push/pull; checkpoint and generated-media transfer may use artifact-copy tooling.
- Preserve unrelated dirty/untracked result files and stage named files only.

---

### Task 1: Deterministic renderer and artifact contract

**Files:**
- Create: `evaluation/analysis/fixed_reset_video_library.py`
- Modify: `scripts/render_policy.py`
- Create: `tests/test_fixed_reset_video_library.py`

**Interfaces:**
- Consumes: the fixed-reset envelope at `docs/results/assets/2026-07-29_lambda_feasibility_stage0/lambda_feasibility_stage0_inputs.json`.
- Produces: `load_fixed_reset(path) -> dict`, `validate_inventory(rows) -> None`, `write_trajectory_png(trace, path)`, `validate_policy_artifacts(root, rows) -> dict`, and renderer outputs `policy.mp4`, `montage.png`, `trajectory.png`, `trace.npz`, `metadata.json`.

- [ ] **Step 1: Write failing focused tests**

Add tests that require:

```python
def test_inventory_requires_exact_56_members():
    rows = expected_rows()
    validate_inventory(rows)
    with pytest.raises(ValueError, match="membership"):
        validate_inventory(rows[:-1])


def test_fixed_reset_digest_is_verified(tmp_path):
    payload = canonical_fixed_reset_payload()
    path = tmp_path / "reset.json"
    path.write_text(json.dumps(payload))
    assert load_fixed_reset(path)["reset_state_digest"] == FIXED_DIGEST
    payload["reset_state"]["realized"]["robot_joint_pos"][0] += 0.01
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        load_fixed_reset(path)


def test_episode_boundary_stops_before_second_episode():
    assert first_episode_frame_count([1, 2, 3, 0, 1]) == 4


def test_artifact_validator_rejects_mixed_reset_digest(tmp_path):
    write_complete_fake_library(tmp_path, reset_digest=FIXED_DIGEST)
    corrupt_one_metadata_reset_digest(tmp_path)
    with pytest.raises(ValueError, match="reset"):
        validate_policy_artifacts(tmp_path, expected_rows())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_fixed_reset_video_library.py -q
```

Expected: failures because the fixed-reset library module and renderer contract do not yet exist.

- [ ] **Step 3: Implement the minimal pure helpers**

In `evaluation/analysis/fixed_reset_video_library.py`:

```python
EXPECTED = {
    "fq4x8": {"F8": range(8, 16), "F0": range(8, 16),
              "D0": range(8, 16), "FQ-min": range(8, 16)},
    "fq3x8": {"F8": range(16, 24), "B8": range(16, 24),
              "FQ": range(16, 24)},
}

def validate_inventory(rows):
    identities = {(r["campaign"], r["arm"], int(r["training_seed"])) for r in rows}
    expected = {(c, a, s) for c, arms in EXPECTED.items()
                for a, seeds in arms.items() for s in seeds}
    if identities != expected or len({r["checkpoint_sha256"] for r in rows}) != 56:
        raise ValueError("checkpoint membership or uniqueness mismatch")
```

Use canonical JSON encoding for reset-state digest verification, SHA-256 every output, render a shared-axis x-z/x-y trajectory plot, and validate readable/non-empty MP4/PNG/NPZ/JSON files.

- [ ] **Step 4: Extend `scripts/render_policy.py` minimally**

Add CLI fields for campaign/arm/seed, reset envelope, metadata provenance, fps, and artifact output. After `env.reset()`:

```python
restore_reset_state(base_env, fixed_reset["reset_state"])
base_env.sim.forward()
obs = base_env.observation_manager.compute()
```

Record hammer-head world position and contact each control step, save `trace.npz`, stop immediately when the first episode boundary is detected, and call the pure helper to write the trajectory and metadata. Do not add campaign looping or analysis logic to this renderer.

- [ ] **Step 5: Run focused and relevant regression tests**

Run:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_fixed_reset_video_library.py tests/test_reset_state_replay.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit named files**

```bash
git add evaluation/analysis/fixed_reset_video_library.py \
  scripts/render_policy.py tests/test_fixed_reset_video_library.py
git commit -m "feat(video): add deterministic fixed-reset policy renderer"
```

---

### Task 2: Freeze the 56-checkpoint inventory and render the library

**Files:**
- Create: `evaluation/analysis/render_fixed_reset_video_library.py`
- Modify: `tests/test_fixed_reset_video_library.py`
- Create outputs under: `docs/results/assets/2026-07-29_56_policy_fixed_reset_library/`

**Interfaces:**
- Consumes: accepted FQ4x8/FQ3x8 campaign manifests, the Task 1 renderer, and all 56 accepted `model_499.pt` checkpoints.
- Produces: `checkpoint_inventory.tsv`, its SHA-256 sidecar, and one complete artifact directory per policy.

- [ ] **Step 1: Add RED tests for manifest construction**

Require the builder to reject a wrong hash, duplicate seed, wrong task, missing checkpoint, unexpected arm, and any inherited dirty provenance; require deterministic row ordering and exactly 56 rows.

- [ ] **Step 2: Implement a small campaign driver**

The driver must:

```python
rows = build_inventory(fq4_manifest, fq3_manifest, checkpoint_roots)
validate_inventory(rows)
write_inventory(rows, output_root / "checkpoint_inventory.tsv")
for row in rows:
    run_single_policy_renderer(row, fixed_reset, output_root)
validate_policy_artifacts(output_root, rows)
```

It may resume only when an existing policy directory passes full hash and metadata validation; otherwise it rerenders that policy. It must not rank, filter, or select outcomes.

- [ ] **Step 3: Verify and obtain all checkpoints**

Hash the 32 local FQ4x8 checkpoints against the accepted manifest. Locate or copy the 24 FQ3x8 accepted checkpoints from Vega using artifact transfer, then hash them against the accepted FQ3x8 manifest. Stop on any mismatch.

- [ ] **Step 4: Run one smoke policy from every arm**

Render one seed from each of the seven arm labels and verify:

```bash
ffprobe -v error -show_entries stream=width,height,nb_frames \
  -of default=noprint_wrappers=1 <policy.mp4>
```

Inspect each montage and trajectory PNG, verify one shared reset digest, and confirm no second-episode frames.

- [ ] **Step 5: Render all 56 policies**

Run the driver on the fastest reliable backend, using Vega only if it preserves the same CPU/MuJoCo rendering semantics and clean tracked-source provenance. Copy generated media back to the local result asset directory if rendering remotely.

- [ ] **Step 6: Run the fail-closed library validator**

Require:

```text
policies=56
unique_checkpoint_sha256=56
unique_reset_state_digest=1
missing_artifacts=0
invalid_video=0
second_episode_contamination=0
```

- [ ] **Step 7: Commit source and inventory only**

Commit the driver, tests, and small inventory/sidecar. Do not commit the large media library until its size and repository convention are reviewed.

---

### Task 3: Comparison grids, index, and independent review

**Files:**
- Modify: `evaluation/analysis/fixed_reset_video_library.py`
- Modify: `scripts/render_policy.py`
- Modify: `tests/test_fixed_reset_video_library.py`
- Create: `docs/results/assets/2026-07-29_56_policy_fixed_reset_library/fq4x8_trajectories_grid.png`
- Create: `docs/results/assets/2026-07-29_56_policy_fixed_reset_library/fq3x8_trajectories_grid.png`
- Create: `docs/results/assets/2026-07-29_56_policy_fixed_reset_library/README.md`

**Interfaces:**
- Consumes: the validated 56-policy library from Task 2.
- Produces: two shared-axis trajectory grids and a complete clickable Markdown index.

- [ ] **Step 1: Add RED tests for grid/index completeness**

Tests must assert that all 56 identities appear once in the index, all relative links resolve, grid axes are common across panels, and each arm/seed maps to the correct checkpoint hash.
Tests must also require every `trace.npz` to contain a finite
`reference_polyline_m` array with shape `(3, 3)` and require each trajectory
plot to draw that array as one thin dashed black line labelled
`SingleStrikeReference (observation only)`.

- [ ] **Step 2: Implement grid and index generation**

At reset, obtain the shared `SingleStrikeReference` after it has anchored and
store its exact `phi={0, 0.5, 1}` waypoint vertices in `trace.npz` as
`reference_polyline_m`. Do not substitute a start-to-contact chord or the
physically lagged open-loop playback trace.

Generate the familiar x-z side-view grid style: green start, normalized-time
color, red contact, brown nail axis, policy/seed title, and fixed global x/z
limits. Draw `reference_polyline_m` behind the realized policy trace as a thin
dashed black line. Label it `SingleStrikeReference (observation only)` and
state in the README that none of these 56 arms enabled the separate `r_imit`
tracking reward; the line is neither an optimal path nor a rewarded path.
Generate a corresponding x-y top-view link per policy through
`trajectory.png`; keep campaign grids uncluttered.

- [ ] **Step 3: Verify outputs**

Run focused tests, the artifact validator, `ffprobe` over all MP4s, and image decoding over every PNG. Visually inspect both grids and at least one montage/video/trajectory from every arm.

- [ ] **Step 4: Independent code and artifact review**

Give a fresh reviewer the design, implementation diff, inventory, validator output, and representative artifacts from all seven arms. Resolve every Critical/Important finding and rerun the focused suite.

- [ ] **Step 5: Bank the result**

Update the library README with honest scope: fixed-reset qualitative comparison only; no reward/impulse causality and no best-episode claim. Record output hashes and exact commands. Stage only named small source/docs files; handle large media according to the repository’s existing result-asset convention.

# T0.5 — Replace the primitive hammer with a real claw-hammer mesh

**Date:** 2026-06-15 (corrected 2026-06-17) · **Branch:** `hammer-z1` · **Status:** **INTEGRATED, striking & RECALIBRATED** (both XMLs, grasp #10); arm gravcomp + sim-audit fixes done; Q1/I_ref re-measured and `NAIL_SUCCESS_THRESHOLD` re-pinned **0.030 → 0.027** (2026-06-17). Ready for Vega V1.

> **Integration result (2026-06-15, corrected 2026-06-17):** real claw hammer mounted in both `z1_hammer_robot.xml` (training) and `z1_mocap_hammer.xml` (viewer). Final mount is **grasp #10**: hammer geoms at quat `0.7071 0 0 0.7071`, pos `0 0.006 0` (an 80 mm grip slide toward the head), flat **face down**, forked **claw up**. Piece identity: the single solid piece `decomp_4` = c4 = the striking **FACE** (geom `hammer_head_0`); the two small pieces c1/c2 = the forked **claw**. Contact sensor → `hammer_head_.*` (face + neck only); `hammer_head_site` sits at the c4 face centroid.
>
> **Correction to the original integration:** the first mount did NOT actually strike — the control site was ~12 cm off the mesh face, so the open-loop reference played to completion with the nail stuck at 0 mm. Root-caused and fixed by re-siting to the face centroid (see the `hammer-site-off-face` memory). The old "contact step 15 → **29.6 mm**" figure was the **pre-`gravcomp` gravity-creep artifact** (the nail self-sank ~9.6 mm/s under its own weight with no real contact), not a true strike. **Verified now (2026-06-17):** `verify_contact_sensor` OK (first contact on the FACE), `validate_rewards` all phases incl. the real strike, 155 `pytest` pass, collision faithful (0.98 mm align, face leads 42 mm, ncon=0 at rest). Head mass 0.5 kg. **Recalibration DONE (2026-06-17):** Q1 single-strike best = **28.3 mm** (shaped reference @ approach 0.06 m), `I_ref ≈ 0.32–0.34 N·s`, contact @ step 7, slow press succeeds in ~89 steps. Best strike < old 0.030 threshold → re-pinned `NAIL_SUCCESS_THRESHOLD` **0.030 → 0.027** (single-strike-reachable, ~1.3 mm margin; the reward is anchored to a single-strike reference). `playback_reference.py` PHASE M GATE now PASS; full gate green. Numbers + rationale in `OPEN_QUESTIONS.md` Q1.
**Why:** Prof. Khadiv asked for a realistic hammer (the current model is a box-on-a-capsule).
**Decisions (user, 2026-06-15):** full realistic **mesh for BOTH visual and collision**; **keep the hammer rigidly mounted** (no move toward grasped/dexterous — SimToolReal's *architecture* is out of scope; only its *asset* is reused).

## Asset
- Source: **SimToolReal** (Lum et al., arXiv:2602.16863), repo `tylerlum/simtoolreal`, **MIT license** — reused with attribution (`assets/meshes/claw_hammer/ATTRIBUTION.txt`).
- Files (downloaded to `safe_impact_manipulation/hammer_z1_env/assets/meshes/claw_hammer/`):
  `claw_hammer.obj` (visual) + `decomp_0..4.obj` (5-piece convex collision decomposition — solves MuJoCo's non-convex-collision problem for free).
- Measured geometry (metres, scale 1.0): extents **0.195 (X) × 0.086 (Y) × 0.028 (Z)**. Handle along **+X** (X∈[−0.052, 0.104], thin), **head at the +X end** (X∈[0.104, 0.143], Y-span 0.086 = face↔claw, full Z thickness). Face/claw are the two Y-extremes.
- Their mass (0.048 kg) is a manipulation-toy weight → **override** with a realistic compact claw hammer: head ≈ 0.4 kg, handle ≈ 0.1 kg, **total ≈ 0.5 kg** (vs current 0.33 kg, head 0.25 kg). Tunable.

## Integration steps
1. **Both robot XMLs** (keep synced, like the scene files): `z1_hammer_robot.xml` (loaded by mjlab for training, via `z1_constants.py`) AND `z1_mocap_hammer.xml` (viewer). Replace the `hammer` body's two primitive geoms (`hammer_handle` capsule, `hammer_head` box) with:
   - `<asset>`: `<mesh name="claw_hammer_vis" file="claw_hammer/claw_hammer.obj"/>` + 5 `<mesh>` for the decomp pieces.
   - 1 **visual** mesh geom (group 2, `contype=0 conaffinity=0`).
   - 5 **collision** mesh geoms (the decomp pieces), one of which is the head/striking piece.
2. **Orientation**: rotate the mesh body so mesh-+X (handle→head) aligns with the old local +Y (head at the far end from the wrist) and the striking **face points local −Z** (the strike direction the policy drives). Verify by **offscreen render** (`scripts/render_reference.py`), correct the quaternion, repeat. The face-vs-claw Y sign is resolved visually (flip 180° about the handle axis if the claw ends up down).
3. **Re-site `hammer_head_site`** to the centre of the striking face. LOAD-BEARING — it is the DiffIK frame, the `head_pos/head_vel` observations, and the T1 `SingleStrikeReference` target. Keep `ee_center_site` as-is.
4. **Re-point the contact sensor** (`config/z1/env_cfgs.py`): `ContactMatch(pattern="hammer_head")` → the head collision geom's name (e.g. the decomp piece covering the face, renamed `hammer_head` for stability, or an explicit pattern). `verify_contact_sensor.py` must still resolve ≥1 primary.
5. **Mass/inertia**: assign mass to the head collision geom(s) (≈0.4 kg) and a light handle (≈0.1 kg); let MuJoCo compute inertia from the meshes, or set explicit `<inertial>`. Head-heavy is the point.

## Recalibration (the ripple — heavier head delivers more impact)
Re-run after integration, in order:
- `verify_contact_sensor.py` (new geom pattern resolves) + `verify_reward_setup.py`.
- `test_single_strike.py` (Q1) — re-measure single-strike depth with the new mass; **re-pin `NAIL_SUCCESS_THRESHOLD`** if needed (invariant: press-stall < threshold ≤ best clean strike).
- `playback_reference.py` (Phase M) — re-measure `I_ref`, contact step, and the press-vs-strike gap; the reference apex/overshoot may need retuning for the new head geometry.
- `validate_rewards.py` (10 phases) + full `pytest`.
- `scripts/render_reference.py` — visual confirmation the real hammer strikes the nail with its face.
- Update `OPEN_QUESTIONS.md` Q1 and `hammering` live-weights/threshold notes with the new numbers.

## Watch-outs
- Mesh-vs-cylinder nail contact is noisier than box-vs-cylinder; keep contacts inelastic (Acosta/SimBenchmark finding) and re-check the impact window.
- The hammer head mass feeds directly into `m_eff·v` / `I_ref` — every downstream impact/impulse number shifts; nothing downstream of T0.5 should quote pre-swap figures.
- Nail `gravcomp` fix (2026-06-15) stands; unaffected by the hammer swap.

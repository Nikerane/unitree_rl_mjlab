# Nail-head precision curriculum (shrinking target) — idea capture

**Status:** brainstorm (2026-06-23), pre-spec. Feasibility probed and confirmed; full design not yet finalized.

## The idea

Start training with the current **oversized** nail head (Ø24 mm — easy to hit) and **shrink it over training** so the policy progressively learns to strike precisely, **ending the curriculum at the real nail's head diameter**. This unifies "switch to a realish nail" with "learn precision": the curriculum *endpoint* is the real nail, so the final policy is precise enough for sim-to-real, and there's a clean "precision emerges from curriculum" story for the thesis. It also fixes the cold-start problem a tiny target would otherwise cause (near-zero contact signal early → nothing to learn).

## What "precision" actually means here (reframed)

The hammer striking face (~Ø25–30 mm) is comparable to / larger than the nail head, so a smaller head is still *hittable by overlap* — shrinking it does **not** gate "did it touch at all." What it tightens is **central, square hits**: as the head shrinks, only a well-centered, axial strike delivers full impulse and drives depth; a glancing/edge hit loses impulse. That is exactly what the existing delivered-impulse + `nail_depth_delta` reward already pays for, so the shrinking head smoothly turns up that pressure. Frame the curriculum as **"learn to hit it squarely,"** reinforced by the existing reward — not "hit a smaller dot."

## Feasibility — CONFIRMED for primitive geoms (probed 2026-06-23)

`mujoco_warp` honors a **runtime `geom_size` change** for collision. Probe: a sphere positioned so contact depends purely on its radius went `ncon` **1 (r=0.10) → 0 (r=0.05) → 1 (r=0.10)** by mutating `wm.geom_size` on-device (`wm.geom_size.assign(...)`) and re-running `forward`. `geom_size` is shape **`(nworld, ngeom)` of vec3 → PER-ENVIRONMENT**, so we can run a global schedule (curriculum) *or* per-env randomization (DR), or DR around the current curriculum level.

- **Works for PRIMITIVE geoms** (cylinder/sphere/box) — the solver reads size each step. **NOT for mesh collision** (convex hull is precomputed at put_model). → **Keep the nail head a primitive cylinder** for the curriculum; use the realish JRL `nail.stl` (`helene-AV/Hammering_Task/addFiles/`) as a **visual-only** geom.
- **Wiring:** mjlab's `reward_curriculum` only mutates reward *weights*, so this needs a **new curriculum/event term** that reaches the env's warp model and assigns the new head radius. (The plain mjModel `geom_size` won't propagate to the device on its own.)

## Two levers

- **A) Geometry shrink** — *confirmed feasible.* Shrink the `nail_head` cylinder radius over training, ending at the real nail Ø. Physically faithful; sim-to-real precision is explicit.
- **B) Reward-precision** — keep the head; tighten an on-center / square-hit reward tolerance (Gaussian std) over training. Also feasible (pure reward-weight/param curriculum); softer, implicit precision. Can complement A.

## Open design questions (resolve before spec)

- **Schedule:** start Ø24 mm → end at the real nail head Ø (need the real nail's spec, or measure the JRL nail's head). Linear vs staged vs **performance-gated** (advance only when success rate / mean depth clears a bar). Over how many steps / iterations?
- **Curriculum vs DR:** monotonic shrink (precision progression) vs per-episode random size (robustness to nail size). Likely DR *around* the current curriculum level.
- **Geometry coupling:** shrinking the head changes the "head flush at qpos=0.032" geometry and the `nail_top` strike-target site slightly — re-check `NAIL_GOAL_DEPTH` / success geometry per level.
- **Signal at small sizes:** smaller head → sparser contact → noisier impulse/depth signal late in the curriculum. Monitor.
- **Interaction** with the strike precision already required, with the (about-to-be-re-solved) `NEAR_NAIL` reset pose, and with the orientation-robust arm direction ([[orientation-robust-vicon-arm]]).
- **Realish mesh proportions:** the JRL nail is visual-only; reconcile its look with the (oversized→shrinking) collision cylinder.

## Decision recorded

Lever A is feasible, so the geometry-shrink precision curriculum is a real option (not only the reward-tolerance fallback). Next step when resumed: converge schedule + endpoint + wiring, then write a proper spec/plan. Independent of the in-flight L6-fixture work (`z1_hammer_robot.xml`) and its pending `NEAR_NAIL` re-solve.

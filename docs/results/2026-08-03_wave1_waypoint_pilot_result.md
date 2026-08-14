# Wave-1 waypoint-guidance pilot result (2026-08-03)

> **Impulse-threshold provenance correction (2026-08-14):** “Manufacturer cap/limit”
> wording and `[1.64, 3.28, ...]` below record the historical registered-task boundary,
> not a validated Z1 reaction-impulse or damage limit. The result body remains frozen;
> see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

Six policies, three arms, two seeds each, one identical fixed physical reset.

**Backend disclosure — read before any number below.**
Every measurement in this note is a **CPU fixed-reset** run. The trajectory half comes
from `scripts/render_policy.py`; the impulse half from `scripts/diag_impulse_trace.py`.
**CUDA impulse channels are unavailable because the validated 500 Hz recorder is
intentionally CPU-only** (`scripts/diag_impulse_trace.py:1191-1192`, present since the
tool was introduced in `25f06be`). A CUDA attempt was made and rejected by that guard;
that is recorded as a **design rejection, not an infrastructure retry** — the guard was
not worked around. CUDA fixed-reset impulse validation is an explicitly OPEN item.

**Provenance limitation (inferred, not recorded).** Renderer metadata did not record
device. CPU provenance is inferred from execution on the CPU-only local environment and
bit-identical shared physical channels against the hard-guarded CPU diagnostic for all
six policies. A `device` field and a validator requirement are DEFERRED to future
rendering campaigns; Wave 1 was deliberately NOT re-rendered, because regenerating six
deterministic videos would invalidate their hashes to record something already
established by the bridge above.

Both halves are the same checkpoint, same simulator, same machine and the same realized
reset digest, so joining them is physically coherent. This was verified, not assumed —
see "Pilot cross-check".

## Identity

| | |
|---|---|
| Training revision | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |
| Analysis revision | `e40068ace92172978b52c801bf4f950c303d0787` |
| Asset revision | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` |
| Fixed-reset digest | `bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319` |
| `imp_max_p` | 0.0 (asserted from the registered config, log-only) |
| Manufacturer caps (N·m·s) | `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` — unchanged |
| Rollout | `num_envs=1`, `nsteps=80`, auto-reset disabled, one episode |

## Pilot cross-check (C0 seed 3), CPU renderer vs CPU diagnostic

Bit-identical on every same-phase shared channel: pre-integration qvel, post-integration
qvel, per-substep control index, head position (`max|diff| = 0.000e+00`), contact
(0 substeps differ), peak |qvel|, terminal reason.

Two apparent differences, both definitional and resolved:

- `control_step` in the diagnostic is per-control-step (8 entries); the per-substep key
  is `control_step_index`. Comparison error on my side, not divergence.
- Nail depth: the renderer stores the **raw** slide joint position; the diagnostic
  stores `clamped_nail_depth`, bounded to the physical stop `[0, 0.032]`. A hard strike
  elastically overshoots the stop, so raw > clamped. Physical depth agrees at 32.0 mm.
  **Consequence: all previously reported depths above 32 mm were elastic excursion, not
  penetration.** Physical depth is reported below.

## Comparison table

Generator: `build_comparison_table.py` (in this result tree; rerun it to regenerate).
`perp` columns are the production-recorded segment-clamped error;
`lateral excursion` is the |Y| residual off the unclamped guideline AXIS, i.e. sideways
drift, deliberately a different quantity from the segment error.

Full table with hashes: `tables/wave1_six_policy_comparison.csv`
(sha256 `4ab58b67658ef6c8…`, regenerate to re-verify).

### CPU fixed-reset trajectory

Perpendicular error is ONLY the production-recorded `substep_perpendicular_error_m`
(finite entry->nail segment, `project_to_reference` clamps progress to [0,1]). It is
never re-derived. Two windows are reported:

- **pre-contact** — substeps from reset through the first contact-onset sample,
  INCLUSIVE. This is the approach path and is the PRIMARY path-quality result. With no
  contact anywhere (C0/2), the whole rollout is pre-contact by construction.
- **full rollout** — every substep. Larger for the gate-completers because the hammer
  follows through BEYOND the nail endpoint after the strike, and finite-segment
  projection clamps the closest point to that endpoint, so residual travel past the nail
  is scored as perpendicular error.

Path-length ratio is evaluated over the SAME pre-contact window, so it is directly
comparable. Gates completed is a max over the FULL rollout.

| arm/seed | contact substeps | success | control steps | depth (mm, physical) | gates | **pre-contact perp max/RMS (mm)** | full-rollout perp max/RMS (mm) | path ratio (pre-contact) | backward (mm) | lateral (mm) | peak qvel | qvel legal | treatment payout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0/2 | 0 | no | 80 | 0.0 | 0 | **381.40 / 270.41** | 381.40 / 270.41 | — | 841.14 | 284.58 | 5.256 | **no** | 0.0 |
| C0/3 | 6 | yes | 8 | 32.0 | 0 | **52.50 / 31.16** | 52.50 / 29.92 | 1.420 | 0.57 | 23.69 | 4.981 | **no** | 0.0 |
| G/2 | 8 | yes | 7 | 32.0 | 0 | **54.00 / 32.59** | 54.00 / 30.80 | 1.412 | 0.00 | 46.69 | 5.002 | **no** | 0.0 |
| G/3 | 8 | yes | 7 | 32.0 | 6 | **10.88 / 6.00** | 31.03 / 9.61 | 1.039 | 0.00 | 3.52 | 4.959 | **no** | 0.16 |
| P/2 | 6 | yes | 7 | 32.0 | 6 | **11.88 / 7.92** | 18.65 / 8.54 | 1.097 | 0.00 | 1.79 | 4.966 | **no** | 0.16 |
| P/3 | 8 | yes | 7 | 32.0 | 6 | **11.50 / 5.48** | 30.88 / 9.46 | 1.035 | 0.00 | 5.28 | 4.975 | **no** | 0.16 |

### CPU fixed-reset impulse diagnostic

| arm/seed | v_precontact (m/s) | delivered (N·s) | axial peak (N) | Λ_j1 | Λ_j2 | Λ_j5 | max Λ_j/cap | joint | dwell (substeps) |
|---|---|---|---|---|---|---|---|---|---|
| C0/2 | 0.000 | **0.000** | **0.000** | **0.0** | **0.0** | **0.0** | **0.0** | — | 0 |
| C0/3 | 1.835 | 0.3466 | 32.83 | 0.1593 | 0.1720 | 0.1080 | 0.0971 | 1 | 6 |
| G/2 | 1.709 | 0.3647 | 33.03 | 0.1687 | 0.1764 | 0.0946 | 0.1029 | 1 | 8 |
| G/3 | 1.566 | 0.3453 | 26.61 | 0.0712 | 0.1675 | 0.0403 | 0.0511 | 2 | 8 |
| P/2 | 1.325 | 0.3089 | 32.44 | 0.1545 | 0.1519 | 0.1093 | 0.0942 | 1 | 6 |
| P/3 | 1.568 | 0.3418 | 26.93 | 0.0542 | 0.1655 | 0.0344 | 0.0505 | 2 | 8 |

**C0 seed 2 (the non-contact policy) is reported, not omitted: Λ is exactly zero on
every joint, delivered impulse zero, axial force zero** — consistent with a rollout that
never touches the nail.

## Verification status

**Independent verification: REQUEST-CHANGES, now addressed.** An external reviewer
rebuilt the table from raw artifacts only and confirmed identity, hashes, reset digest,
revisions, caps, contact/success/depth/gates/payout, path ratio, backward travel, qvel
legality and every impulse channel, and extended the renderer/diagnostic value-identity
check from 2 policies to all 6. It raised one **Critical** defect, since fixed:

> The perpendicular-error columns measured distance to the INFINITE guideline line, not
> the finite entry->nail SEGMENT that `project_to_reference` implements and that the
> trace already records as `substep_perpendicular_error_m`. This understated the
> gate-completers' error by up to 2.85x (G/3: 10.88 -> 31.03 mm).

The earlier "self-verification: 0 cells disagree" did NOT catch this and was not
evidence against it: both of its code paths were algebraically the same infinite-line
formula, so it tested arithmetic equivalence, not the metric definition. The table now
reads the recorded field directly. The qualitative separation survives the correction;
the previously quoted "<= 11.9 mm" bound did not and has been replaced.

## PROVEN — for these exact six seeds, on this identical fixed reset

- **Five of six policies strike and succeed.** C0 seed 2 never contacts the nail in 80
  control steps; it orbits away and times out. Its own final training iteration also
  logs `contact_seen 0.0000`, so the capture is faithful.
- **Every policy that reached all six gates has a markedly straighter APPROACH.** On the
  primary pre-contact window the three gate-completers (G/3, P/2, P/3) hold perpendicular
  error 10.88-11.88 mm max / 5.48-7.92 mm RMS at path ratio 1.035-1.097. The two
  non-completers that still struck (C0/3, G/2) run 52.50-54.00 mm max / 31.16-32.59 mm
  RMS at ratio 1.412-1.420. The ranges do not overlap on either measure, with a factor
  of ~4.5 between them.
- **Full-rollout error is larger for the completers and this is expected, not a
  contradiction.** After the strike the hammer follows through past the nail; finite-
  segment projection clamps the closest point to the nail endpoint, so travel beyond it
  is scored as perpendicular error (G/3 31.03 mm, P/3 30.88 mm full-rollout vs ~11 mm
  pre-contact). The approach window is the path-quality measure; the full rollout mixes
  approach with follow-through.
- **Guidance did not harm contact or depth.** All five strikers reach the same 32.0 mm
  physical stop. Delivered impulse spans 0.309–0.365 N·s with no systematic penalty to
  the guided arms.
- **Neither treatment guarantees straightness.** G seed 2 earned *zero* gate payout and
  is as curved as ungated C0 seed 3. Straightness tracks *whether the policy learned to
  track the waypoint guideline*, not merely which arm it was trained under.
- **No policy is hardware-legal.** Peak |q̇| is 4.96–5.26 rad/s against the 3.1415 rad/s
  manufacturer limit — every one of the six, including the non-striker. Velocity CaT and
  deterministic velocity termination were disabled by design in this campaign, so nothing
  enforced it.
- **Λ never approaches the caps.** Maximum cap ratio across all six is 0.103 (G/2,
  joint 1). The binding joints are 1 and 2. The impulse constraint is nowhere near active
  at this strike energy.

## OPEN — what this pilot does not establish

- **Two training seeds per arm.** Six policies total. No population-level or causal claim
  about C0 vs G vs P is supported; the G-seed-2 outlier alone shows within-arm variance
  comparable to the between-arm difference.
- **A single fixed reset is not a generalization test.** These are one-initial-condition
  replays, chosen for comparability, not robustness.
- **200 iterations only**, versus 500 in earlier campaigns.
- **No constraint enforcement.** `imp_max_p = 0.0` throughout: Λ is logged, never acted
  on. The qvel illegality above is an observation, not a violated enforcement.
- **CUDA fixed-reset impulse validation is unavailable** and remains open (see the
  backend disclosure). Known CPU/CUDA substep differences are therefore unmeasured for
  the impulse channels.
- **No claim about variable impedance.** VIC is untouched by this pilot.

## Recommendation for the next minimal training wave

The straightness signal is large (≈5× reduction in perpendicular error and a path ratio
near 1.0) but rests on three of four guided seeds. The single most informative next step
is **more seeds, not more reward machinery**: repeat G and P at 4–6 seeds each with C0
as control, unchanged weights, to establish whether the G-seed-2 failure is a tail event
or a real per-arm failure rate. Raising iterations from 200 to 500 would separately test
whether C0 seed 2's collapse is an undertraining artifact.

Do not add reward terms, enable enforcement, or move to VIC on this evidence.

## Artifacts

- Videos and 500 Hz traces: `videos/wave1_{c0,g,p}_seed{2,3}/`
- Impulse diagnostics: `impulse/wave1_{c0,g,p}_seed{2,3}/` (trace.npz, metadata.json with
  `payload_sha256` / `npz_sha256` / `metadata_sha256`, trace.png)
- Grids: `figures/wave1_six_panel_xz.png` (sha256 `38971a62608408d6…`),
  `figures/wave1_six_panel_xy.png` (sha256 `514f1a3970c0b282…`)
- Table: `tables/wave1_six_policy_comparison.csv`
- Checkpoints and manifest: `checkpoints/`, `artifact_manifest.json`

## Freeze record (2026-08-03)

This result is frozen. The evidence root is
`evaluation/results/2026-08-02_wave1_waypoint/`.

**Reproducibility gate.** The published table was not merely re-hashed: it was deleted and
regenerated from `build_comparison_table.py` against the raw `trace.npz` artifacts, and
came back byte-identical to the reviewed copy.

| Artifact | sha256 |
| --- | --- |
| `SHA256SUMS.txt` (inventory root) | `4d1e747353c2f136bd6254c623a635a5656fa7ca388dff584e7045b132a7c626` |
| `SHA256SUMS.meta.json` | `2d9e3a5c5e3233e3e8bef7a22e1d7cb7a6f9d7d43953ddab933b81a8ec5ee1cb` |
| `tables/wave1_six_policy_comparison.csv` | `4ab58b67658ef6c899a21ffbd200d4c568327352c72da0dd1c386437dda86268` |
| `build_comparison_table.py` | `a0a1b7cdfb42a716390ac5b88e1317a048f7827fc5697b878c664bed5ba92c7a` |
| `artifact_manifest.json` | `ea4b4cfeb9491bb0b72e83fa806b233756badb0f4617b180a6cc477ccf91ca89` |
| `artifact_manifest.csv` | `aac7909fd2d3f7a40cf1705c77e2c338986190ad78d4a55175492dd22af65e6c` |
| `figures/wave1_six_panel_xz.png` | `38971a62608408d68825b9b54bf4cd90520999f4fb74b1edd9f4f7ea0121a030` |
| `figures/wave1_six_panel_xy.png` | `514f1a3970c0b282af5edf50956921a2db307c775fc9fd352b67a48dbdf6dad1` |

`SHA256SUMS.txt` covers **190 files** of the complete result tree — every checkpoint,
trace, video, metadata sidecar, training log, qualification and smoke artifact — in
`LC_ALL=C` path order. Verify with `shasum -a 256 -c SHA256SUMS.txt` from the evidence
root. `SHA256SUMS.meta.json` records the exact regeneration command and the three
excluded paths (`.DS_Store`, and the two inventory files themselves).

**Three revisions, deliberately distinct.** The result commit is *not* the revision that
produced the policies, and *not* the revision that produced the analysis:

| Role | Revision |
| --- | --- |
| Wave-1 training (produced the six checkpoints) | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` |
| Analysis / rendering / diagnostics | `e40068ace92172978b52c801bf4f950c303d0787` |
| Asset (`safe_impact_manipulation`) | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` |
| Result freeze (this note + table + figures + inventory) | recorded in the freeze commit |

**What is committed vs. what is preserved.** Only the small, human-readable result
files are tracked: this note, the campaign preregistration, the table generator, the
comparison CSV, the two grids, and the manifests/inventory. Checkpoints (`.pt`), traces
(`.npz`), videos (`.mp4`), full run directories and Slurm logs are **preserved locally
and mirrored to frozen Vega evidence**, never committed. `SHA256SUMS.txt` is what binds
the committed summary to the uncommitted bulk.

**Qualification artifacts left as-is.** `qualification/per_reset.csv` and
`qualification/qualification.json` contain a `corridor_max_m` column emitted by
`qualify_reference.py`. That is a legitimate field name in its own context — a pass/fail
band on the *scripted reference* trajectory — and is not a claim that any corridor or
tube treatment was trained. Wave 1 trained a **waypoint guideline**, not a corridor. The
frozen artifacts are not rewritten to satisfy a text scan.

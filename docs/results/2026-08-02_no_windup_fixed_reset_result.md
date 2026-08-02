# No-wind-up fixed-reset direct-strike qualification

**Date:** 2026-08-02
**Scope:** Unitree Z1 simulation; CPU scripted reference; fixed impedance; log-only impulse machinery (`imp_max_p=0.0`)

## Result

The owner-approved direct, no-wind-up scripted reference qualifies the
fixed-reset C0/C-Gate pre-GPU experiment.  Across deterministic execution
seeds 1000--1015, the single realized reset passed 16/16 times: it crossed all
six production gates, reached accepted contact, drove the nail, stayed within
the 5 mm corridor, remained below the joint-speed rail, and produced finite
actions and traces.  This establishes reference feasibility and live,
consistent measurement plumbing only; it does not establish learned-policy
behavior.

| CPU direct-strike fact | Bound / configuration | Observed result |
| --- | --- | --- |
| Fixed physical reset and execution seeds | reset range `[0.0, 0.0]`; 16 ordered seeds | identical six-joint reset; 16/16 qualified |
| Geometry | six gates; 5.0 mm corridor through accepted onset | 6/6 gates; max 1.3001 mm |
| Contact and task progress | production accepted-contact tracker | accepted/raw contact true; 32.000 mm nail progress |
| Joint velocity | `<= 3.1415 rad/s` | peak 2.5308 rad/s |
| Impulse characterization | unchanged `IMP_J_LIMIT`; log-only `imp_max_p=0.0` | scripted-reference max `Lambda/cap = 0.0484` |

The scripted direct reference is therefore far from binding the unchanged
manufacturer-derived caps.  Its repeated executions are deterministic
repeatability/plumbing samples, **not** an impact-intensity distribution or a
population-percentile estimate.  C2 enforcement is explicitly deferred;
`imp_max_p` remains `0.0`.

## Frozen provenance and evidence

- Code revision: `5d9b14c704c56be7551f9e9e92949da7bf9c6291` (clean).
- Z1 hammer-asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`
  (clean).
- Runtime: mjlab 1.4.0, MuJoCo/MuJoCo-Warp 3.8.1, Torch 2.12.0.
- Frozen caps: `[1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` N m s.
- Source hashes: production config
  `63e5438968aff4ad8b8ea07aeb5e1b4935ff6dd88436680aff7afbb0e99d5dba`;
  scripted reference
  `1ccba6357600879fc07a6779c00290632149c8865eb41ff2af383b997ca198cb`;
  qualification script
  `6f700f5a86c948d8818bf628f019deedb01a82ab4428f257ca5c3456c1fe6ec1`.

| Evidence artifact | SHA-256 |
| --- | --- |
| `per_reset.csv` | `68fc62e6029e307fe56a3ccab1abd561ce9e05015459edf934771d37c0d57f6c` |
| `qualification.json` | `4bad30007182866eb36fe09660af00c07e8f3e68a3bf9e84b8bf0e201519f54b` |
| `reference_xz_xy.png` | `2864664b889df5b8c54733eee141cd07ee36f0be1696e16bff43c963a186cfa6` |

The evidence is in
`evaluation/results/2026-08-02_no_windup_fixed_reset_qualification/`.

## Interpretation and boundaries

The earlier randomized-reset qualification artifact
`evaluation/results/2026-08-01_guideline_qualification/` is historical and
superseded for this fixed-reset C0/C-Gate design.  It is neither deleted nor
invalid for its original randomized-reset question.

Open questions are whether learned C-Gate policies are straighter than C0
without harming success, speed, or impulse; whether any policy can bind the
manufacturer caps meaningfully; and the separate enforcement/VIC work.  A
feasible direct scripted reference is not evidence that learned policies will
be straight, safe, successful, or cap-binding.

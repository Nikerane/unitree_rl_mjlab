# Three-route autonomous-speed pilot — preflight submitted

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-25
- Verification Status: UNVERIFIED (GPU preflight and training pending)
- Version Label: three_route_pilot_v1

## Scope

Fresh learned-impedance policies on `Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed`: external-speed versus policy-selected-speed modes, paired training seed labels 10/11/12. Six total runs, 50 iterations and 1,024 worlds each, at most three concurrent GPUs, two-hour job caps. No changes to safety limits, reward settings or speed bounds [1.5, 2.2] m/s. No checkpoint transfer.

Evaluation: 384 initial worlds per condition, held-out seed 20260925, native and common-window protocols. External goals 1.5/1.85/2.2 m/s; autonomous mode uses its own selected target without an override. Per-route counts and outcomes, exactly-once selection evidence, and actual versus requested speed are required. Reset hashes and route signs determine whether initial-state pairing is justified; seed labels alone do not.

## Local software evidence

- Task2 implementation: `8c10247041c016c110a113011acb8c1308533633`.
- Implementer focused suite: 58 passed, 49 known deprecation warnings, 44.52 s.
- Root contact-recorder/strike-evaluation/qualification/train-routing integration: 31 passed, 42 warnings, 56.35 s.
- Task2 independent spec/quality review approved. Full continuation review found one P2: last-rollout-only telemetry could falsely reject valid earlier one-shot speed selections. Fixed in `4b1bf7e`; final focused verification: 78 passed, 53 suppressed warnings in 74.78 s. Scoped independent re-review is pending.
- An additional full-repository test run was interrupted, not passed: 824 passed and one pre-existing documentation-date assertion failed. The affected focused suite above is the verification scope for this change.
- Local test imports required a command-local PATH excluding the MiKTeX `fc-list` shim; a sampled stack verified that font helper was waiting in its filesystem watcher. No global environment or simulator settings were changed.

## Isolated staging provenance

Remote directory: `/cephhome/eunikhilr/campaigns/z1-three-route-speed-20260925.yctE0R`.

Reconstructed from verified original `source-e3fda7e.tar.gz` SHA256 `03f8fddd6affbf52f7345673fab8c0ebe301e85156eacc87a4ef8d7e79d86e5b`, existing retry patch SHA256 `9883d3f956f7102c9f5160d8cc95607ae4e5095dd1a670caf8c47c2f0cf242ab`, then committed `eafb795..8c10247` patch SHA256 `bc47db7595302dd9af6d15702098140d524623fc0d0e1c0e67607f9ef5ac4905`. The staged hash inventory contains 1,774 files. Existing campaigns are untouched.

Updated with committed `8c10247..4b1bf7e` patch SHA256 `2c1e3eb02bee0db973afc2e5bbfa030eddf2461ce342fd812d46d5957021a722`, after validating the preceding inventory. The refreshed 1,775-file inventory SHA256 is `e91ff9f60c69bc0127e3f101eddec796022f7050be6e97d1bccee1ceb46dbc36`. Remote launcher syntax check passed.

Scoped independent re-review approved `4b1bf7e`, with all findings addressed and no new important breakage. Authorized one-GPU preflight submitted as **45953544**, with an explicit 30-minute cap. Six-run training array has not been submitted; its release requires successful preflight evidence.

## Operational gates still required

1. DONE: independently verified separately owned diagnostic replay `45947897`, completed 0:0 in 8m50s; all 576 first episodes passed artifact-integrity checks. The old policies are not load-compliant; the pilot tests autonomy and route coverage, not an established safety remedy.
2. Resolve final software review and stage/hash its exact approved revision.
3. Submit one bounded GPU preflight with a 30-minute cap; verify both fresh speed modes, strict reload, live mask/latch, and evaluation.
4. Release only the six authorized training slots using `0-5%3`, after successful preflight, with two-hour caps and no requeue.
5. Verify every run's completion record and checkpoint hash before the pairing audit and seed/route-stratified performance analysis. Submission alone is not completion.

These are simulation experiments, not hard-constraint or hardware-safety qualification. Three training seeds are exploratory; report misses, incomplete windows, and safety exceedances rather than filtering them out.

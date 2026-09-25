# Contact replay: completed; compliance remains unresolved

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-25
- Verification Status: execution and artifact integrity verified; descriptive diagnosis available
- Version Label: contact_replay_attempt_v1

Authorized evaluation-only replay; no training or safety-limit changes.

## Successful independently owned retry

The joint-load diagnosis task owns retry **45947897** in `/cephhome/eunikhilr/campaigns/z1-contact-diagnostics-retry-20260925.uarjss`. This task submitted no duplicate. Root independently verified scheduler **COMPLETED, exit 0:0, elapsed 00:08:50, cap 00:30:00** and all six result sets.

- Exact source: verified ce94e3b snapshot plus binary diff ce94e3b..ab0ff7d; runtime change is the Git module setup only. Recorder/evaluator remain the reviewed original version, not the newer three-route evaluator.
- Recorder gate: 4 tests passed, 40 dependency warnings, 174.74 s.
- Each of six runs contains 96 first episodes (32 worlds × three goals), 96 complete common windows, three trace artifacts, and zero learning updates.
- Total: 576 first attempts, all 576 common windows complete, 18 trace artifacts. First-episode selector count is zero throughout, as required for these external-speed checkpoints.
- Large trace artifacts remain on Vega. Completion does not establish safety or autonomous speed selection.

Root's independent read-only CPU validation passed for all **576** first episodes: nonempty contiguous masks; sample count agreement; native-success indices within the observed episode and matching evaluator markers; one accepted onset matching evaluator contact time; exact `max(rolling, control-step peak)` decomposition and exact episode-peak agreement; desired-speed and latched measured-speed agreement; and zero external selector events. This is cross-channel artifact integrity, not independent physical calibration or causal evidence. No policy was rerun during these checks.

## Original failed attempt (preserved)

- Vega job: **45946767**, one A100 allocation, four CPUs, 24 GB RAM, 30-minute wall cap, no requeue.
- Campaign: `/cephhome/eunikhilr/campaigns/z1-contact-diagnostics-20260925.yPFIQH`.
- Exact local source revision: `ce94e3b` (recorder implementation `8699196`).
- Source construction: verified original `source-e3fda7e.tar.gz` SHA256 `03f8fddd6affbf52f7345673fab8c0ebe301e85156eacc87a4ef8d7e79d86e5b`, existing retry patch SHA256 `9883d3f956f7102c9f5160d8cc95607ae4e5095dd1a670caf8c47c2f0cf242ab`, then exact committed overlay `overlay-ce94e3b.tar.gz` SHA256 `7c33cfaa8418aa702a5ccf0fa948d186555b3c09b9320b3f7c15d94d6b8bbeb9`.
- Remote `source.sha256` inventories 1,768 files; the job verifies it and the 32-file external asset manifest before execution.
- External asset revision checked: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Existing checkpoints read-only: `z1-impact-speed-retry-20260924.IGsdQz/runs/run-{0..5}/model_49.pt`; each hash checked against its completion record.
- Six policies × goals 1.5/1.85/2.2 m/s × 32 initial worlds; held-out reset seed 20260924; LEFT task, external goals, common-window protocol.
- Recorder tests precede evaluation. Local assigned regression: 27 passed; independent task review approved source and launcher, with minor coverage/deprecation observations only.
- Outputs: `results/run-N/condition-K/contact-trace.pt`, adjacent metadata, plus episode JSONL and evaluation summaries. Scheduler log: `logs/45946767.out`.

## Observed failure and next gate

Scheduler confirms FAILED, exit 1:0, elapsed 00:01:27. The recorder test gate reported 1 passed / 3 failed, all three failures during imports with GitPython `Bad git executable`. The new wrapper omitted `module load git`, which the previously successful campaign wrapper includes. No policy replay or training occurred. A read-only login-shell check with `module load git` resolved GitPython and reported Git 2.45.1; this verifies the missing dependency remedy at that boundary, not a successful compute-node replay.

The implementation agent committed the one-line setup correction as `4134b3a`; shell syntax verification and independent scoped review passed. This task initially requested retry approval, then deferred submission when the separately authorized diagnosis task took ownership of the successful retry above. Preserve the failed directory and logs. The original failed test log is banked under `docs/results/assets/2026-09-25_contact_replay/45946767-observer-tests.log`.

Offline analysis helper `docs/results/analyze_contact_replay.py` was checked against the local untrained smoke trace (5,120 samples): trace decomposition and evaluator episode peaks agree exactly. Missing native-success evidence is preserved as null rather than a false timing conclusion. This check is software evidence only; the smoke did not contact the nail and is not a policy-performance result.

Independent analysis review confirms one-based crossing arithmetic. Speed, selection count and common-window impulse remain evaluator-provided fields, not independently reconstructed by this helper. Event bounds/nonempty traces require explicit validation before interpreting replay artifacts. Relative timing is on the recorder sample clock; native site state can lag integrated joint state by one 0.25 ms sample, so near-zero offsets must not be interpreted as exact physical or causal latency. Any pooled gain-window statistics must report eligible worlds and observed sample counts. All attempts, including no-contact and no-success cases, remain visible.

The original failed submission alone is not evidence of replay completion; the successful retry above is. Neither establishes safe behavior. Analyse threshold crossings relative to the same-rollout native-success marker; do not subtract separate native/common-window violation counts to infer post-success causes. Existing campaigns remain untouched.

## Completed diagnostic interpretation

The separately owned report is available at `/Users/nikerane/repos/unitree_rl_mjlab/docs/results/2026-09-25_z1_contact_aligned_diagnosis.md`, with reconstruction and per-attempt artifacts beside it. Root inspected this report after independently validating all 576 trace prefixes above.

Each treatment has 288 attempts across three training seeds. Fixed versus learned impedance: any load-cap breach 134/288 versus 133/288; mean axial impulse 0.3771 versus 0.4243 N·s; measured contact speed 1.0812 versus 1.4028 m/s; absolute speed error 0.7701 versus 0.4692 m/s. Full-observed-episode joint-speed breaches are 257/288 versus 288/288. These are a new 32-world-per-condition replay population, not the prior 128-world comparison.

Of 267 load-violating attempts, 216 first breach by their same-rollout native-success check. J2 and J6 account for observed load breaches. J6 has no command-clipping samples in any of the 576 observed episodes, and learned J6 stiffness at its crossings is 0.931–1.104 times nominal, above its 0.8 minimum. Local J6 torque clipping or minimum-stiffness saturation therefore does not explain these failures. This is not proof that lowering stiffness would solve them.

The enforced quantity remains a contact-masked generalized constraint-reaction residual integrated over 200 physics samples with an 80-sample policy-step peak hold. It is not independently calibrated gearbox stress. Missing contact-row decomposition and normalized CaT histories prevent a definitive physical or learning-cause conclusion. The separately owned follow-up shadow diagnostic must not be duplicated here. The authorized three-route pilot addresses speed autonomy and route coverage, not a demonstrated compliance remedy; all caps remain unchanged.

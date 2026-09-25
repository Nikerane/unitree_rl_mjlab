# Contact replay: failed before policy evaluation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-25
- Verification Status: UNVERIFIED (no completed policy replay)
- Version Label: contact_replay_attempt_v1

Authorized evaluation-only replay; no training or safety-limit changes.

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

The implementation agent committed the one-line setup correction as `4134b3a`; shell syntax verification and independent scoped review passed. User approval for one fresh 30-minute retry was requested; no retry has been submitted. Preserve this failed directory and logs. The original failed test log is banked under `docs/results/assets/2026-09-25_contact_replay/45946767-observer-tests.log`.

Offline analysis helper `docs/results/analyze_contact_replay.py` was checked against the local untrained smoke trace (5,120 samples): trace decomposition and evaluator episode peaks agree exactly. Missing native-success evidence is preserved as null rather than a false timing conclusion. This check is software evidence only; the smoke did not contact the nail and is not a policy-performance result.

Independent analysis review confirms one-based crossing arithmetic. Speed, selection count and common-window impulse remain evaluator-provided fields, not independently reconstructed by this helper. Event bounds/nonempty traces require explicit validation before interpreting replay artifacts. Relative timing is on the recorder sample clock; native site state can lag integrated joint state by one 0.25 ms sample, so near-zero offsets must not be interpreted as exact physical or causal latency. Any pooled gain-window statistics must report eligible worlds and observed sample counts. All attempts, including no-contact and no-success cases, remain visible.

This submission is not evidence of completed replay or safe behavior. Analyse threshold crossings relative to the same-rollout native-success marker; do not subtract separate native/common-window violation counts to infer post-success causes. Existing campaigns remain untouched.

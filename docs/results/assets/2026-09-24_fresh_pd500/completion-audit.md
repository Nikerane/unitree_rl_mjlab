# Pilot completion audit

## Verified requirements

- Fresh bounded training: `pilot.py` creates a new output directory, does not
  load a training checkpoint, sets seed 2 and 250 iterations; `train250.log`
  contains exactly 250 learning-iteration entries.
- Checkpoint contract: `train250/completion.json` records strict reload and
  iteration 249. Local checkpoint SHA256 independently matches
  `0de912ae08faf988454c3cc561da62abc0df4bc744940383c870c97466377815`.
- Selected gains and physical timing: saved contract contains canonical
  Kp=[256,384,384,256,192,128], Kd=[6.4]*6, 2ms PD, 0.25ms physics and
  20ms policy. `pilot.py` preflight checks measured actuator torque against
  held torque at every substep, including reset.
- Limits unchanged: contract preserves 3.1415rad/s, all-six substep detection,
  and [30,60,30,30,30,30]Nm effort caps. Sampled-controller implementation
  explicitly validates actuator mapping/caps and clips requested torque.
- Existing task/reward/speed phase: pilot selects the existing
  Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed task, external speed mode,
  bounds [1.5,2.2], cost 0.8493676, and impulse dose 0.2. Controller patch
  does not replace reward functions; 50ms windows are preserved as 200 substeps.
- Required gates: reward-gates.log says ALL PHASES PASSED (A–M), contact and
  setup logs confirm their checks; tests.log records 207 passed. Separate
  smoke completion contains successful checkpoint reload.
- Evaluation: six saved condition files cover held-out seeds 2026092401/02
  and goals 1.5/1.85/2.2, 128 worlds each. Evaluator retains terminal substep
  peaks and masks subsequent reset episodes; summary has no smoke step cap.
  All 768 first episodes productive, no speed-limit violations or episode
  censoring. Maximum J2=3.01409197, J6=1.95007277rad/s.
- Report and next decision: parent results report records results, proposes
  two unchanged independent training seeds, and explicitly withholds
  joint-load/hardware safety and autonomous speed/learned-impedance claims.
  No extra training launched.

## Supplementary verification closed with caveats

Broad regression job45858771 finished FAILED (3458 passed,287 failed,4 skipped).
All9 sampled-controller tests passed. Re-running286 failures with system Git,
without inherited module function/BASH_ENV, yielded285 passed and1 failed.
The remaining PNG test assumes filesystem enumeration order; the other original
failure is a strict float32 equality check in unchanged native variable-impedance
code. A standalone calculation reproduces its exact1.52587890625e-5 discrepancy.
Neither controls this pilot's applied fixed gains or saved evaluation. No tests
were edited or safety limits relaxed. Full log is banked alongside this audit.
This does not assert a full-suite pass; these two unrelated test issues remain.

All bounded pilot requirements above are satisfied. Next decision is two
independent training seeds with identical settings, not automatically launched.
Joint-impact-load/hardware safety and learned impedance remain outside the
claims established by this experiment.

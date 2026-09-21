# Task 6 implementation report — qualification-to-selection initialization

## Implemented

- Added `initialize_selector_for_autonomy`: verifies the qualified actor's pending input column is still zero, validates the actual first/final linear dimensions and learnable scalar Gaussian standard deviation, then zeros only selector output row/bias and sets its std to 1. The critic pending column is also checked by the runner transfer path; learned columns are never overwritten.
- Added `HammerOnPolicyRunner.load_qualification_checkpoint` as a distinct external→policy transfer. It validates the full experimental contract, loads actor/critic/normalizers into a newly constructed runner, excludes optimizer and iteration restoration, initializes the selector, and records `impact_speed_phase_transition` in PT checkpoint infos. Ordinary `load` retains strict same-mode validation and preserves the transition record on subsequent resume/save. A fresh non-transfer save removes stale transition info supplied by callers.
- Added the optional `scripts/train.py --qualification-checkpoint` route, guarded against ordinary resume, non-experimental task IDs/signatures, external destination mode, and missing files. Default train/resume behavior is unchanged.
- Strengthened the selector KL-direction test with unequal old/new standard deviations and an asymmetric analytic expectation.
- Added a three-transition CPU CatPPO regression showing the delayed terminal reward flows through the first selector-active mask into a selector-row update. The paired test-fixture control disables that first mask and observes no selector-row update; it does not alter production CaT/PPO code.

The zero-mean selector corresponds to the interval midpoint; unit-std sampling provides initial exploration. This is an **initialization**, not evidence that speed selection has been learned.

## TDD evidence

- **RED — policy:** `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q tests/test_impact_speed_policy.py -k 'selector_transfer or pending_selector_is_active'` failed on the absent initializer. The KL fixture initially indexed the full std row, rather than the selector coordinate; correcting that test fixture made its analytic expectation pass.
- **RED — runner:** `... -m pytest -q tests/test_impact_speed_contract.py -k external_qualification_transfer` failed because `HammerOnPolicyRunner` had no `load_qualification_checkpoint`.
- **RED — CLI:** `... -m pytest -q tests/test_impact_speed_train_routing.py` failed because `scripts.train` had no qualification checkpoint route.
- **RED — provenance:** `... -m pytest -q tests/test_impact_speed_contract.py -k empty_real_checkpoint_still_contains_contract` failed because a caller-supplied stale phase transition was retained.
- **GREEN — initial focused suite:** `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q tests/test_impact_speed_policy.py tests/test_impact_speed_contract.py tests/test_impact_speed_train_routing.py tests/test_impact_speed_wrapper.py -W ignore::DeprecationWarning` → **65 passed in 9.39 s** before the delayed-credit regression was added. The warning filter only suppresses upstream TorchScript/ONNX deprecation warnings.
- **GREEN — delayed-credit regression:** `... -m pytest -q tests/test_impact_speed_policy.py -k delayed_cat_return -W ignore::DeprecationWarning` → **1 passed**. First-step selector mask was `[1,1]` followed by two zero-mask steps, the initial return matched the analytic `−0.5(0.99·0.95)²(0.8)(0.7)`, and the selector row changed only in the selected case. The `pending=0` control kept the selector row unchanged.
- **GREEN — final focused suite after this addition:** same four-file command above → **66 passed in 10.15 s**.
- CLI discovery: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python scripts/train.py Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed --help` exited 0 and displayed `--qualification-checkpoint`.

The real CPU transfer test starts with populated source Adam moments, nontrivial actor/critic physical normalizer statistics and modified selector parameters. It checks control/value preservation, empty destination optimizer state, raw fixed-scale pending, selector gradient, explicit phase metadata, normal resume of the autonomous checkpoint, and rejection of learned actor or critic pending columns.

## Self-review and limits

- Exactly one `ImpactSpeedModel` type remains for the actor/critic; no second policy was added. Transfer touches only the selector row/bias/std after weight loading, and the existing reward/actuator paths are untouched.
- No GPU training, qualification campaign, physical interval calibration, or end-to-end autonomous learning claim was performed. The `(0.2, 0.8)` interval and `0.4` cost in tests are arithmetic software fixtures only. The task-specific focused tests were run, not the full repository suite, as requested for this packet.

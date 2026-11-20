# Contact-relative termination on z1-u22-impulse

The frozen tracking baseline remains at tag `u22-tracking-420ms-v1` on
`z1-reward-simplify`. This branch restores the existing B first-contact timer
for new training; it does not introduce a second timer implementation.

## Rule

- Record the first hammer–nail contact using the existing first-strike tracker.
- End at the first 20 ms policy boundary at least 200 ms after that contact.
- Keep the existing 5 s episode timeout for misses and very late contacts.
- Do not terminate merely because the recorded reference ends.

The reused function is `horizon_window_complete(window_s=.2)` in
`src/tasks/hammer/mdp/spline_horizon.py`. It uses the original contact timestamp
and integer physics ticks, so separation/recontact does not restart the clock.
For example, contact at 413 ms ends at 620 ms. A contact exactly at 400 ms can
end at 600 ms. A timeout can cut short a strike that starts near the 5 s cap.

`--termination contact` is the new-training default. `--termination reference`
retains the old duration-derived cutoff (U22 step21/420ms, U20 step19/380ms).
The B configuration's native contact timer and 5 s cap are retained as-is.
Finite-horizon PPO handling remains enabled, as in the tracking baseline.

## What remains unchanged

The optimized reward remains only `0.5*dt*exp(-k*sum(error_m**2))`.
It continues to score the clamped final target after the recording ends.
No post-reference reward mask, impulse reward, active CaT pressure, physics,
action-space, observation-layout, normalization or recorded-phase change is
included. Phase still wraps using the recording's original duration. Longer
rollouts therefore expose continued-policy behavior beyond the training clip.

## Checkpoints and evaluation

New contact-mode checkpoints record the contact window, 5 s cap, reference
step count, maximum episode steps, and endpoint-tracking treatment.
Evaluation defaults to the checkpoint's own training termination mode.
A frozen reference-mode checkpoint can be explicitly continued with:

```sh
python -m scripts.z1_u22_timed_pair --arm preview --seed-index 0 \
  --coefficient 100 --evaluate MODEL_199_PT --termination contact \
  --worlds 32 --output NEW_EVAL_DIRECTORY
```

That is a separate evaluation protocol, saved in `evaluation-protocol.json`.
The checkpoint objective is still checked unchanged. Omit `--termination` to
reproduce the saved training horizon. CUDA-tagged checkpoint files require a
CUDA runtime, or a separately CPU-remapped copy for a CPU-only check.

Reference-end, contact-end and global timeout flags are reported separately.
`full_clip_rmse_mm` retains the original reference-prefix meaning, including
the endpoint once. Whole-episode and post-reference errors are separate.
Raw first-window impulse statistics include all finalized contacts, independent
of the productive-depth gate; completed windows and censored contacts are
counted. Values are captured before native reset.

## Verification

- Six new tests cover native timer reuse, reference-mode compatibility, exact
  policy-boundary timing, no-contact timeout, U20's cutoff and invalid modes.
- All 156 required reward/constraint/phase/termination tests passed locally.
- Fresh four-world, one-update CPU smoke verifies contact-mode metadata and
  checkpoint reload. No production training or Vega jobs were launched.
- A saved k100 preview checkpoint's four-world CPU-remapped evaluation has an
  exactly matching 21-step physical/reward prefix, then ends all four episodes
  at620ms. All four first-50-ms impulse windows finalize before reset.
- Legacy checkpoint auto-selection and contact-mode checkpoint evaluation are
  checked separately. Local CPU checks do not establish new GPU performance.

---
title: Z1 Hammer Reward Explorer
emoji: 🔨
colorFrom: orange
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Z1 Hammer Reward Explorer

Interactive view of the Z1 hammer-strike RL reward terms (Phase-0 thesis baseline). See each
term's equation, plot it with parameter sliders, and overlay the per-term contributions on a
real trained-policy strike — to *understand*, *share*, and *tune* the reward.

- **Term explorer** — pick a term → its LaTeX + a plot over its input domain, with a σ / param slider.
- **Strike overlay** — the depth/contact-driven terms' weighted contribution over a real `a_track`
  strike (nail 0→25 mm in ~3 contact steps), with weight sliders.

## Honesty to the code

The formulas live in `reward_formulas.py` as plain numpy (no mjlab/torch → light Space). They
mirror `src/tasks/hammer/mdp/rewards.py`; **`tests/test_reward_viz_parity.py` (in the main repo)
pins these formulas and their default params to the live env config**, so the viz can't silently
drift from training. `approach` / `r_imit` need head-position distances, so they appear in the
term explorer but not the strike overlay (the empirical trace only carries depth/contact/speed).

## Run locally

```bash
pip install -r requirements.txt
python app.py        # prints a local URL
```

## Deploy (to your Hugging Face account, private by default)

```bash
hf auth login
hf upload <your-username>/z1-reward-explorer . --repo-type space --create
# (or create a Space in the UI with SDK=gradio and push this folder)
```

## Regenerate the strike trace

`rollout_trace.json` is a recorded `a_track` strike. To refresh it from a new checkpoint, run a
rollout (`scripts/render_policy.py` prints the per-step depth/contact table) and update the JSON,
or extend it with head-position distances to light up the `approach` / `r_imit` overlays too.

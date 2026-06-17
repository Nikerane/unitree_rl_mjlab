"""Z1 Hammer Reward Explorer — a Gradio app to see / plot / tune the reward terms.

Light Space: numpy + matplotlib + gradio only (no mjlab/torch). Formula honesty to the
training code is enforced by tests/test_reward_viz_parity.py in the main repo.

Run:  python app.py   (prints a local URL)
"""

from __future__ import annotations

import gradio as gr

from plots import LATEX, PARAM, TERMS, overview_markdown, strike_overlay_figure, term_curve_figure
from reward_formulas import WEIGHTS

_LATEX_DELIMS = [{"left": "$$", "right": "$$", "display": True}]
_OVERLAY_TERMS = ["nail_driven", "nail_depth_delta", "impact_progress", "completion"]


def _on_term(term: str):
    """Term changed -> update equation, reconfigure the param slider, redraw."""
    label, lo, hi, default = PARAM[term]
    has_param = label is not None
    slider = gr.update(
        visible=has_param,
        label=label or "param",
        minimum=lo,
        maximum=hi,
        value=default,
        step=max((hi - lo) / 100.0, 1e-4),
    )
    return f"$$ {LATEX[term]} $$", slider, term_curve_figure(term, default)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Z1 Hammer Reward Explorer") as demo:
        gr.Markdown(
            "# 🔨 Z1 Hammer Reward Explorer\n"
            "See each reward term's math, plot it, and tune it. Formulas are pinned to the "
            "training code by a parity test (`tests/test_reward_viz_parity.py`)."
        )

        with gr.Tab("All terms"):
            gr.Markdown(overview_markdown(), latex_delimiters=_LATEX_DELIMS)

        with gr.Tab("Term explorer"):
            term = gr.Dropdown(TERMS, value="nail_driven", label="reward term")
            eq = gr.Markdown(f"$$ {LATEX['nail_driven']} $$", latex_delimiters=_LATEX_DELIMS)
            p_label, p_lo, p_hi, p_def = PARAM["nail_driven"]
            param = gr.Slider(
                minimum=p_lo, maximum=p_hi, value=p_def, step=max((p_hi - p_lo) / 100.0, 1e-4),
                label=p_label,
            )
            curve = gr.Plot(term_curve_figure("nail_driven", p_def))
            term.change(_on_term, term, [eq, param, curve])
            param.change(lambda t, p: term_curve_figure(t, p), [term, param], curve)

        with gr.Tab("Strike overlay"):
            gr.Markdown(
                "Per-term **weighted** contribution over a real `a_track` strike "
                "(nail 0→25 mm in ~3 contact steps). Tune the weights and watch the shape change. "
                "(`approach`/`r_imit` need head-position distances, absent from this trace.)"
            )
            w_sliders = [
                gr.Slider(
                    minimum=0.0, maximum=max(WEIGHTS[n] * 3.0, 1.0), value=WEIGHTS[n],
                    label=f"{n} weight",
                )
                for n in _OVERLAY_TERMS
            ]
            overlay = gr.Plot(strike_overlay_figure({}))

            def _redraw(*vals):
                return strike_overlay_figure(dict(zip(_OVERLAY_TERMS, vals)))

            for s in w_sliders:
                s.change(_redraw, w_sliders, overlay)

    return demo


if __name__ == "__main__":
    build_demo().launch()

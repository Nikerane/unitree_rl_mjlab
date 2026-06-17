"""Generate `reward_sheet.html` — a standalone, shareable datasheet of every reward term.

Pinned to the same source as the app (plots.LATEX/DEFINITIONS + reward_formulas.WEIGHTS), so
it can't drift. Pure HTML/CSS + KaTeX (CDN) — opens in any browser, hostable on GitHub Pages.

Run:  python build_sheet.py        # -> reward_sheet.html
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plots import DEFINITIONS, LATEX  # noqa: E402
from reward_formulas import WEIGHTS  # noqa: E402

# Phase/role chip per term.
TAG = {
    "approach": "always-on",
    "nail_driven": "always-on",
    "nail_depth_delta": "progress",
    "impact_progress": "at contact",
    "completion": "at success",
    "r_imit": "ante-impact · annealed",
    "action_rate": "penalty",
    "joint_pos_limits": "penalty",
}

# Signature device: where each term fires across the 8 control steps of a real strike
# (1-5 approach · 6 first contact · 7-8 drive · success just after).
ACTIVITY = {
    "approach": [1, 1, 1, 1, 1, 0, 0, 0],
    "nail_driven": [0, 0, 0, 0, 0, 1, 1, 1],
    "nail_depth_delta": [0, 0, 0, 0, 0, 1, 1, 1],
    "impact_progress": [0, 0, 0, 0, 0, 1, 0, 0],
    "completion": [0, 0, 0, 0, 0, 0, 0, 1],
    "r_imit": [1, 1, 1, 1, 1, 0, 0, 0],
    "action_rate": [1, 1, 1, 1, 1, 1, 1, 1],
    "joint_pos_limits": [1, 1, 1, 1, 1, 1, 1, 1],
}

# r_imit gets the cool secondary accent (its special annealed/ante-impact nature); the rest
# take an orange (reward) or red (penalty) rail by sign.
def _rail(name: str) -> str:
    if name == "r_imit":
        return "var(--cyan)"
    return "var(--neg)" if WEIGHTS[name] < 0 else "var(--accent)"


def _badge(w: float) -> str:
    cls = "neg" if w < 0 else "pos"
    return f'<span class="badge {cls}">{w:+g}</span>'


def _cells(bits, cls="") -> str:
    return '<span class="cells">' + "".join(
        f'<i class="cell{" on" if b else ""}"></i>' for b in bits
    ) + "</span>"


CSS = """
:root{
  --bg:#0F141B;--panel:#19212C;--well:#1F2A38;--line:#2C3845;
  --text:#E8EDF4;--muted:#93A1B3;--accent:#FF6A1A;--cyan:#62D0E8;--pos:#56D98A;--neg:#FF6B6B;
}
*{box-sizing:border-box}
body{margin:0;color:var(--text);line-height:1.55;-webkit-font-smoothing:antialiased;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  background:radial-gradient(1100px 560px at 82% -12%,#172230 0%,transparent 60%),var(--bg);}
.wrap{max-width:880px;margin:0 auto;padding:72px 24px 110px;}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.24em;
  text-transform:uppercase;color:var(--accent);}
h1{font-family:"Space Grotesk",sans-serif;font-weight:700;letter-spacing:-.02em;
  font-size:clamp(34px,6vw,58px);margin:12px 0 10px;}
.lede{color:var(--muted);font-size:17px;max-width:62ch;margin:0 0 30px;}
.legend{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 34px;
  font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:12px 0;}
.legend b{color:var(--text);font-weight:500;}
.cells{display:inline-flex;gap:3px;vertical-align:middle;}
.cell{width:13px;height:8px;border-radius:2px;background:#28333f;display:inline-block;}
.cell.on{background:var(--accent);box-shadow:0 0 7px rgba(255,106,26,.5);}
.card{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px 18px;margin:14px 0;overflow:hidden;}
.card::before{content:"";position:absolute;inset:0 auto 0 0;width:4px;background:var(--rail);}
.chead{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px;}
.tname{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600;}
.tag{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:3px 10px;}
.badge{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;
  border:1px solid;border-radius:8px;padding:3px 10px;}
.badge.pos{color:var(--pos);border-color:#2f6b4b;background:#12241b;}
.badge.neg{color:var(--neg);border-color:#6b3030;background:#241313;}
.eq{background:var(--well);border:1px solid var(--line);border-radius:10px;
  padding:14px 14px;margin:14px 0 12px;overflow-x:auto;}
.def{color:#C6D0DC;font-size:15px;margin:0 0 16px;}
.strip{display:flex;align-items:center;gap:10px;}
.strip .lab{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--muted);letter-spacing:.05em;}
.foot{margin-top:36px;color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:11.5px;line-height:1.7;}
.foot a{color:var(--accent);text-decoration:none;}
@media (prefers-reduced-motion:no-preference){
  .card{transition:transform .16s ease,border-color .16s ease;}
  .card:hover{transform:translateY(-2px);border-color:#3a4757;}
}
"""


def build_html() -> str:
    cards = []
    for name in LATEX:
        cards.append(
            f'<article class="card" style="--rail:{_rail(name)}">'
            f'<div class="chead"><span class="tname">{name}</span>'
            f'<span class="tag">{TAG[name]}</span>{_badge(WEIGHTS[name])}</div>'
            f'<div class="eq">$$ {LATEX[name]} $$</div>'
            f'<p class="def">{DEFINITIONS[name]}</p>'
            f'<div class="strip">{_cells(ACTIVITY[name])}'
            f'<span class="lab">fires across the strike →</span></div>'
            "</article>"
        )
    legend = (
        '<div class="legend"><b>strike timeline</b> · 8 control steps: '
        f'approach {_cells([1,1,1,1,1,0,0,0])} → first contact (6) → '
        f'drive {_cells([0,0,0,0,0,1,1,1])} → success ~step 9</div>'
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Z1 Hammer · Reward Datasheet</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.17.0/dist/katex.min.css" integrity="sha384-vlBdW0r3AcZO/HboRPznQNowvexd3fY8qHOWkBi5q7KGgqJ+F48+DceybYmrVbmB" crossorigin="anonymous">
<style>{CSS}</style>
</head><body>
<main class="wrap">
  <div class="eyebrow">Z1 hammer · reward datasheet · A-TRACK</div>
  <h1>Eight terms, one strike.</h1>
  <p class="lede">How the policy is rewarded for driving a nail flush — every reward term,
  its equation, and where it fires across the ~9-step strike. Weights are the live training
  values; equations are pinned to the code by a parity test.</p>
  {legend}
  {''.join(cards)}
  <div class="foot">Generated from <code>viz/reward_explorer/</code> (pure-numpy formulas,
  pinned by <code>tests/test_reward_viz_parity.py</code>). Interactive version:
  the Reward Explorer Gradio Space.</div>
</main>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.17.0/dist/katex.min.js" integrity="sha384-AtrdNsnxl/75rvBneBVH7DtOvCxSVahR2zWqle1coBKd8DEmLoviqNeJSx64gNAs" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.17.0/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}]}})"></script>
</body></html>"""


if __name__ == "__main__":
    out = Path(__file__).with_name("reward_sheet.html")
    out.write_text(build_html())
    print(f"wrote {out} ({len(build_html())} chars, {len(LATEX)} terms)")

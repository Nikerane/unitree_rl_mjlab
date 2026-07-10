# Docs Consolidation & Anti-Staleness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every default-reachable doc current and code-consistent; quarantine everything else in `docs/archive/` with in-file banners; enforce freshness with a pytest guard and an archive-read hook.

**Architecture:** Three phases — (A) audit every doc against the current code to produce a worklist, (B) execute fixes/merges/moves centrally with verbatim-assembly merges, (C) verify with poison-phrase greps, link checks, and the full test suite. Spec: `docs/superpowers/specs/2026-07-05-docs-consolidation-design.md` (READ IT FIRST — it is the authority for classification and policy).

**Tech Stack:** Markdown, Python/pytest (guard test), Claude Code settings.json hook.

## Global Constraints

- **Merge policy = VERBATIM ASSEMBLY** (spec, user decision): copy-paste source blocks unchanged, grouped by topic/paper, each block tagged `[from <filename>]`; delete ONLY exact duplicates, stale claims on the audit worklist, and boilerplate. NO paraphrasing/synthesis. New text = headers + one-line connective tissue only.
- **Thesis content is banner-only:** `docs/thesis/README.md`, `thesis_synthesis.md`, `thesis_direction_update.md`, `thesis_handoff_brief_original.md`, `docs/archive/*` bodies, `OPUS_AUDIT`, `PEER_REVIEW_v2` — never edit bodies, only prepend banners/move.
- **Nothing is deleted except `graphify-out/`** — superseded docs move to `docs/archive/`.
- **Commits:** repo rule is commit-only-when-asked. ASK the user once at execution start whether to commit per task; if declined, skip every commit step.
- **Archive banner template (exact, first lines of every archived file):**
  ```markdown
  > ⚠️ **ARCHIVED 2026-07-05** — superseded by `<path or “nothing; historical record”>`.
  > Facts below may contradict the current code. Do not act on them; check `docs/README.md`.
  ```
- Current-truth sources for fixing stale claims: the code on branch `soft-cat`, `derive_impulse_thresholds.py` output, and the memory files `impulse-cat-deep-review.md` / `impulse-cat-c0-findings.md`.

---

### Task 1: Phase A audit → worklist

**Files:**
- Create: `docs/superpowers/plans/2026-07-05-docs-audit-worklist.md`
- Read (clusters): (1) `docs/research/reward-design/*.md`; (2) `docs/research/*.md`; (3) `docs/*.md` + `docs/thesis/README.md` + `docs/results/*`; (4) root: `CLAUDE.md`, `README.md`, `thesis_*.md`, `docs/superpowers/**`; (5) `~/.claude/projects/-Users-nikerane-repos-unitree-rl-mjlab/memory/*.md`; (6) `~/repos/safe_impact_manipulation/{CLAUDE.md,README.md,hammer_z1_env/README.md}`.

**Interfaces:**
- Produces: the worklist file with one row per doc: `path | verdict (current/fix/merge→target/archive/record) | stale claims (quote + contradicting evidence file:line) | inbound refs (grep hits of its filename repo-wide)`.

- [ ] **Step 1:** For each cluster, read every file fully. For each doc record verdict per the spec's provisional classification (spec §End state), overriding with evidence when the spec guessed wrong. For stale claims, check against: `src/tasks/hammer/mdp/impulse_bound.py` (pulse semantics, cumulative+capped delivered), `src/tasks/hammer/cat/hook.py` (self-seeding normalizer, guards), `config/z1/env_cfgs.py`, `validate_rewards.py` (phases A–M), the seeded checklist in spec §Known stale-claims.
- [ ] **Step 2:** For each doc, run `grep -rn "<filename>" --include="*.md" --include="*.py" . ~/repos/safe_impact_manipulation` from the repo root and record inbound references (these must be repointed when the file moves).
- [ ] **Step 3:** Write the worklist file. Every `fix` row must contain the exact old text → new text. Every `archive` row must name the superseded-by target (or "historical record").
- [ ] **Step 4:** Verify completeness: `find docs CLAUDE.md README.md thesis_*.md -name "*.md" | wc -l` count equals worklist row count (excluding vendored dirs `deploy/ doc/ simulate/ .pytest_cache/`).
- [ ] **Step 5:** Commit (if authorized): `git add docs/superpowers/plans/2026-07-05-docs-audit-worklist.md && git commit -m "docs: audit worklist for consolidation"`

### Task 2: L4 freshness guard (write it RED first)

**Files:**
- Create: `tests/test_docs_current.py`

**Interfaces:**
- Produces: `LIVING_SET` (list of paths), `POISON_PHRASES` (list of regex), consumed conceptually by Tasks 3–7 (they turn this test green).

- [ ] **Step 1: Write the failing test**

```python
"""Freshness regression guard: the LIVING doc set must never re-assert superseded facts.

Archive/, results/, and dated evidence-records are exempt (they carry banners and may quote
old facts as history). Extend POISON_PHRASES whenever an iteration supersedes a documented fact.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVING_SET = [
  REPO / "CLAUDE.md",
  REPO / "docs/README.md",
  REPO / "docs/thesis/README.md",
  *sorted((REPO / "docs/research/reward-design").glob("*.md")),
]
POISON_PHRASES = [
  r"persists (?:through the air phase )?until the next window",   # old latch semantics
  r"until the next window overtakes",
  r"not exposed on the Entity",                                    # corrected qfrc_constraint note
  r"\*\*9 phases\*\*",                                             # gate is A–M now
  r"57\.3? ?ms",                                                   # C0-era window as current
  r"\[3\.44, 6\.88",                                               # C0-era J_limit as current
  r"imp_seed[^.\n]{0,40}p95",                                      # seed-as-scale (it is a floor)
  r"one switch",                                                   # C2 is a two-file switch
  r"gripper (?:holds|grips|carries) the hammer",                   # pre-L6 EE
]

def test_no_poison_phrases_in_living_docs():
  hits = []
  for p in LIVING_SET:
    if not p.exists():
      continue
    text = p.read_text(encoding="utf-8")
    for pat in POISON_PHRASES:
      for m in re.finditer(pat, text):
        line = text.count("\n", 0, m.start()) + 1
        hits.append(f"{p.relative_to(REPO)}:{line}: {pat!r} -> {m.group(0)!r}")
  assert not hits, "Superseded facts in LIVING docs:\n" + "\n".join(hits)

def test_index_and_claude_md_paths_exist():
  missing = []
  for src in (REPO / "CLAUDE.md", REPO / "docs/README.md"):
    if not src.exists():
      missing.append(f"{src} itself missing"); continue
    for ref in re.findall(r"`((?:docs|src|tests|scripts|hammer_z1_env)/[^`\s]+?\.(?:md|py))`",
                          src.read_text(encoding="utf-8")):
      if not (REPO / ref).exists():
        missing.append(f"{src.name} -> {ref}")
  assert not missing, "Dangling doc references:\n" + "\n".join(missing)
```

- [ ] **Step 2: Run it — expect FAIL** (living docs still contain poison; `docs/README.md` doesn't exist yet):
`~/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_docs_current.py -q` → FAIL listing hits.
- [ ] **Step 3:** Do NOT fix docs yet. Extend `POISON_PHRASES` with any additional recurring stale claims the Task 1 worklist surfaced (full-phrase patterns only).
- [ ] **Step 4:** Commit (if authorized): `git add tests/test_docs_current.py && git commit -m "test: docs freshness guard (RED until consolidation lands)"`

### Task 3: Fix-in-place edits

**Files:**
- Modify (per worklist `fix` rows; expected): `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md`, `FAITHFUL_SOFT_CAT_IMPL_PLAN.md`, `CONSTRAINED_RL_LANDSCAPE.md`, `OPEN_QUESTIONS.md`, `ORIENTATION_ROBUST_SIM2REAL_ARM.md`, `docs/VEGA_TRAINING_PLAN.md`, `README.md`.

- [ ] **Step 1:** Apply every `fix` row from the worklist verbatim (old text → new text). For numbers, always state EE-dependence: "measured on <date/EE>; re-derive via `derive_impulse_thresholds.py`".
- [ ] **Step 2:** `OPEN_QUESTIONS.md`: move answered items to a `## Resolved` tail with date + one-line answer + pointer (verbatim move, no rewriting of the question text).
- [ ] **Step 3:** Run `pytest tests/test_docs_current.py::test_no_poison_phrases_in_living_docs -q` — expect the reward-design hits gone (index hits remain until Task 6).
- [ ] **Step 4:** Commit (if authorized): `git add -A docs README.md && git commit -m "docs: fix stale claims in living docs (audit worklist)"`

### Task 4: Verbatim-assembly merges

**Files:**
- Create: `docs/research/reward-design/LITERATURE.md`
- Modify: `docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md` (absorb `CAT_DEEP_DIVE.md`), `CONSTRAINED_RL_LANDSCAPE.md` (absorb `JOINT_VELOCITY_BOUND_RESEARCH.md`)
- Move (archive in Task 5): `REWARD_LITERATURE.md`, `docs/research/hammering_literature_notes.md`, `docs/research/impact_tracking_rl_litreview.md`, `CAT_DEEP_DIVE.md`, `JOINT_VELOCITY_BOUND_RESEARCH.md`

- [ ] **Step 1:** Build `LITERATURE.md`: header = "Merged verbatim from REWARD_LITERATURE.md + hammering_literature_notes.md + impact_tracking_rl_litreview.md (2026-07-05); grouped by paper; annotations kept verbatim, tagged." Group entries by paper (match on arXiv ID/title); paste each source's annotation unchanged under `[from <file>]` tags; drop exact-duplicate lines and worklist-flagged stale lines only.
- [ ] **Step 2:** Append `CAT_DEEP_DIVE.md` body verbatim to `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` under `## Appendix: CaT conceptual deep-dive [from CAT_DEEP_DIVE.md, verbatim]`; likewise `JOINT_VELOCITY_BOUND_RESEARCH.md` → `CONSTRAINED_RL_LANDSCAPE.md` appendix. Remove only stale lines flagged in the worklist.
- [ ] **Step 3:** Verify assembly lossless-ness: for each source file, `grep -c "" <src>` vs count of its lines present in target — differences must equal exactly the deleted-duplicate/stale/boilerplate lines listed in the worklist row.
- [ ] **Step 4:** Commit (if authorized): `git add docs/research && git commit -m "docs: verbatim-assembly merges (literature + CaT appendices)"`

### Task 5: Archive moves + banners

**Files:**
- Move to `docs/archive/`: every worklist `archive` row + the Task 4 merge sources. Expected set: `OPUS_AUDIT.md`, `PEER_REVIEW_v2.md`, `IMPACT_PROGRESS_IMPL_SPEC.md`, `REWARD_VALIDATION_METHODOLOGY.md`, `hammering_lit_sweep_RUNBOOK.md`, `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`, `HANDOVER.md`, `FUTURE_UPDATES.md`, `REAL_HAMMER_PLAN.md`, `REWARD_LITERATURE.md`, `hammering_literature_notes.md`, `impact_tracking_rl_litreview.md`, `CAT_DEEP_DIVE.md`, `JOINT_VELOCITY_BOUND_RESEARCH.md`, dated `docs/superpowers/plans/*.md` + `specs/*.md` older than this effort.
- Modify: `docs/archive/README.md` (one line per file: what it was, why archived, superseded-by).
- Banner (template from Global Constraints) prepended to every moved file AND to the kept evidence-records (`hammering_reward_design_deep_dive_v2.md`, `tracking_impact_impulse_design_research.md`, `thesis_handoff_brief_original.md` — record-variant banner: "dated research record, kept for provenance").

- [ ] **Step 1:** `git mv` each file (preserves history); prepend the banner with the worklist's superseded-by target.
- [ ] **Step 2:** Repoint every inbound reference recorded in the Task 1 worklist (grep hits) to the new path or the superseding doc.
- [ ] **Step 3:** Verify: `grep -rln "reward-design/REWARD_LITERATURE\|CAT_DEEP_DIVE\|JOINT_VELOCITY_BOUND_RESEARCH" --include="*.md" --include="*.py" docs src tests scripts CLAUDE.md README.md` → only `docs/archive/**` hits.
- [ ] **Step 4:** Commit (if authorized): `git add -A && git commit -m "docs: archive superseded docs with banners + archive index"`

### Task 6: `docs/README.md` index

**Files:**
- Create: `docs/README.md`

- [ ] **Step 1:** Write the index with exactly four sections, pointers-only (no facts that live elsewhere):
  1. **Current truth map** — one line per living doc (from the worklist `current`/`fix` rows) + the authority rule ("code > this index > living docs; archive/ = history").
  2. **Reading order** — for onboarding: CLAUDE.md → thesis/README → IMPULSE_CAT_IMPL_PLAN → FAITHFUL_SOFT_CAT_IMPL_PLAN → CONSTRAINED_RL_LANDSCAPE → LITERATURE → OPEN_QUESTIONS.
  3. **How we got here** — ≤25 lines, chronological: baseline reward → velocity-bound ablations → soft-CaT → impulse arm C0 → two review rounds → L6-fixture migration (pending); each line links its archived evidence.
  4. **Code entry map** — ~30 lines: `src/tasks/hammer/` files → one-line role → suggested read order (env_cfgs → hammer_env_cfg → mdp/rewards → mdp/impulse_bound → cat/* → rl/*), + scripts and gates.
- [ ] **Step 2:** Run `pytest tests/test_docs_current.py -q` — `test_index_and_claude_md_paths_exist` must now pass for the index.
- [ ] **Step 3:** Commit (if authorized): `git add docs/README.md && git commit -m "docs: single current-truth index with journey + code map"`

### Task 7: CLAUDE.md rewrite + memory pruning + sibling pointers

**Files:**
- Modify: `CLAUDE.md`, `~/.claude/projects/-Users-nikerane-repos-unitree-rl-mjlab/memory/*` (per worklist), `~/repos/safe_impact_manipulation/CLAUDE.md` (pointer check only).

- [ ] **Step 1:** Rewrite CLAUDE.md: keep thesis-context + direction-update + live-weights + environment sections (verbatim where still true); replace the doc-pointer paragraphs with the authority contract (spec L3, exact three lines) + a pointer to `docs/README.md`; fix any worklist-flagged stale lines.
- [ ] **Step 2:** Memory: apply worklist verdicts (merge near-duplicates, delete falsified, refresh `MEMORY.md` one-liners). Corrections must keep their dated UPDATE markers.
- [ ] **Step 3:** Sibling repo CLAUDE.md/README: fix only dangling pointers into this repo (no content edits).
- [ ] **Step 4:** Run the FULL suite: `~/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/ -q` → all pass including `test_docs_current.py` fully green.
- [ ] **Step 5:** Commit (if authorized): `git add CLAUDE.md && git commit -m "docs: CLAUDE.md authority contract + index pointer"` (memory lives outside the repo — no commit).

### Task 8: L6 hook + graphify-out removal + final verification

**Files:**
- Modify: `.claude/settings.json` (create if absent)
- Delete: `graphify-out/` (untracked)

- [ ] **Step 1:** Add the PreToolUse hook via the update-config skill (harness executes hooks; settings must be edited through it). Target behavior — on Read of `docs/archive/**`, inject:
  `⚠️ This file is ARCHIVED and superseded. Do not act on its contents; check docs/README.md for current truth.`
  Hook shape (adjust to the skill's schema): PreToolUse matcher `Read`, command filtering `tool_input.file_path` for `/docs/archive/`, emitting the warning as additionalContext (exit 0 — warn, don't block).
- [ ] **Step 2:** Verify the hook fires: Read any archived file in a fresh tool call → the warning appears in the tool result context.
- [ ] **Step 3:** `rm -rf graphify-out/` (untracked, regenerable via /graphify post-cleanup).
- [ ] **Step 4:** Phase C sweep: (i) `pytest tests/ -q` all green; (ii) for every file moved in Task 5, `grep -rn "<old filename>" --include="*.md" --include="*.py" . | grep -v docs/archive` → empty; (iii) open `docs/README.md` and spot-check 5 random pointers resolve.
- [ ] **Step 5:** Commit (if authorized): `git add .claude/settings.json && git commit -m "chore: archive-read warning hook; drop stale graphify snapshot"`

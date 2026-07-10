# Docs consolidation & anti-staleness design (2026-07-05)

**Goal.** After six weeks of heavy iteration, the ~35 project-authored markdown docs (plus 17
agent-memory files) contain contradictions with the current code and each other. Two failure modes
to eliminate: (1) a human diving into the code gets misled by a stale doc; (2) an agent ingests a
superseded fact (via CLAUDE.md, memory, or grep-driven reading) and acts on it. The deliverable is
a **trustworthy default surface**: everything reachable without deliberate digging is current and
code-consistent; everything else is loudly quarantined; freshness is enforced by a regression test
and a harness hook, not by good intentions.

**Non-goals.** No thesis-content rewriting (`docs/thesis/README.md` and the three root
`thesis_*.md` direction docs keep their content; banners only if needed). No code changes. No
deletions of provenance-bearing content — archive instead (git history is the ultimate archive; the
user granted free hand to edit/move, option (a)).

## End state

```
CLAUDE.md                        rewritten: shorter, authority contract, pointers into the index
docs/README.md                   NEW single index:
                                   - current-truth map (each living doc, one line)
                                   - reading order for onboarding
                                   - "How we got here" timeline (≤25 lines, pointers to archive)
                                   - code entry map (~30 lines: file → role → read order)
docs/thesis/README.md            unchanged (deep decision rationale)
docs/research/reward-design/     living set (~6 docs + scripts):
                                   IMPULSE_CAT_IMPL_PLAN.md        (fixed in place)
                                   FAITHFUL_SOFT_CAT_IMPL_PLAN.md  (fixed; absorbs CAT_DEEP_DIVE)
                                   CONSTRAINED_RL_LANDSCAPE.md     (fixed; absorbs JOINT_VELOCITY_BOUND_RESEARCH)
                                   LITERATURE.md                   (NEW: merge of REWARD_LITERATURE +
                                                                    hammering_literature_notes +
                                                                    impact_tracking_rl_litreview, with
                                                                    source attribution lines)
                                   OPEN_QUESTIONS.md               (pruned to actually-open; answered
                                                                    items moved to a "Resolved" tail
                                                                    with date + pointer)
                                   NAIL_PRECISION_CURRICULUM.md    (kept)
                                   *.py validation/gate scripts    (kept)
docs/results/                    kept as-is (dated records by design; README notes that)
docs/archive/                    all superseded docs; archive/README.md gets one line per file:
                                   what it was, why archived, superseded-by pointer
graphify-out/                    DELETED (untracked, generated from pre-cleanup docs — a fossilized
                                   poison reservoir; regenerate on demand)
tests/test_docs_current.py      NEW freshness regression guard (see Layer 4)
.claude/settings.json           NEW PreToolUse hook (see Layer 6)
```

**Provisional classification** (finalized by the Phase A audit; the audit may re-classify with
evidence):

- **Fix in place:** IMPULSE_CAT_IMPL_PLAN, FAITHFUL_SOFT_CAT_IMPL_PLAN, CONSTRAINED_RL_LANDSCAPE,
  OPEN_QUESTIONS, ORIENTATION_ROBUST_SIM2REAL_ARM, repo README.md, VEGA_TRAINING_PLAN (or archive
  if dead), sibling-repo CLAUDE.md pointer check.
- **Merge then archive originals:** REWARD_LITERATURE + hammering_literature_notes +
  impact_tracking_rl_litreview → LITERATURE.md; CAT_DEEP_DIVE → FAITHFUL_SOFT_CAT_IMPL_PLAN;
  JOINT_VELOCITY_BOUND_RESEARCH → CONSTRAINED_RL_LANDSCAPE.
- **Archive:** OPUS_AUDIT, PEER_REVIEW_v2, IMPACT_PROGRESS_IMPL_SPEC (shipped),
  REWARD_VALIDATION_METHODOLOGY (if validate_rewards.py docstring suffices),
  hammering_lit_sweep_RUNBOOK, TRACKING_IMPACT_IMPULSE_IMPL_PLAN (absorbed portions noted),
  HANDOVER, FUTURE_UPDATES (if stale), REAL_HAMMER_PLAN (if executed), dated
  docs/superpowers/plans+specs (except this one).
- **Keep as dated evidence-records (banner: "research record"):** hammering_reward_design_deep_dive_v2,
  tracking_impact_impulse_design_research, thesis root docs, docs/results/*.
- **Memory:** apply the same rubric to the 17 files — merge near-duplicates, correct or delete
  falsified ones, refresh MEMORY.md hooks.

## Anti-poisoning layers (all in scope)

- **L1 — Kill contradictions in living docs.** The Phase A audit reads every doc against the
  current code and extracts each stale claim with file:line evidence; Phase B fixes them.
- **L2 — In-file self-identification.** Every archived/superseded doc gets a first-line banner:
  `> ⚠️ ARCHIVED <date> — superseded by <path>. Facts below may contradict current code; do not
  act on them.` The banner travels with grep-driven reads; the `docs/archive/` path in every hit
  is the second signal.
- **L3 — Authority contract in CLAUDE.md.** "Current truth = code > docs/README.md index > the
  living docs it lists. docs/archive/** and dated records are historical evidence — never act on
  them without checking the index."
- **L4 — Freshness as a failing test.** `tests/test_docs_current.py`: greps a defined LIVING set —
  CLAUDE.md, docs/README.md, docs/thesis/README.md, and the reward-design living docs (NOT
  archive/, results/, or dated evidence-records, which carry banners instead and may legitimately
  quote old facts as history) — for a poison-phrase list seeded from the audit (at minimum: the
  latched "persists until the next window" semantics, 57 ms / 3.44 / 6.88 / seed-0.16 / 0.137 N·s
  asserted as current numbers, "not exposed on the Entity", "9 phases", "one switch", pre-L6
  gripper-holds-hammer claims, imp_seed-as-scale). Memory files are checked with full-phrase
  patterns only (they may record old numbers inside explicitly-marked correction notes). Also
  asserts every path referenced by CLAUDE.md and docs/README.md exists. Runs in the normal pytest
  suite.
- **L5 — Delete graphify-out/** (regenerate on demand post-cleanup).
- **L6 — Harness hook (hard mechanism).** PreToolUse hook in `.claude/settings.json` matching
  Read of `docs/archive/**`: injects "⚠️ archived doc — superseded; verify against docs/README.md
  before acting on anything in it." Configured via the update-config skill.

## Known stale-claims checklist (Phase A seeds; audit extends)

latch→pulse window semantics; per-event delivered cap (press-farming); imp_seed = decay floor
(self-seeding normalizer); C2 = two-file switch (imp_max_p + per-joint imp_limit tensor);
gate numbers are EE-dependent (L6 fixture drift: 57→38 ms etc.) and must be re-derived;
"9 phases" → A–M; "qfrc_constraint not exposed" corrected; weld-vs-friction contaminant;
decided items: soft-CaT chosen (Khadiv confirm pending), MuJoCo-native ground truth (Pinocchio
deferred); Λ-quantity decision (baseline-subtracted + rigorous isolation) still open.

## Process

- **Phase A — audit (workflow, parallel readers).** Cluster the docs (~5 clusters + memory);
  each reader returns per-doc: verdict (current/fix/merge/archive/record), stale claims with
  doc-line + contradicting code/gate evidence, inbound references. Output = the execution
  worklist + the poison-phrase list for L4.
- **Phase B — execute (centrally, single editorial voice).** Fixes, merges (with source
  attribution), moves + banners, new index, CLAUDE.md rewrite, memory pruning, L4 test, L6 hook,
  graphify-out deletion.
- **Phase C — verify.** (i) poison-phrase grep over living set + memory = zero hits;
  (ii) link-check every path in CLAUDE.md/index/thesis digest/archive README; (iii) full pytest
  (includes the new guard + doc-referencing suites); (iv) repo-wide grep for each moved filename
  to catch broken inbound links.

## Risks

- Merges drop nuance → **merge policy is VERBATIM ASSEMBLY, not synthesis** (user decision,
  2026-07-05): copy-paste source blocks unchanged, grouped by topic/paper with every source's
  annotation kept verbatim under a `[from <file>]` tag; delete ONLY exact duplicates, stale claims
  on the audit list, and boilerplate. The only newly written text is headers and one-line
  connective tissue (pointer-grade, no facts). Paraphrase is where new errors enter — none is
  allowed. Originals archived, not deleted.
- The index becomes tomorrow's staleness → pointers-only; no fact lives in the index that lives
  elsewhere; L4 covers it.
- Hook fatigue → L6 fires only on archive reads, which should be rare and deliberate.
- Aggressive banner-editing of thesis docs could disturb the defense narrative → thesis content
  edits are banner-only (non-goal guard).

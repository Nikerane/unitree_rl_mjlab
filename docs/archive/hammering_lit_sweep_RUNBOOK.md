> ⚠️ **ARCHIVED 2026-07-05** — executed sweep playbook, kept for provenance; its OUTPUT lives in `docs/research/reward-design/LITERATURE.md`.
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Hammering Literature — Full-Sweep RUNBOOK

**Purpose.** A re-runnable, **account-switch-aware** playbook for a comprehensive *learning-for-hammering* literature sweep using Claude + the **Workflow** tool. Structured into phases with explicit **🛑 STOP** points so you can top up tokens (switch accounts) between heavy steps.

**How to run it.** Tell Claude: *"Execute the hammering full-sweep runbook (`docs/research/hammering_lit_sweep_RUNBOOK.md`)."* Claude runs **one phase**, reports, then **STOPS at the checkpoint**. You switch account, then say *"continue with Phase X."* Repeat.

**Outputs feed:** `hammering_literature_notes.md` (the annotated field map, §2.x + harvested ideas) and `hammering_reward_design_deep_dive_v2.md` (the reward-design report). Don't duplicate — append/extend.

---

## 0. Process rules (baked-in lessons from the 2026-05-29 run)

These are *non-negotiable* — they fix what broke last time:

1. **Top up tokens BEFORE each phase, switch at the STOPs.** Mid-workflow account switches are what fragmented the last run. Each phase is sized to run inside one topped-up session.
2. **Free-text workflow agents, NEVER `schema`.** A schema-based run returned **0 usable results** (agents finished without calling `StructuredOutput`). The free-text + synthesis pattern below works.
3. **`agentType: 'general-purpose'`** on every workflow agent — proven web access (default workflow agents had patchy web/tool access).
4. **Claude (the orchestrator, on Opus) verifies every load-bearing citation itself** via `WebFetch`. Do **not** trust agent self-verification for the permanent record — last run, agent claims contained ≥3 real errors (a fabricated "force-on-nail sensor," a non-hammering paper misc', a wrong headline number).
5. **CN/JP and rate-limited DBs (CNKI, CiNii, Wanfang, J-STAGE) → main-loop DIRECT searches, not parallel agents.** The workflow's CN/JP lenses died on `WebSearch` "Not logged in" limits; the orchestrator's own `WebSearch`/`WebFetch` worked.
6. **Optionally set a budget directive** (e.g. append `+300k` to your message) at a phase start so the workflow can scale fan-out deterministically instead of Claude eyeballing it.
7. **Tier every citation** (see §6) and mark `◔` anything an agent reported but Claude didn't re-fetch.

---

## 1. What is already covered (EXCLUDE — only surface NET-NEW)

Full annotated set: `hammering_literature_notes.md` §2.1–2.7. Compact exclusion list to paste into agent prompts:

**Hammering-specific (control + learning):** Tandon thesis (USC/Arbib); Vu et al. 2026 QP impact-momentum (HRP-5P, AMC); Ti et al. 2024 tool-affordance nail hammering (2402.05502); Hitaka & Izumi 1995 / Izumi & Zhou 1999 (flexible-link hammer); Karniel & Inbar 1997; Petrič 2017 ("Hammering Does Not Fit Fitt's Law"); Wang & Kheddar 2019 (RSS impact-friendly); Romanyuk et al. 2019 (ICMA multiple-working-mode); Tsujita 2008 (humanoid nailing); Matsumoto 2006 (board-breaking); Imran & Yi 2016 (dual-arm hammering); Garabini 2011 (VSA hammer); Kobayashi 2016 (input shaping); ARMADA (Kim 2025, 2502.16908).
**Adroit `hammer` benchmark + users:** Rajeswaran 2018 DAPG (1709.10087); Fu 2020 D4RL (2004.07219); AdroitHandHammer env; RRL (2107.03380); VRL3 (2202.10324); H-InDex (2310.01404); MoDem (2212.05698); DexHandDiff (CVPR 2025); DORA (2505.14819); SimToolReal (2602.16863); CQL; IQL (2110.06169); Orbik (TUM MSc 2020 / ICDL 2021); EAAI-2025 "BBE single-demo".
**Direct nail-driving (learning, real):** Tool-as-Interface (2504.04612, CoRL 2025); HMAMP (2510.24257, Robotica 2025); "Let Robots Swing a Hammer" (IROS 2025, IEEE 11246617); Teramae 2018 (PMC6232299).
**Repetitive/ballistic impact (drumming):** Robot Drummer (2507.11498); DexDrummer (2603.22263); Karbasi 2024 (PMC11609846).
**Adjacent striking + novel-reward + analogy:** Poke-and-Strike (2509.00178); Prolonging Tool Life (2507.17275); Robometer (2603.02115); Chang impedance-DMP (2203.07191); + the full v2 corpus (CPO, reward machines, Siekmann, Skalse, Pan, CPG-RL, DeXtreme, TossingBot, VICES, CAPS, AutoMate, Beltran-Hernandez, Wang/Dehio/Kheddar 2022, Qian, Hwangbo, etc. — see v2 References).
**Theses:** Tandon (USC), Orbik (TUM), Van Rooyen (UVic), Qin (Yale).

## 2. What is SATURATED (do NOT re-run — diminishing returns)

- Broad "RL hammering / nail-driving" keyword sweeps (run twice; the Adroit ecosystem + real long tail are found).
- CN/JP broad queries — **CiNii returned 0** for nail-driving-RL and hammering-RL; CNKI/Wanfang/J-STAGE surfaced only general RL. Empty.
- Generic dexterous-manipulation benchmark suites (robosuite/ManiSkill/Bi-DexHands/DexMV have **no** native hammer task — verified).

**Only the targeted searches in §3 are worth re-running.**

---

## 3. The phased sweep (with STOP points)

> Time/token estimates are rough; each phase fits one topped-up session.

### Phase A — Citation-graph crawl *(the highest-value NEW search)* → 🛑 STOP
**Why:** different *method* than keyword search — catches papers that share no keywords. **This is the one broad search still worth doing.**
**Run:** the **Citation-Crawl workflow** in §5.1 (anchors: DAPG `1709.10087`, Tool-as-Interface `2504.04612`, IROS-25 tactile hammer, Vu QP, Robot Drummer `2507.11498`).
**Output:** list of net-new citing/cited papers, agent-verified.
**🛑 STOP:** Claude reports the raw candidate list, then halts. **Switch account.**

### Phase B — Verification (Claude self-fetch) → 🛑 STOP
**Why:** integrity. Agents over-claim.
**Run:** Claude `WebFetch`-verifies each Phase-A candidate (authors/venue/year + the hammering relation), in parallel batches of ~6. Drop anything unconfirmable (gray zone = FAIL). Tier per §6.
**🛑 STOP:** Claude reports the verified net-new set. **Switch account.**

### Phase C — Targeted gap searches (Claude direct, NOT a workflow) → 🛑 STOP
**Why:** these are the rate-limited / login-walled / narrow areas — do them in the main loop.
Claude runs, directly:
- **CNKI/Wanfang full-text** (only if you have institutional access — otherwise skip; CiNii being empty makes yield low).
- **Construction/industrial venues** keyword search: *Automation in Construction*, ISARC, IEEE CASE for percussive/nailing/riveting **learning** (not arXiv).
- **`◔` cleanup:** confirm any still-unverified attributes (e.g. EAAI-2025 author list; SimToolReal hammer subtask; HITTER/badminton relevance).
- **New-since-last-run arXiv listing:** `cs.RO` "hammer / percussive / impact striking + learning" for anything dated after the last sweep.
**🛑 STOP:** Claude reports. **Switch account.**

### Phase D — Synthesis + write-up → 🛑 STOP
**Run:** Claude folds verified net-new finds into `hammering_literature_notes.md` (§2.7 table + harvested-ideas list) and, if any change the reward design, into `hammering_reward_design_deep_dive_v2.md` (§4–§8 + References). Apply integrity corrections in-line.
**🛑 STOP:** Claude reports the diff. **Done, or switch for Phase E.**

### Phase E — Monitoring (optional, set-and-forget)
**Why:** the field is moving fast (the real long tail is all 2025–26). Cheaper than re-sweeping.
**Run:** save the §5.1 discovery workflow as a named workflow and `/schedule` it monthly, OR set Google-Scholar/arXiv alerts for `robot hammering reinforcement learning`, `percussive manipulation learning`, `impact striking RL`. Claude can also do a 1-shot "new since <date>" check on demand.

---

## 4. Optional Phase A′ — Full broad discovery re-sweep (only if >6 months elapsed)

If a lot of time has passed, re-run the **9-lens discovery workflow** in §5.2 (it's the pattern that worked last time). Otherwise skip — §2 says it's saturated.

---

## 5. Embedded workflow scripts (copy-paste ready, proven pattern)

> Both use **free-text agents** (no schema) + `agentType:'general-purpose'` + a synthesis agent. This is the pattern that produced results last run.

### 5.1 Citation-Crawl workflow (Phase A)

```javascript
export const meta = {
  name: 'hammering-citation-crawl',
  description: 'Citation-graph crawl on anchor hammering/striking papers to find net-new learning work',
  phases: [ { title: 'Crawl' }, { title: 'Synthesize' } ],
}
const KNOWN = `EXCLUDE (already in corpus): DAPG/Adroit, D4RL, RRL, VRL3, H-InDex, MoDem, DexHandDiff, DORA, SimToolReal, CQL, IQL, Orbik, EAAI-2025 BBE, Tool-as-Interface, HMAMP, Let-Robots-Swing-a-Hammer, Teramae 2018, Robot Drummer, DexDrummer, Karbasi 2024, Poke-and-Strike, Prolonging Tool Life, Vu 2026, Ti 2024, Romanyuk 2019, Tsujita 2008, ARMADA, Tandon/Van Rooyen/Qin theses. (Full list: hammering_literature_notes.md §1–2.7.)`
const RULES = `Use WebSearch + WebFetch (load via ToolSearch "select:WebSearch,WebFetch" if needed). Confirm title+authors+venue from the source. Report ONLY confirmed items; mark LEARNING vs CONTROL and hammering relation. No fabrication. Concise.`
const ANCHORS = [
  { key: 'dapg', q: `Papers that CITE "Learning Complex Dexterous Manipulation with Deep RL and Demonstrations" (Rajeswaran 2018, arXiv:1709.10087) AND specifically use/extend the HAMMER task. Use Semantic Scholar "cited by" (semanticscholar.org/arxiv/1709.10087) and Google Scholar.` },
  { key: 'tool-as-interface', q: `Citing + cited-by papers of "Tool-as-Interface" (Chen et al., arXiv:2504.04612) that involve hammering/striking/impact LEARNING.` },
  { key: 'iros-tactile', q: `Citing + related papers of "High-dynamic Tactile Sensing... Let Robots Swing a Hammer" (IROS 2025, IEEE Xplore 11246617) on learned hammering/impact.` },
  { key: 'vu-qp', q: `Citing + cited-by of Vu et al. 2026 "QP-based impact momentum maximization for a hammering task" (AMC 2026) that are LEARNING-based (RL/IL) for hammering/impact.` },
  { key: 'robot-drummer', q: `Citing + related of "Robot Drummer" (arXiv:2507.11498) and "DexDrummer" (arXiv:2603.22263) on learned repetitive/rhythmic impact relevant to hammering.` },
]
phase('Crawl')
const found = await parallel(ANCHORS.map((A) => () =>
  agent(`Citation-graph crawler. ${A.q}\n\n${KNOWN}\n\n${RULES}\nReturn confirmed NET-NEW items: Title — Authors (Year), Venue, id, METHOD=learning|control, RELATION, one-line. Plus "unverified leads" and "dead ends".`,
    { agentType: 'general-purpose', label: `crawl:${A.key}`, phase: 'Crawl' })))
const block = ANCHORS.map((A, i) => `## ${A.key}\n${found[i] || '(none)'}`).join('\n\n')
phase('Synthesize')
const synthesis = await agent(`Dedupe + rank these citation-crawl findings into a markdown master list (Title|Authors(Year)|Venue|id|relation|1-line). Separate LEARNING from CONTROL. List items needing human re-verification. Be honest about dead-ends.\n${KNOWN}\n\nFINDINGS:\n${block}`,
  { agentType: 'general-purpose', label: 'synthesize', phase: 'Synthesize' })
return { synthesis, anchors: ANCHORS.map((A, i) => ({ key: A.key, text: found[i] || null })) }
```

### 5.2 Broad discovery workflow (Phase A′ — the pattern that worked)

```javascript
export const meta = {
  name: 'hammering-discovery-sweep',
  description: 'Free-text multi-lens discovery sweep for learning-based hammering, self-verifying + synthesis',
  phases: [ { title: 'Sweep' }, { title: 'Synthesize' } ],
}
const KNOWN = `EXCLUDE (already found): <<paste the §1 exclusion list>>. Hunt the LONG TAIL of LEARNING-BASED (RL/imitation/LfD/evolutionary/diffusion) HAMMERING / nail-driving / percussive / impact striking.`
const RULES = `Use WebSearch + WebFetch (load via ToolSearch "select:WebSearch,WebFetch" if needed). 5-8 varied searches; FETCH + CONFIRM each before reporting. Report only confirmed; mark LEARNING vs CONTROL + hammering relation; "unverified leads" + "dead ends" sections. No fabrication.`
const LENSES = [
  { key: 'direct-rl', f: `"reinforcement learning hammer nail robot", "learning to hammer", "deep RL nail driving", "RL percussive manipulation".` },
  { key: 'imitation-lfd', f: `"learning from demonstration hammering", "diffusion policy hammer", "DMP hammering primitive", "residual policy hammering".` },
  { key: 'drumming-striking', f: `"reinforcement learning robotic drumming", "learned timed/rhythmic impact", "robot learning to strike", "ballistic striking RL".` },
  { key: 'humanoid-locomanip', f: `"humanoid learning hammering", "loco-manipulation forceful tool RL", "whole-body impact task learning humanoid".` },
  { key: 'novel-reward', f: `"tool-wear / fatigue reward reinforcement learning", "recoil-aware impact learning", "impact-aware reward design RL".` },
  { key: 'recent-arxiv', f: `cs.RO 2025-2026 + CoRL/RSS/ICRA/IROS: "hammer / percussive / impact striking + learning" newest first.` },
]
phase('Sweep')
const sweep = await parallel(LENSES.map((L) => () =>
  agent(`Discovery agent. LENS: ${L.f}\n\n${KNOWN}\n\n${RULES}\nReturn ~10 confirmed items max.`,
    { agentType: 'general-purpose', label: `sweep:${L.key}`, phase: 'Sweep' })))
const block = LENSES.map((L, i) => `## ${L.key}\n${sweep[i] || '(none)'}`).join('\n\n')
phase('Synthesize')
const synthesis = await agent(`Dedupe + rank into a markdown master list; separate learning vs control; flag items to re-verify; honest dead-ends.\n${KNOWN}\n\nFINDINGS:\n${block}`,
  { agentType: 'general-purpose', label: 'synthesize', phase: 'Synthesize' })
return { synthesis, lenses: LENSES.map((L, i) => ({ key: L.key, text: sweep[i] || null })) }
```

> **Note for Claude:** after a workflow returns, **read the full result from its `.output` file** (the notification truncates), then **WebFetch-verify** the load-bearing items yourself before writing anything into the notes.

---

## 6. Integrity / tiering rules

- `[E1]` peer-reviewed venue, Claude verified by direct fetch. `[E2]` preprint, verified to exist. `[E2*]` agent-reported canonical, not re-fetched. `[E3]` inherited corpus. `◔` agent-reported, one attribute (usually the hammer subtask) unconfirmed. `[ANALOGY]` not about hammering. `[PROPOSAL]` Claude's own idea, no citation.
- **Gray zone = FAIL.** If existence can't be confirmed, it does not enter the notes.
- **Never blend authors across papers** (this project had a real fabrication incident — Meta-World authors).
- Quote the reward/impact formula verbatim where the source gives one.

---

## 7. Quick checklist (per run)

- [ ] Tokens topped up; budget directive set (optional).
- [ ] Phase A citation-crawl workflow → **STOP**, switch account.
- [ ] Phase B Claude-verify candidates → **STOP**, switch account.
- [ ] Phase C direct gap searches (CNKI access? industrial venues? `◔` cleanup? new-arXiv) → **STOP**, switch.
- [ ] Phase D write into `hammering_literature_notes.md` (+ v2 if design-relevant), with tiers + corrections → **STOP**.
- [ ] Phase E (optional) monitoring alerts / scheduled monitor.
- [ ] Update §1 exclusion list + §2 "saturated" with anything new, so the next run starts where this one ended.

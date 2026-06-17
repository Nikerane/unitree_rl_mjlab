# Archived docs

Superseded / historical documents, moved here **2026-06-17** to declutter the live
`docs/` tree. Nothing here is deleted — it remains the provenance/research trail and
is still cross-referenced (by filename) from active docs. If a prose mention elsewhere
points at one of these, look here.

| Doc | Why archived | Superseded by |
|---|---|---|
| `DEEP_RESEARCH_REPORT.md` | Pre-2026-06-02 synthesis; predates the 7-term config + `impact_progress` | `../research/hammering_reward_design_deep_dive_v2.md` (current reward-design entry point) |
| `RECOMMENDED_REWARD_SPEC.md` | "Aspirational" 9-term spec; the implemented 7-term reward (code) is the source of truth | `src/tasks/hammer/hammer_env_cfg.py` + `IMPACT_PROGRESS_IMPL_SPEC.md` |
| `REWARD_DESIGN_MATRIX.md` | Lightweight term table; duplicates `RECOMMENDED_REWARD_SPEC.md` §3–4 | `RECOMMENDED_REWARD_SPEC.md` (also archived) |
| `IMPACT_TRACKING_REWARD_SPEC.md` | The generate-then-track / DeepMimic architecture the supervisor **walked back** | `../research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (weak annealed reward prior, not two-stage) |
| `BASELINE_AUDIT.md` | 2026-05-22 point-in-time validation snapshot | `../VEGA_TRAINING_PLAN.md` + `../HANDOVER.md` |

Still live (NOT archived): the literature corpus (`REWARD_LITERATURE`, `hammering_*`,
`impact_tracking_rl_litreview`), the audit/review trail (`OPUS_AUDIT`, `PEER_REVIEW_v2`),
`REWARD_VALIDATION_METHODOLOGY`, `IMPACT_PROGRESS_IMPL_SPEC` (its term is still in the
reward), and all current plans (`TRACKING_IMPACT_IMPULSE_IMPL_PLAN`, `OPEN_QUESTIONS`,
`REAL_HAMMER_PLAN`, `VEGA_TRAINING_PLAN`).

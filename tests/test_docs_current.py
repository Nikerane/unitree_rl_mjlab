"""Freshness regression guard: the LIVING doc set must never re-assert superseded facts.

The LIVING set is everything reachable by default without deliberate digging — CLAUDE.md,
the docs/README.md index, the thesis digest, and the reward-design living docs. docs/archive/,
docs/results/, dated evidence-records, and the agent-memory dir are EXEMPT: they carry banners
and may legitimately quote old facts as history, so they are not scanned here.

Each POISON_PHRASE is a fact that a docs-consolidation iteration superseded. Every phrase is
chosen to be:
  * RED now  — it currently matches at least one LIVING doc, and
  * GREEN after consolidation — a fix/merge/archive step removes that match.
When an iteration supersedes a new fact, add its phrase here (full-phrase, specific — never a
current fact like "7-term" or "position-only DiffIK", and never a phrase that lives only in
memory/archive, which this guard does not scan).

The curated list below was validated by an adversarial review (2026-07-05): each entry was
grep-confirmed to hit a living doc now and traced to the consolidation step that removes it.
"""
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# LIVING set — default-reachable, must stay current. docs/README.md is created by the
# consolidation (Task 6); the reward-design glob shrinks as merge-sources are archived (Task 5).
LIVING_SET = [
    REPO / "CLAUDE.md",
    REPO / "docs/README.md",
    *sorted((REPO / "docs/thesis").rglob("*.md")),
    *sorted((REPO / "docs/research/reward-design").glob("*.md")),
]

# Each phrase is tagged with the living doc it currently hits and the consolidation step that
# clears it. `fix` = fixed in place (Task 3/7); `merge/archive` = source leaves the living glob
# (Task 4/5). Unicode literals (× · − – ≈ Λ) are matched as-is (source is UTF-8).
POISON_PHRASES = [
    # IMPULSE_CAT_IMPL_PLAN.md:6 gripper-era constants — Task 3 poison-free replacement (C-1)
    r"4\.7% of the worst-joint limit",
    r"0\.137 N[·.]s",
    r"~?21× more violent",
    r"not the weld \(~0\.2\)",
    r"\[3\.44, 6\.88",
    # IMPULSE_CAT_IMPL_PLAN.md:87 quoted stale code comment — Task 3 rewrite (C-10)
    r"not exposed on the Entity",
    # FAITHFUL_SOFT_CAT_IMPL_PLAN.md:3 status — Task 3 fix
    r"\*\*Status:\*\* pre-implementation",
    # FAITHFUL_SOFT_CAT_IMPL_PLAN.md:275,304 stale filename — Task 3 fix (incl. C-9 line 304)
    # (\b keeps a future soft_cat_hook.py from false-positing: "_" is a word char, so no boundary)
    r"\bcat_hook\.py",
    # gate phase counts: FAITHFUL:309 fixed; OPUS_AUDIT/IMPACT_PROGRESS/PEER/TRACKING archived (Task 5)
    # (lookbehind so a future "19 phases" doesn't false-positive)
    r"(?<!\d)9 phases",
    r"All 8 validation phases",
    r"all 8 phases pass",
    r"8 phases pass unchanged",
    # reward-stack term count: OPEN_QUESTIONS:185 fixed (C-11); OPUS_AUDIT archived
    r"6-term reward stack",
    # aspirational 9-term spec: PEER_REVIEW_v2 archived
    r"9-term SPEC",
    # CAT_DEEP_DIVE.md claims falsified by shipped code — dropped on merge, source archived (Task 4/5)
    r"NOT real CaT",
    r"the crude approximation",
    r"A3's is a no-op",
    r"mjlab 1\.4\.0 may not expose",
    r"does mjlab expose substep",
    # stale nail_depth_delta weight arithmetic: REWARD_LITERATURE/RVM archived; OPEN_QUESTIONS table replaced (C-11)
    # (short pattern by design — a spurious hit on future arithmetic like "4096 × 500" is a cheap
    # human inspection, a missed weight-arithmetic regression is not)
    r"× 500",
    r"already at −1\.0 in current config",
    # TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md archived (Task 5) — stale counts/gripper-era I_ref/superseded design
    r"147 unit tests",
    r"10/10 phases",
    r"I_ref ≈ 0\.32 N·s",
    r"windowed axial impulse",
    r"phases A–I \*\*\+ J/K/L/M",
    # CLAUDE.md:14 — Task 7 fix (C-2)
    r"weld-pollution blocker",
    # CAT_DEEP_DIVE.md link must be retargeted to FAITHFUL everywhere (Task 4/5, C-12)
    r"CAT_DEEP_DIVE\.md",
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
    # hammer_z1_env/ paths are sibling-repo (never exist here) and are deliberately NOT matched;
    # bare script names (validate_rewards.py) are ambiguous and also unchecked.
    missing = []
    for src in (REPO / "CLAUDE.md", REPO / "docs/README.md"):
        if not src.exists():
            missing.append(f"{src} itself missing")
            continue
        text = src.read_text(encoding="utf-8")
        # backticked repo paths, optionally command-prefixed (`python scripts/x.py --flag`);
        # the path must end at a backtick or a space (flags/args may follow inside the span)
        for ref in re.findall(
            r"`(?:(?:mj)?python3? |pytest )?((?:docs|src|tests|scripts)/[^`\s]+?\.(?:md|py))[`\s]",
            text,
        ):
            if not (REPO / ref).exists():
                missing.append(f"{src.name} -> {ref}")
        # the three repo-root direction docs are referenced bare — check them too
        for ref in re.findall(r"`(thesis_[a-z_]+\.md)`", text):
            if not (REPO / ref).exists():
                missing.append(f"{src.name} -> {ref}")
    assert not missing, "Dangling doc references:\n" + "\n".join(missing)


def test_index_points_to_current_fic_baseline_and_vic_prototype_routes():
    text = (REPO / "docs/README.md").read_text(encoding="utf-8")
    required_refs = (
        "docs/superpowers/specs/2026-08-12-z1-direct-reference-fic-design.md",
        "docs/superpowers/plans/2026-08-12-z1-direct-reference-fic.md",
        "docs/superpowers/specs/2026-08-13-z1-native-vic-design.md",
        "docs/superpowers/plans/2026-08-13-z1-native-vic.md",
        "docs/superpowers/specs/2026-08-13-z1-vic-canary-design.md",
        "docs/superpowers/plans/2026-08-13-z1-vic-canary.md",
        "docs/results/2026-08-14_z1_vic_seed2_canary.md",
        "docs/results/2026-08-15_z1_impulse_cat_step1.md",
        "docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md",
        "docs/superpowers/specs/2026-08-15-z1-impulse-diag90-500-design.md",
        "docs/superpowers/plans/2026-08-15-z1-impulse-diag90-500.md",
        "docs/superpowers/specs/2026-08-17-z1-impulse-diag90-post-training-evaluation-design.md",
        "docs/superpowers/plans/2026-08-17-z1-impulse-diag90-post-training-evaluation.md",
        "docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md",
        "docs/superpowers/specs/2026-08-18-z1-impulse-cat-thesis-campaign-design.md",
        "docs/superpowers/plans/2026-08-18-z1-impulse-cat-p02-bridge.md",
        "docs/results/2026-08-18_z1_impulse_p02_bridge_preflight.md",
        "docs/results/2026-08-18_z1_impulse_p02_bridge.md",
        "docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_PACKET.md",
        "docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_REVIEWS.md",
        "docs/research/reward-design/Z1_IMPULSE_CAT_CAMPAIGN_CROSS_MODEL_PACKET.md",
        "docs/research/reward-design/Z1_IMPULSE_CAT_SIMPLIFIED_CROSS_MODEL_REVIEWS.md",
        "docs/thesis/decisions/2026-08-13_vic_impulse_cat_and_trajectory_direction.md",
        "src/tasks/hammer/mdp/variable_impedance.py",
        "scripts/slurm/vega_fic_direct_reference_smoke.sbatch",
        "scripts/slurm/vega_fic_direct_reference.sbatch",
        "scripts/slurm/vega_vic_canary.sbatch",
        "scripts/slurm/vega_vic_impulse_diag90_500.sbatch",
        "scripts/analyze_vic_impulse_diag90_500_evaluation.py",
        "scripts/preflight_vic_impulse_p02_bridge.py",
        "scripts/analyze_vic_impulse_p02_bridge_evaluation.py",
        "scripts/slurm/vega_vic_impulse_p02_bridge.sbatch",
        "scripts/slurm/vega_vic_impulse_p02_bridge_eval.sbatch",
        "evaluation/joint_position/evaluate_fic_pilot.py",
    )
    missing = [ref for ref in required_refs if f"`{ref}`" not in text]
    assert not missing, "Current FIC/VIC paths missing from index:\n" + "\n".join(missing)

    assert "# Docs index — current truth map (2026-08-18)" in text
    assert "the checked-out code wins" in text
    route = next(
        line for line in text.splitlines() if line.startswith("**Current GPU route:**")
    )
    assert "clean detached worktree" in route
    assert "VIC-TT qualification is banked and complete" in route
    assert "same nominal arm twice" in route
    assert "A100" in route
    assert "seed-2 engineering canary is also banked and complete" in route
    assert "Training job `41119011` completed 500 iterations" in route
    assert "one-seed engineering result, not a VIC-superiority claim" in route
    assert "impulse CaT was log-only" in route
    assert "user-authorized provisional-cap `p=0.2` bridge is complete" in route
    assert "bridge passed every preregistered gate in all three stochastic populations" in route
    assert "independent training-seed confirmation is not yet run or authorized" in route
    assert "Event-dose calibration is not complete" in route
    assert "redistribution/trade-off, not clean enforcement" in route
    assert "true 500 Hz velocity-limit violation risk increased" in route
    assert "lower native observed J3 impulse exposure" in route
    assert (
        "unequal native episode/contact horizons prevent a causal or full-contact interpretation"
        in route
    )
    assert "direct-reference FIC-0/FIC-TT campaign" not in route
    assert (
        "direct-reference FIC launchers are banked baseline/reproducibility routes"
        in route
    )
    assert (
        "lower *native observed* J3 impulse exposure"
        in text
    )
    assert (
        "The evaluation observed lower native J3 diagnostic violation risk"
        in text
    )
    assert (
        "`scripts/slurm/vega_fic_direct_reference_smoke.sbatch` "
        "(banked direct-reference FIC CUDA baseline/reproducibility)"
        in text
    )
    assert (
        "`scripts/slurm/vega_fic_direct_reference.sbatch` "
        "(banked direct-reference FIC training reproducibility)"
        in text
    )
    prototype = next(
        line
        for line in text.splitlines()
        if line.startswith("**Current VIC prototype route:**")
    )
    assert "implementation and qualification are banked" in prototype
    assert "one seed-2 engineering canary" in prototype
    assert "is complete and banked" in prototype
    assert "no additional seeds or formal FIC–VIC comparison" in prototype
    assert "training is deferred" not in text
    assert "training is explicitly deferred" not in text
    assert (
        "`scripts/slurm/vega_vic_canary.sbatch` "
        "(completed guarded one-seed VIC-TT engineering-canary route)"
        in text
    )


def test_vic_docs_disclose_action_dimension_learner_comparison_caveat():
    for relative_path in (
        "docs/superpowers/specs/2026-08-13-z1-native-vic-design.md",
        "docs/superpowers/plans/2026-08-13-z1-native-vic.md",
    ):
        text = (REPO / relative_path).read_text(encoding="utf-8")
        assert "entropy, log probability, and KL" in text
        assert "summed across action dimensions" in text
        assert "6D FIC" in text
        assert "12D VIC" in text
        assert "formal comparison" in text


def test_claude_md_preserves_direction_quote_and_clarifies_native_vic_path():
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")

    assert "policy commanding per-joint stiffness via `set_gains`" in text
    clarification = next(
        line
        for line in text.splitlines()
        if line.startswith("> **IMPLEMENTATION CLARIFICATION (2026-08-13):**")
    )
    assert "`BuiltinPositionActuator` has no literal `set_gains()` API" in clarification
    assert "`actuator_gainprm`" in clarification
    assert "`actuator_biasprm`" in clarification
    assert "prototype implementation and qualification" in clarification
    assert "not a training or results claim" in clarification


def test_claude_md_records_the_live_nail_driven_weight():
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`nail_driven=0.5`" in text
    assert "`nail_driven=2.0`" not in text


def test_diag90_result_keeps_observed_prefix_and_causal_claims_separate():
    text = (
        REPO / "docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md"
    ).read_text(encoding="utf-8")
    assert "shorter target episodes therefore do not explain" not in text
    assert "does not resolve the unequal-horizon/contact-censoring confound" in text
    assert "native observed endpoint remains valid" in text
    assert "shadow is required for a full-contact or causal interpretation" in text
    assert "Soft CaT shapes learned behavior" not in text


def test_contact_flush_plan_uses_current_rolling_lambda_without_a_second_window():
    text = (
        REPO
        / "docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md"
    ).read_text(encoding="utf-8")
    assert "fixed 25-sample Boolean ring" not in text
    assert "current rolling Lambda is zero" in text
    assert "positive Lambda sample, then exactly 24" in text
    assert "25th off-contact sample" in text
    assert "no second 25-sample window" in text
    result_text = (
        REPO / "docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md"
    ).read_text(encoding="utf-8")
    assert "current rolling Lambda is zero (no second window)" in result_text


def test_diag90_result_names_pressure_gain_and_provenance_claim_boundaries():
    text = (
        REPO / "docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md"
    ).read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "unique physical-event dose is not banked" in text
    assert (
        "activation-window pressure classified by associated physical-event status"
        in text
    )
    assert "impulse-versus-velocity attribution and activation-window pressure" in text
    assert "event-pressure quantiles" not in text
    assert "precontact J4 median" in text
    assert "control `p=+1, Kp=1250, Kd=111.80`" in text
    assert "target `p=-1, Kp=800, Kd=89.44`" in text
    assert "stochastic precontact means also differ at J1, J4, and J5" in text
    assert "checkpoint binaries were not downloaded or independently rehashed" in text
    assert "latent action-noise equality is a procedural RNG contract" in flat
    assert "scheduler status comes from the Task-8 execution record" in flat
    assert "not derived from trace contents" in flat
    assert "Analysis-consumed array alignments" in flat
    assert "delta_velocity, offline delta_impulse, and their exact max" in text


def test_p02_bridge_result_and_review_packet_preserve_the_screening_claim():
    result_path = REPO / "docs/results/2026-08-18_z1_impulse_p02_bridge.md"
    packet_path = (
        REPO
        / "docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_PACKET.md"
    )
    reviews_path = (
        REPO
        / "docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_REVIEWS.md"
    )
    analysis_path = (
        REPO
        / "docs/results/assets/2026-08-18_z1_impulse_p02_bridge/analysis.json"
    )
    manifest_path = analysis_path.parent / "SHA256SUMS"

    result = result_path.read_text(encoding="utf-8")
    flat_result = " ".join(result.split())
    assert "**Verdict: bridge PASS**" in result
    assert "all three stochastic populations passed every preregistered gate separately" in result
    assert "one PPO training seed" in result
    for boundary in (
        "not a hard clamp",
        "not a hardware-safety result",
        "not evidence that `p=0.2` is optimal",
        "not proof across training seeds",
        "native observed exposure",
    ):
        assert boundary in flat_result
    for seed in ("`2`", "`2026081701`", "`2026081702`"):
        assert seed in result

    packet = packet_path.read_text(encoding="utf-8")
    assert "VERDICT: CONFIRM | STOP | REVISE" in packet
    assert "Do the preregistered results justify matched independent-seed confirmation" in packet
    assert "No pooled gate" in packet
    for forbidden in ("/Users/", "/private/", "/ceph/", "eunikhilr", "nikerane"):
        assert forbidden not in packet
    for seed in ("`2`", "`2026081701`", "`2026081702`"):
        assert seed in packet

    reviews = reviews_path.read_text(encoding="utf-8")
    flat_reviews = " ".join(reviews.split())
    assert "External review status: complete" in reviews
    assert "Five initial one-shot attempts" in flat_reviews
    assert "Two user-authorized single corrected attempts" in flat_reviews
    assert "Kimi K3" in reviews and "original attempt failed before inference" in reviews
    assert "GLM-5.2" in reviews and "original attempt failed before inference" in reviews
    for session_id in (
        "ses_feba5cbe1ffekvYzZOXyvSXDix",
        "ses_feba3f003ffekOBTJjyglZ79o1",
        "ses_febacb884ffeHpIAmgnOoybzXb",
        "ses_febac38e0ffefKkxZE6AEEwb7I",
        "ses_febaba3bcffelKhCypldwCohwf",
    ):
        assert session_id in reviews
    for response_hash in (
        "de852428883961a84674ad703548fdc864fe690aad732ca30ff7d53fa4a7718d",
        "65ecac553b87dd6f3ee30c80813b4c617111260c1d15d8d7cfeb84eea8c10992",
        "fa0cc4994a88f6ef0978c88e83d59444c06e1e1f2bbd36f2235b4de392064ed1",
        "ddd8aa9a70f2a70861555a886d37b4b5648c8e5017836480910f7c731f715f5c",
        "e4bf976dce79e5a01e8a29625ec95921278149211b179800478ed662c9557b16",
    ):
        assert response_hash in reviews
    assert reviews.count("CONFIRM") >= 5
    assert "unknown unit and currency" in flat_reviews
    assert "preregistered numerical PASS controls the scientific verdict" in flat_reviews
    assert "single PPO training seed" in flat_reviews
    assert "complete-contact or actuator-loading claim" in flat_reviews
    assert "paired newly trained controls" in flat_reviews
    assert "not automatically authorized" in flat_reviews

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert tuple(analysis["training_like_by_seed"]) == ("2", "2026081701", "2026081702")
    assert all(
        population["verdict"]["pass"] is True
        for population in analysis["training_like_by_seed"].values()
    )

    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest = dict(row.split("  ", 1)[::-1] for row in rows)
    for name in (
        "analysis.json",
        "../../../../scripts/analyze_vic_impulse_p02_bridge_evaluation.py",
        "../../../research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_PACKET.md",
        "../../../research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_REVIEWS.md",
    ):
        path = analysis_path.parent / name
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_index_routes_to_the_completed_p02_bridge_without_overclaiming():
    text = (REPO / "docs/README.md").read_text(encoding="utf-8")
    assert "`docs/results/2026-08-18_z1_impulse_p02_bridge.md`" in text
    assert (
        "`docs/research/reward-design/Z1_IMPULSE_CAT_P02_BRIDGE_CROSS_MODEL_PACKET.md`"
        in text
    )
    route = next(
        line for line in text.splitlines() if line.startswith("**Current GPU route:**")
    )
    assert "bridge passed every preregistered gate in all three stochastic populations" in route
    assert "independent training-seed confirmation is not yet run or authorized" in route
    assert "one target training run and frozen matched evaluation remain" not in route
    assert "Five successful external reviews recommend confirmation" in route
    assert "two initial attempts failed before inference" in route
    assert "user-authorized corrected attempts succeeded" in route

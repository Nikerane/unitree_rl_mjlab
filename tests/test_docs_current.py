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
import math
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
    assert "provisional-cap compact dose screen is complete" in route
    assert "p=.2 is the strongest tested policy instance but not a formal PASS" in route
    assert "Independent training-seed confirmation remains unrun" in route
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
    assert "p=.2 is the strongest tested policy instance but not a formal PASS" in route
    assert "Independent training-seed confirmation remains unrun" in route
    assert "one target training run and frozen matched evaluation remain" not in route
    assert "not an optimal-dose" in route


def test_dose_curve_result_packet_and_analysis_are_exact_and_bounded():
    result_path = REPO / "docs/results/2026-08-18_z1_impulse_cat_dose_curve.md"
    asset_dir = REPO / "docs/results/assets/2026-08-18_z1_impulse_cat_dose_curve"
    analysis_path = asset_dir / "analysis.json"
    manifest_path = asset_dir / "SHA256SUMS"
    packet_path = (
        REPO
        / "docs/research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_REVIEW_PACKET.md"
    )

    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == (
        "8289a3c0247e086ae244c70bbbddb0820cf6390543ee61a90d6d71026da24135"
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    numeric_values = []

    def collect_numbers(value):
        if isinstance(value, dict):
            for nested in value.values():
                collect_numbers(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_numbers(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_values.append(value)

    collect_numbers(analysis)
    assert all(math.isfinite(value) for value in numeric_values)
    assert analysis["provenance"]["code_revision"] == (
        "9718a7958cbd7ec6b2f3e126f8fbae8ef81762df"
    )
    assert analysis["provenance"]["asset_revision"] == (
        "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
    )
    assert analysis["protocol_identity"]["live_imp_max_p"] == 0.0
    assert analysis["protocol_identity"]["provisional_caps_n_m_s"] == [
        0.82,
        1.64,
        0.82,
        0.82,
        0.82,
        0.82,
    ]
    assert analysis["protocol_identity"]["stochastic_seeds"] == [
        2,
        2026081701,
        2026081702,
    ]
    expected_population_identity = {
        "2": (
            "5a7dd99270580d0d71b10fcffe70aff6186e8f3b57987cae6f5521f0639c9cc6",
            {"reset": 10000021, "observation": 20000035, "action": 30000043},
        ),
        "2026081701": (
            "5a7dd99270580d0d71b10fcffe70aff6186e8f3b57987cae6f5521f0639c9cc6",
            {"reset": 2036081720, "observation": 2046081734, "action": 2056081742},
        ),
        "2026081702": (
            "5a7dd99270580d0d71b10fcffe70aff6186e8f3b57987cae6f5521f0639c9cc6",
            {"reset": 2036081721, "observation": 2046081735, "action": 2056081743},
        ),
    }
    for seed, (population_hash, rng_streams) in expected_population_identity.items():
        identity = analysis["protocol_identity"]["populations"]["training_like_by_seed"][seed]
        assert identity["initial_population_sha256"] == population_hash
        assert identity["rng_streams"] == rng_streams
    assert tuple(analysis["dose_results"]) == (
        "dose_p01_target",
        "dose_p02_target",
        "dose_p03_target",
    )
    expected_verdicts = {
        "0.1": (False, False, False),
        "0.2": (True, False, True),
        "0.3": (False, False, False),
    }
    for dose, expected in expected_verdicts.items():
        role = {"0.1": "dose_p01_target", "0.2": "dose_p02_target", "0.3": "dose_p03_target"}[dose]
        dose_result = analysis["dose_results"][role]
        populations = dose_result["training_like_by_seed"]
        assert tuple(populations) == ("2", "2026081701", "2026081702")
        assert tuple(pop["verdict"]["pass"] for pop in populations.values()) == expected
        assert dose_result["overall_verdict"]["pass"] is False

    result = result_path.read_text(encoding="utf-8")
    flat_result = " ".join(result.split())
    for exact_row in (
        "| .1 | 2 | 0.8545->0.3418 (-0.5127; -0.1709), F | .8067->.8161 / .9735->.9485, F | 0.5127->0.7324 (+0.2197; +0.4883), F | 0/0; .9827 (.9817), P |",
        "| .1 | 2026081701 | 0.7324->0.2686 (-0.4639; -0.1709), F | .8070->.8108 / .9071->.9383, F | 0.6836->0.6104 (-0.0732; +0.2197), F | censored initial fragment, F |",
        "| .1 | 2026081702 | 0.8545->0.2197 (-0.6348; -0.3418), F | .8069->.8097 / .9870->.9418, F | 0.6592->1.1719 (+0.5127; +0.8301), F | 0/0; .9834 (.9825), P |",
        "| .2 | 2 | 0.8545->0 (-0.8545; -0.5859), P | .8067->.5158 / .9735->.6434, P | 0.5127->0.1709 (-0.3418; -0.1221), P | 0/0; .9954 (.9948), P |",
        "| .2 | 2026081701 | 0.7324->0 (-0.7324; -0.4883), **F** | .8070->.4559 / .9071->.6361, P | 0.6836->0.1953 (-0.4883; -0.2441), P | 0/0; .9964 (.9957), P |",
        "| .2 | 2026081702 | 0.8545->0 (-0.8545; -0.5859), P | .8069->.5208 / .9870->.6335, P | 0.6592->0.2441 (-0.4150; -0.1465), P | 0/0; .9955 (.9950), P |",
        "| .3 | 2 | 0.8545->0.0732 (-0.7812; -0.4883), F | .8067->.7877 / .9735->.8787, F | 0.5127->0.6104 (+0.0977; +0.3662), F | censored initial fragment, F |",
        "| .3 | 2026081701 | 0.7324->0.0488 (-0.6836; -0.4150), F | .8070->.7866 / .9071->.8753, F | 0.6836->0.6348 (-0.0488; +0.2197), F | 0/0; .9740 (.9729), P |",
        "| .3 | 2026081702 | 0.8545->0.0977 (-0.7568; -0.4639), F | .8069->.7866 / .9870->.8842, F | 0.6592->0.9033 (+0.2441; +0.5371), F | censored initial fragment, F |",
    ):
        assert exact_row in result
    for fact in (
        "p=.1 is too weak",
        "p=.3 is not monotonically better",
        "p=.2 is the strongest tested policy instance",
        "not a formal PASS",
        "identical declared initial-population hashes and RNG stream IDs",
        "control environment `3253`",
        "`0.8731314`",
        "`+0.0531314`",
        "`0.54809135`",
        "`-0.27190864`",
        "`-0.005126953125`",
        "`-0.0048828125`",
        "`0.0001171875`",
        "native-float margin",
        "array `41504493_0`",
        "array `41513714`",
        "00:25:31",
        "0.845 allocated A100 GPU-hours",
        "0b24cab9bcc59e1507b2ca4f3ccdcf63d1bf58901b03a263a7bb1b8cee60371c",
    ):
        assert fact in flat_result
    for boundary in (
        "one training seed",
        "native observed horizon",
        "provisional project caps",
        "soft pressure, not a clamp",
        "no manufacturer or hardware-safety claim",
        "no complete-contact claim",
        "no optimal-dose claim",
        "evaluation replicas are not training seeds",
    ):
        assert boundary in flat_result

    packet = packet_path.read_text(encoding="utf-8")
    flat_packet = " ".join(packet.split())
    assert len(re.findall(r"\b[\w.-]+\b", packet)) <= 700
    assert "moving to independent training-seed confirmation of `p=0.2`" in flat_packet
    assert "rather than increasing dose" in flat_packet
    for forbidden in (
        "/Users/",
        "/private/",
        "/ceph/",
        "eunikhilr",
        "nikerane",
        "model_499.pt",
        "analysis-41513714.json",
        "f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3",
        "9718a7958cbd7ec6b2f3e126f8fbae8ef81762df",
        ".npz",
        "summary.json",
    ):
        assert forbidden not in packet
    assert re.search(r"\b[0-9a-f]{40,64}\b", packet) is None

    manifest = dict(
        row.split("  ", 1)[::-1]
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
    )
    for name in (
        "analysis.json",
        "../../../../scripts/analyze_vic_impulse_dose_curve.py",
        "../../2026-08-18_z1_impulse_cat_dose_curve.md",
        "../../../research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_REVIEW_PACKET.md",
    ):
        assert manifest[name] == hashlib.sha256((asset_dir / name).read_bytes()).hexdigest()


def test_current_docs_route_to_dose_curve_without_overclaiming():
    index = (REPO / "docs/README.md").read_text(encoding="utf-8")
    thesis = (REPO / "docs/thesis/README.md").read_text(encoding="utf-8")
    result = (
        REPO / "docs/results/2026-08-18_z1_impulse_cat_dose_curve.md"
    ).read_text(encoding="utf-8")
    results_index = (REPO / "docs/results/README.md").read_text(encoding="utf-8")
    for text in (index, thesis):
        flat_text = " ".join(text.split())
        assert "p=.1 is too weak" in flat_text
        assert "p=.3 is not monotonically better" in flat_text
        assert "p=.2 is the strongest tested policy instance" in flat_text
        assert "not a formal PASS" in flat_text
        assert "independent training-seed confirmation" in flat_text.lower()
        assert "not an optimal-dose" in flat_text
        assert "unapproved replay-only 10-repeat execution-nondeterminism envelope" in flat_text
    assert "`docs/results/2026-08-18_z1_impulse_cat_dose_curve.md`" in index
    assert (
        "`docs/research/reward-design/"
        "Z1_IMPULSE_CAT_DOSE_CURVE_CROSS_MODEL_REVIEWS.md`"
        in index
    )
    assert "`../results/2026-08-18_z1_impulse_cat_dose_curve.md`" in thesis
    assert (
        "[2026-08-18_z1_impulse_cat_dose_curve.md]"
        "(2026-08-18_z1_impulse_cat_dose_curve.md)"
        in results_index
    )
    flat_result = " ".join(result.split())
    assert "unapproved replay-only 10-repeat execution-nondeterminism envelope" in flat_result
    assert "Neither replay nor training is authorized" in flat_result
    assert "The next scientific question is independent training-seed confirmation" not in result
    assert (
        "Independent training-seed confirmation remains unrun and is the bounded next question"
        not in index
    )
    assert "Independent training-seed confirmation is the bounded next question" not in index


def test_dose_curve_cross_model_reviews_are_bounded_and_manifested():
    result_path = REPO / "docs/results/2026-08-18_z1_impulse_cat_dose_curve.md"
    packet_path = (
        REPO
        / "docs/research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_REVIEW_PACKET.md"
    )
    reviews_path = (
        REPO
        / "docs/research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_CROSS_MODEL_REVIEWS.md"
    )
    manifest_path = (
        REPO
        / "docs/results/assets/2026-08-18_z1_impulse_cat_dose_curve/SHA256SUMS"
    )

    assert hashlib.sha256(packet_path.read_bytes()).hexdigest() == (
        "485bf07d472d94c3a200721b72732ec2735ba1fea02219619fac80cb0a51a6cd"
    )
    assert len(packet_path.read_bytes()) == 3278

    reviews = reviews_path.read_text(encoding="utf-8")
    flat_reviews = " ".join(reviews.split())
    expected_reviews = (
        (
            "opencode/kimi-k3",
            "ses_fea78634effejHlbQaCDimILyc",
            "CONFIRM",
            "e52485ddd075716b015e79c0d9a6dd5417b284eda3c8f59b4fa3ba4bb9f5ae6a",
            "11557 / 1534 / 0 / 20139 / 0",
            "0.0637227",
        ),
        (
            "opencode/glm-5.2",
            "ses_fea778cc1ffeHj1ZmfxvMgTY8v",
            "CONFIRM",
            "38ccfef1095d37ecda28dcfafea547a36b076821e351a894795f6054e661daea",
            "9911 / 3587 / 0 / 152 / 0",
            "0.02969772",
        ),
        (
            "opencode/qwen3.6-plus",
            "ses_fea763a7effet5fEug5xht5c0P",
            "STOP",
            "ae588f1eb460c9cea5755ceffc7b633a37eafbc34828d060d712d1aac27540b0",
            "6 / 713 / 0 / 0 / 10609",
            "0.008772625",
        ),
        (
            "opencode/deepseek-v4-pro",
            "ses_fea75787affeRP4SV1D9ZYolmI",
            "CONFIRM",
            "85b40e5d7f61485af08f6dad8225f909bac2713c9c4bb3cec672831498601b25",
            "10435 / 909 / 0 / 0 / 0",
            "0.02164746",
        ),
        (
            "opencode/claude-opus-5",
            "ses_fea7501f6ffeWDX31EMku6uNAH",
            "REVISE",
            "d91c641923f55f2a88d1cfad57e2a354866dc40352e66bb8c4d7a7a92a6698bb",
            "4 / 6748 / 0 / 14995 / 20176",
            "0.3023175",
        ),
    )
    for model, session, verdict, text_hash, tokens, cost in expected_reviews:
        expected_row = (
            f"| `{model}` | max | `{session}` | `{verdict}` | `{text_hash}` | "
            f"{tokens} | {cost} |"
        )
        assert expected_row in reviews
    for fact in (
        "0.426158005",
        "485bf07d472d94c3a200721b72732ec2735ba1fea02219619fac80cb0a51a6cd",
        "86e1f9e8cc8a96b535e0e3267abce2a78ea1ecd7a53652a23116fadf3f4e6c02",
        "f7ba41a559af3cd73f2fbf0e5cc1c00b983a56c0f9225e31623750c030a7f548",
        "literal opening and closing quote",
        "terminal newline",
        "two pre-launch authorization blocks",
        "no process, session, inference, or cost",
        "exactly one successful inference attempt per model",
        "one inference attempt per model and no inference retry",
        "Qwen's `0.000117 N.m.s` unit is wrong",
        "risk-difference fraction",
        "DeepSeek's optimum claim is unsupported",
        "fresh-RNG rescue",
        "3,278 bytes",
        "`-21/4096`",
        "`-20/4096`",
        "`-20.48/4096`",
        "replay-only execution-nondeterminism envelope",
    ):
        assert fact in flat_reviews
    for model in (
        "opencode/kimi-k3",
        "opencode/glm-5.2",
        "opencode/qwen3.6-plus",
        "opencode/deepseek-v4-pro",
        "opencode/claude-opus-5",
    ):
        assert f"`{model}`" in reviews
    for boundary in (
        "advisory",
        "unknown unit and currency",
        "one PPO training seed",
        "cannot turn numerical FAIL into PASS",
        "no new training",
        "All five treated p=.2 as the strongest tested policy instance",
        "DeepSeek explicitly overcalled it an optimum",
    ):
        assert boundary in flat_reviews

    result = result_path.read_text(encoding="utf-8")
    flat_result = " ".join(result.split())
    assert "Consultation and adjudication" in result
    assert "every tested dose remains a preregistered numerical FAIL" in flat_result
    assert "independent training-seed confirmation of `p=.2`" in flat_result
    assert "not authorized" in flat_result

    manifest = dict(
        row.split("  ", 1)[::-1]
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
    )
    for name in (
        "analysis.json",
        "../../../../scripts/analyze_vic_impulse_dose_curve.py",
        "../../2026-08-18_z1_impulse_cat_dose_curve.md",
        "../../../research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_REVIEW_PACKET.md",
        "../../../research/reward-design/Z1_IMPULSE_CAT_DOSE_CURVE_CROSS_MODEL_REVIEWS.md",
    ):
        assert manifest[name] == hashlib.sha256(
            (manifest_path.parent / name).read_bytes()
        ).hexdigest()

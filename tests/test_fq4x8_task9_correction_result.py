"""Integrity checks for the locally corrected fq4x8 Task-9 result."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "docs/results/assets/2026-08-14_fq4x8_task9_ratio_correction"
RESULT_RECORD = ROOT / "docs/results/2026-08-14_fq4x8_task9_ratio_correction.md"
RESULT_INDEX = ROOT / "docs/results/README.md"

SOURCE_ANALYSIS_SHA256 = (
    "723b3197948338e259add4da21747982ee72445cac65c8a475185be5482e7382"
)
CONTACT_MAP_SHA256 = (
    "40fcf66715b630f969c064e22ac8ad84ed1a095a765620357017a5b80faf7bbf"
)
PACKAGE_FILES = {
    "aggregate_nail_plane_contact_map.png",
    "analysis.json.gz",
    "paired_seed_effects.png",
    "provenance.json",
    "source_analysis.json.gz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_task9_correction_bank_has_exact_hash_verified_inventory() -> None:
    manifest = PACKAGE / "SHA256SUMS"
    package_entries = list(PACKAGE.rglob("*"))
    actual_files = {
        path.relative_to(PACKAGE).as_posix()
        for path in package_entries
        if path.is_file()
    }
    assert actual_files == PACKAGE_FILES | {"SHA256SUMS"}
    assert not any(path.is_symlink() for path in package_entries)

    rows = manifest.read_text(encoding="utf-8").splitlines()
    parsed = {}
    for row in rows:
        digest, relative_path = row.split("  ", 1)
        assert len(digest) == 64
        assert relative_path not in parsed
        parsed[relative_path] = digest

    assert list(parsed) == sorted(PACKAGE_FILES)
    assert set(parsed) == PACKAGE_FILES
    for relative_path, digest in parsed.items():
        path = PACKAGE / relative_path
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path) == digest
    assert parsed["aggregate_nail_plane_contact_map.png"] == CONTACT_MAP_SHA256


def test_task9_correction_recomputes_only_the_registered_ratios() -> None:
    with gzip.open(PACKAGE / "analysis.json.gz", "rt", encoding="utf-8") as handle:
        analysis = json.load(handle)
    provenance = json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["source"]["analysis_sha256"] == SOURCE_ANALYSIS_SHA256
    assert analysis["valid"] is True
    assert len(analysis["seed_aggregates"]) == 32
    assert len(analysis["valid_contact_coordinates"]) == 14_846

    fq = analysis["fq_min_practical_acceptance"]
    assert fq["useful_speed_ratio"]["estimate"] == 1.0204828748706622
    assert fq["useful_speed_ratio"]["one_sided_95_lower"] == 0.9031426891189621
    assert fq["depth_gain_ratio"]["estimate"] == 1.1304631795629263
    assert fq["depth_gain_ratio"]["one_sided_95_lower"] == 1.0036288116342884
    assert fq["gates"]["useful_speed_ratio_lower_gt_0_95"] is False
    assert fq["gates"]["depth_gain_ratio_lower_gt_0_90"] is True
    assert fq["passed"] is False

    d0 = analysis["d0_mechanism"]
    assert d0["depth_gain_ratio"]["estimate"] == 0.9914616558752575
    assert d0["depth_gain_ratio"]["one_sided_95_lower"] == 0.9886613795645048
    assert d0["conditions"]["depth_gain_ratio_lower_gt_0_90"] is True
    assert d0["claim_allowed"] is False
    assert provenance["results"]["overall_conclusions_changed"] is False


def test_task9_correction_changes_exactly_the_declared_source_leaves() -> None:
    source_bytes = gzip.decompress((PACKAGE / "source_analysis.json.gz").read_bytes())
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_ANALYSIS_SHA256
    source = json.loads(source_bytes)
    with gzip.open(PACKAGE / "analysis.json.gz", "rt", encoding="utf-8") as handle:
        corrected = json.load(handle)
    provenance = json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))

    def leaf_paths(value, prefix: str) -> set[str]:
        if isinstance(value, dict):
            return {
                path
                for key, child in value.items()
                for path in leaf_paths(child, f"{prefix}.{key}" if prefix else key)
            }
        return {prefix}

    def changed_leaves(before, after, prefix="") -> set[str]:
        if isinstance(before, dict) and isinstance(after, dict):
            changed = set()
            for key in before.keys() | after.keys():
                child_prefix = f"{prefix}.{key}" if prefix else key
                if key in before and key in after:
                    changed.update(
                        changed_leaves(before[key], after[key], child_prefix)
                    )
                else:
                    changed.update(
                        leaf_paths(before.get(key, after.get(key)), child_prefix)
                    )
            return changed
        return {prefix} if before != after else set()

    assert changed_leaves(source, corrected) == set(
        provenance["correction"]["changed_leaf_paths"]
    )


def test_task9_correction_provenance_scopes_both_plot_runtimes() -> None:
    provenance = json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["correction_runtime"]["matplotlib"] == "3.10.9"
    historical_map = provenance["copied_historical_contact_map"]
    assert historical_map["sha256"] == CONTACT_MAP_SHA256
    assert historical_map["producer_runtime"]["matplotlib"] == "3.11.0"
    assert historical_map["copied_byte_for_byte"] is True


def test_task9_correction_result_is_indexed_and_preserves_the_decisions() -> None:
    record = RESULT_RECORD.read_text(encoding="utf-8")
    assert "FQ-min practical replacement remains rejected" in record
    assert "D0 mechanism claim remains rejected" in record
    assert "No Vega rerun was needed" in record

    index = RESULT_INDEX.read_text(encoding="utf-8")
    assert "2026-08-14_fq4x8_task9_ratio_correction.md" in index

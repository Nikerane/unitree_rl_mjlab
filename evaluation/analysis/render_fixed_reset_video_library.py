"""Freeze and render the exact 56-policy fixed-reset qualitative library.

The accepted-training manifests are the sole source of policy membership.  This
driver deliberately does no outcome selection: every registered arm/seed is
rendered once from the shared Stage-0 reset or the run fails closed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import tyro

from evaluation.analysis.fixed_reset_video_library import (
    ARTIFACT_FILENAMES,
    EXPECTED,
    FIXED_RESET_ENVELOPE,
    _metadata_digest,
    _readable_artifact,
    _sha256,
    _validate_media_properties,
    _validate_metadata_contract,
    _validate_trace_rollout,
    expected_task,
    load_fixed_reset,
    policy_artifact_dir,
    validate_inventory,
    validate_policy_artifacts,
    write_metadata,
)


OUTPUT_ROOT = Path("docs/results/assets/2026-07-29_56_policy_fixed_reset_library")
EXPECTED_MANIFEST_SHA256 = {
    "fq4x8": "fb55f214d6e0cb2da308e6580ef535d4823038bc8ab842a05ca4085ab346ec14",
    "fq3x8": "4692e71bf2f6d201a765f3d4cdb3646fbe37593074965c9efa8a5c6ae29d8645",
}
INVENTORY_FIELDS = (
    "campaign",
    "arm",
    "short",
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_file",
    "checkpoint_sha256",
    "training_code_revision",
    "training_asset_revision",
    "accepted_training_manifest_sha256",
)


def _sha256_text(path: Path) -> str:
    return _sha256(path)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _require_revision(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid {field} revision")
    return text


def _read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("missing TSV header")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read accepted training manifest: {path}") from error
    if not rows or any(None in row for row in rows):
        raise ValueError(f"malformed accepted training manifest: {path}")
    return rows


def _manifest_training_revisions(campaign: str, raw: Mapping[str, str]) -> tuple[str, str]:
    if campaign == "fq4x8":
        if raw.get("clean_state", "").lower() != "true":
            raise ValueError("manifest records inherited dirty training provenance")
        code = raw.get("code_revision", "")
        asset = raw.get("asset_revision", "")
    elif campaign == "fq3x8":
        code = raw.get("training_code_revision", "")
        asset = raw.get("training_asset_revision", "")
    else:  # membership validation below keeps this defensive branch explicit.
        raise ValueError(f"unexpected campaign: {campaign}")
    return (
        _require_revision(code, field="training code"),
        _require_revision(asset, field="training asset"),
    )


def _inventory_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    campaigns = tuple(EXPECTED)
    campaign = str(row["campaign"])
    arms = tuple(EXPECTED[campaign])
    return campaigns.index(campaign), arms.index(str(row["arm"])), int(row["training_seed"])


def _checkpoint_file(root: Path, checkpoint_path: str) -> Path:
    original = Path(checkpoint_path)
    if original.name != "model_499.pt":
        raise ValueError(f"accepted checkpoint is not model_499.pt: {checkpoint_path}")
    # The copied cache preserves the manifest run-directory basename, while the
    # original absolute path is retained verbatim as immutable training provenance.
    return root / original.parent.name / original.name


def build_inventory(
    fq4_manifest: str | Path,
    fq3_manifest: str | Path,
    checkpoint_roots: Mapping[str, str | Path],
    *,
    expected_manifest_sha256: Mapping[str, str] = EXPECTED_MANIFEST_SHA256,
) -> list[dict[str, Any]]:
    """Build the exact inventory after validating manifests and transferred bytes."""
    manifests = {"fq4x8": Path(fq4_manifest), "fq3x8": Path(fq3_manifest)}
    if set(checkpoint_roots) != set(manifests):
        raise ValueError("checkpoint roots must name exactly fq4x8 and fq3x8")
    rows: list[dict[str, Any]] = []
    for campaign, manifest_path in manifests.items():
        actual_manifest_sha256 = _sha256_text(manifest_path)
        expected_manifest_digest = _require_sha256(
            expected_manifest_sha256.get(campaign, ""), field=f"{campaign} manifest"
        )
        if actual_manifest_sha256 != expected_manifest_digest:
            raise ValueError(f"{campaign} accepted manifest SHA-256 mismatch")
        for raw in _read_tsv(manifest_path):
            required = ("campaign", "disposition", "arm", "short", "task", "training_seed", "checkpoint_path", "checkpoint_sha256")
            if any(not raw.get(field) for field in required):
                raise ValueError(f"manifest row is missing required identity fields: {campaign}")
            if raw["campaign"] != campaign:
                raise ValueError(f"manifest campaign mismatch: {raw['campaign']}")
            if raw["disposition"] != "accepted":
                raise ValueError(f"manifest disposition is not accepted: {campaign}")
            arm = raw["arm"]
            if arm not in EXPECTED[campaign]:
                raise ValueError(f"unexpected arm in accepted manifest: {campaign}/{arm}")
            try:
                training_seed = int(raw["training_seed"])
            except ValueError as error:
                raise ValueError(f"invalid training seed: {raw['training_seed']}") from error
            task = expected_task(campaign, arm)
            if raw["task"] != task:
                raise ValueError(f"manifest task mismatch: {campaign}/{arm}")
            training_code_revision, training_asset_revision = _manifest_training_revisions(
                campaign, raw
            )
            checkpoint_sha256 = _require_sha256(raw["checkpoint_sha256"], field="checkpoint")
            checkpoint_file = _checkpoint_file(Path(checkpoint_roots[campaign]), raw["checkpoint_path"])
            if not checkpoint_file.is_file():
                raise ValueError(f"accepted checkpoint is missing: {checkpoint_file}")
            if _sha256_text(checkpoint_file) != checkpoint_sha256:
                raise ValueError(f"accepted checkpoint SHA-256 mismatch: {checkpoint_file}")
            rows.append(
                {
                    "campaign": campaign,
                    "arm": arm,
                    "short": raw["short"],
                    "task": task,
                    "training_seed": training_seed,
                    "checkpoint_path": raw["checkpoint_path"],
                    "checkpoint_file": str(checkpoint_file),
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_code_revision": training_code_revision,
                    "training_asset_revision": training_asset_revision,
                    "accepted_training_manifest_sha256": actual_manifest_sha256,
                }
            )
    rows.sort(key=_inventory_sort_key)
    validate_inventory(rows)
    if len({row["checkpoint_path"] for row in rows}) != len(rows):
        raise ValueError("accepted manifest has duplicate checkpoint paths")
    if len({row["checkpoint_sha256"] for row in rows}) != len(rows):
        raise ValueError("accepted manifest has duplicate checkpoint hashes")
    return rows


def write_inventory(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write a deterministic TSV and its SHA-256 sidecar."""
    validate_inventory(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=_inventory_sort_key):
            writer.writerow({field: row[field] for field in INVENTORY_FIELDS})
    digest = _sha256_text(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def _renderer_revision() -> str:
    source_paths = (
        "scripts/render_policy.py",
        "evaluation/analysis/fixed_reset_video_library.py",
        "evaluation/analysis/render_fixed_reset_video_library.py",
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *source_paths], check=False
    )
    if dirty.returncode != 0:
        raise ValueError("renderer tracked source is dirty; commit it before rendering")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return _require_revision(revision, field="renderer code")


def _policy_artifact_is_valid(
    output_root: Path,
    row: Mapping[str, Any],
    fixed_reset: Mapping[str, Any],
    *,
    renderer_code_revision: str,
    renderer_asset_revision: str,
) -> bool:
    """Check one complete policy leaf before allowing a resume skip."""
    leaf = policy_artifact_dir(output_root, row)
    try:
        metadata = json.loads((leaf / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            return False
        for field in ("campaign", "arm", "training_seed", "checkpoint_sha256"):
            if str(metadata.get(field)) != str(row[field]):
                return False
        if metadata.get("reset_state_digest") != fixed_reset["reset_state_digest"]:
            return False
        if metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
            return False
        _validate_metadata_contract(metadata, row, leaf)
        if metadata.get("code_revision") != renderer_code_revision:
            return False
        if metadata.get("asset_revision") != renderer_asset_revision:
            return False
        training = metadata.get("training_provenance")
        if not isinstance(training, Mapping) or any(
            training.get(key) != row[key]
            for key in (
                "checkpoint_path",
                "training_code_revision",
                "training_asset_revision",
                "accepted_training_manifest_sha256",
            )
        ):
            return False
        for filename in ARTIFACT_FILENAMES:
            artifact = leaf / filename
            _readable_artifact(artifact)
            if metadata.get("artifacts", {}).get(filename) != _sha256_text(artifact):
                return False
        _validate_trace_rollout(leaf / "trace.npz", metadata["rollout"])
        _validate_media_properties(leaf, metadata["rollout"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def run_single_policy_renderer(
    row: Mapping[str, Any],
    fixed_reset: Mapping[str, Any],
    output_root: str | Path,
    *,
    renderer_code_revision: str,
    renderer_asset_revision: str,
    device: str = "cpu",
) -> bool:
    """Render one policy, returning True only when a fully validated leaf was reused."""
    output_root = Path(output_root)
    if _policy_artifact_is_valid(
        output_root,
        row,
        fixed_reset,
        renderer_code_revision=renderer_code_revision,
        renderer_asset_revision=renderer_asset_revision,
    ):
        return True
    leaf = policy_artifact_dir(output_root, row)
    leaf.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/render_policy.py",
        "--checkpoint-file", row["checkpoint_file"],
        "--campaign", row["campaign"],
        "--arm", row["arm"],
        "--training-seed", str(row["training_seed"]),
        "--checkpoint-sha256", row["checkpoint_sha256"],
        "--code-revision", renderer_code_revision,
        "--asset-revision", renderer_asset_revision,
        "--task", row["task"],
        "--out-dir", str(leaf),
        "--fixed-reset-envelope", str(FIXED_RESET_ENVELOPE),
        "--metadata-provenance", "fixed-reset 56-policy library; CPU single-env reinference",
        "--device", device,
    ]
    subprocess.run(command, check=True)
    metadata_path = leaf / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_provenance"] = {
        "checkpoint_path": row["checkpoint_path"],
        "training_code_revision": row["training_code_revision"],
        "training_asset_revision": row["training_asset_revision"],
        "accepted_training_manifest_sha256": row["accepted_training_manifest_sha256"],
    }
    metadata["renderer_provenance"] = {
        "code_revision": renderer_code_revision,
        "asset_revision": renderer_asset_revision,
        "device": device,
    }
    write_metadata(metadata_path, metadata)
    if not _policy_artifact_is_valid(
        output_root,
        row,
        fixed_reset,
        renderer_code_revision=renderer_code_revision,
        renderer_asset_revision=renderer_asset_revision,
    ):
        raise RuntimeError(f"renderer emitted an invalid policy artifact: {leaf}")
    return False


def render_library(
    fq4_manifest: str | Path,
    fq3_manifest: str | Path,
    checkpoint_roots: Mapping[str, str | Path],
    output_root: str | Path,
    *,
    renderer_asset_revision: str,
    device: str = "cpu",
    expected_manifest_sha256: Mapping[str, str] = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Freeze the inventory, render every unvalidated policy, then fail-close validate."""
    if device != "cpu":
        raise ValueError("the fixed-reset library is restricted to one-env CPU rendering")
    rows = build_inventory(
        fq4_manifest, fq3_manifest, checkpoint_roots,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    validate_inventory(rows)
    output_root = Path(output_root)
    write_inventory(rows, output_root / "checkpoint_inventory.tsv")
    fixed_reset = load_fixed_reset(FIXED_RESET_ENVELOPE)
    renderer_code_revision = _renderer_revision()
    renderer_asset_revision = _require_revision(renderer_asset_revision, field="renderer asset")
    reused = 0
    for row in rows:
        reused += run_single_policy_renderer(
            row,
            fixed_reset,
            output_root,
            renderer_code_revision=renderer_code_revision,
            renderer_asset_revision=renderer_asset_revision,
            device=device,
        )
    validated = validate_policy_artifacts(output_root, rows)
    return {"policies": len(rows), "reused": reused, "validated": validated}


@dataclass(frozen=True)
class Cfg:
    fq4_manifest: str
    fq3_manifest: str
    fq4_checkpoint_root: str
    fq3_checkpoint_root: str
    renderer_asset_revision: str
    output_root: str = str(OUTPUT_ROOT)
    device: str = "cpu"


def main(cfg: Cfg) -> None:
    result = render_library(
        cfg.fq4_manifest,
        cfg.fq3_manifest,
        {"fq4x8": cfg.fq4_checkpoint_root, "fq3x8": cfg.fq3_checkpoint_root},
        cfg.output_root,
        renderer_asset_revision=cfg.renderer_asset_revision,
        device=cfg.device,
    )
    print(f"policies={result['policies']} reused={result['reused']}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))

"""Freeze and render the exact 56-policy fixed-reset qualitative library.

The accepted-training manifests are the sole source of policy membership.  This
driver deliberately does no outcome selection: every registered arm/seed is
rendered once from the shared Stage-0 reset or the run fails closed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
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
from src.assets.robots.unitree_z1.z1_constants import Z1_HAMMER_XML


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Z1_HAMMER_XML.resolve().parents[2]
OUTPUT_ROOT = SOURCE_ROOT / "docs/results/assets/2026-07-29_56_policy_fixed_reset_library"
FIXED_RESET_PATH = SOURCE_ROOT / FIXED_RESET_ENVELOPE
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
    "checkpoint_cache_path",
    "checkpoint_sha256",
    "training_code_revision",
    "training_asset_revision",
    "accepted_training_manifest_sha256",
)
FQ4_MANIFEST_FIELDS = tuple(
    """campaign disposition arm short task training_seed checkpoint_path
    checkpoint_sha256 training_attempt retry_history code_revision asset_revision
    campaign_config_sha256 treatment_config_sha256 reader_sha256 normalizer_sha256
    treatment_reward_sha256 fixed_action_signature_sha256
    fixed_impedance_signature_sha256 cap_signature_sha256 clean_state""".split()
)
FQ3_MANIFEST_FIELDS = tuple(
    """campaign disposition arm short task training_seed run_name checkpoint_path
    checkpoint_sha256 training_code_revision training_asset_revision slurm_array_id
    retry_history""".split()
)
MANIFEST_FIELDS = {"fq4x8": FQ4_MANIFEST_FIELDS, "fq3x8": FQ3_MANIFEST_FIELDS}
SHORT_BY_CAMPAIGN_ARM = {
    ("fq4x8", "F8"): "f8",
    ("fq4x8", "F0"): "f0",
    ("fq4x8", "D0"): "d0",
    ("fq4x8", "FQ-min"): "fq",
    ("fq3x8", "F8"): "f8",
    ("fq3x8", "B8"): "b8",
    ("fq3x8", "FQ"): "fq",
}
SOURCE_SCOPE = ("src", "scripts", "evaluation/analysis")
ASSET_SCOPE = ("hammer_z1_env/assets",)
FINALIZED_FILES = frozenset((*ARTIFACT_FILENAMES, "metadata.json"))
ROLLOUT_CONTROL_STEPS = 80


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


def _read_tsv(path: Path, campaign: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("missing TSV header")
            if tuple(reader.fieldnames) != MANIFEST_FIELDS[campaign]:
                raise ValueError(f"{campaign} accepted manifest header mismatch")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read accepted training manifest: {path}") from error
    if not rows or any(None in row for row in rows):
        raise ValueError(f"malformed accepted training manifest: {path}")
    return rows


def _manifest_training_revisions(campaign: str, raw: Mapping[str, str]) -> tuple[str, str]:
    if campaign == "fq4x8":
        if raw.get("clean_state", "").lower() != "true":
            raise ValueError("manifest clean_state records dirty training provenance")
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


def capture_render_provenance() -> dict[str, Any]:
    """Derive immutable revisions after checking every render-affecting repo path."""
    def inspect(path: str | Path, scope: Sequence[str], label: str) -> str:
        root = Path(path).resolve()

        def git(*arguments: str) -> str:
            try:
                return subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except subprocess.CalledProcessError as error:
                raise ValueError(f"git provenance check failed for {root}") from error

        if root != Path(git("rev-parse", "--show-toplevel")).resolve():
            raise ValueError(f"{label} repository root is not its git top-level")
        status = git(
            "status", "--porcelain=v1", "--untracked-files=all", "--", *scope
        )
        if status:
            raise ValueError(
                f"renderer {label} scope is dirty: {status.splitlines()[0]}"
            )
        return _require_revision(git("rev-parse", "HEAD"), field=f"renderer {label}")

    expected_asset_root = SOURCE_ROOT.parent / "safe_impact_manipulation"
    if ASSET_ROOT != expected_asset_root.resolve():
        raise ValueError("resolved task asset root is not the source checkout sibling")
    return {
        "code_revision": inspect(SOURCE_ROOT, SOURCE_SCOPE, "source"),
        "asset_revision": inspect(ASSET_ROOT, ASSET_SCOPE, "asset"),
        "source_scope": list(SOURCE_SCOPE),
        "asset_scope": list(ASSET_SCOPE),
        "device": "cpu",
    }


def require_unchanged_render_provenance(
    before: Mapping[str, Any],
) -> None:
    """Fail if a render-affecting revision or worktree path changed during the batch."""
    after = capture_render_provenance()
    if dict(before) != after:
        raise ValueError("renderer source or asset revision changed during batch")


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
        actual_manifest_sha256 = _sha256(manifest_path)
        expected_manifest_digest = _require_sha256(
            expected_manifest_sha256.get(campaign, ""), field=f"{campaign} manifest"
        )
        if actual_manifest_sha256 != expected_manifest_digest:
            raise ValueError(f"{campaign} accepted manifest SHA-256 mismatch")
        for raw in _read_tsv(manifest_path, campaign):
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
            if raw["short"] != SHORT_BY_CAMPAIGN_ARM[(campaign, arm)]:
                raise ValueError(f"manifest short mismatch: {campaign}/{arm}")
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
            checkpoint_file = _checkpoint_file(
                Path(checkpoint_roots[campaign]).resolve(), raw["checkpoint_path"]
            )
            if not checkpoint_file.is_file():
                raise ValueError(f"accepted checkpoint is missing: {checkpoint_file}")
            if _sha256(checkpoint_file) != checkpoint_sha256:
                raise ValueError(f"accepted checkpoint SHA-256 mismatch: {checkpoint_file}")
            rows.append(
                {
                    "campaign": campaign,
                    "arm": arm,
                    "short": raw["short"],
                    "task": task,
                    "training_seed": training_seed,
                    "checkpoint_path": raw["checkpoint_path"],
                    "checkpoint_cache_path": str(
                        Path(Path(raw["checkpoint_path"]).parent.name)
                        / Path(raw["checkpoint_path"]).name
                    ),
                    "_checkpoint_file": str(checkpoint_file),
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


def _inventory_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=INVENTORY_FIELDS, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in sorted(rows, key=_inventory_sort_key):
        writer.writerow({field: row[field] for field in INVENTORY_FIELDS})
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_inventory(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Atomically create the frozen TSV, refusing any non-identical existing pair."""
    validate_inventory(rows)
    path = Path(path)
    payload = _inventory_bytes(rows)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("utf-8")
    if path.exists() and (not path.is_file() or path.read_bytes() != payload):
        raise ValueError("existing inventory differs from frozen bytes")
    if sidecar.exists() and (
        not sidecar.is_file() or sidecar.read_bytes() != sidecar_payload
    ):
        raise ValueError("existing inventory sidecar differs from frozen bytes")
    if not path.exists():
        _atomic_write(path, payload)
    if not sidecar.exists():
        _atomic_write(sidecar, sidecar_payload)
    return sidecar


def run_renderer_process(arguments: Sequence[str]) -> None:
    """Execute the renderer from the checkout whose source was provenance-checked."""
    subprocess.run(
        [sys.executable, str(SOURCE_ROOT / "scripts/render_policy.py"), *arguments],
        cwd=SOURCE_ROOT,
        check=True,
    )


def resume_leaf_shape_is_valid(leaf: str | Path, metadata: Mapping[str, Any]) -> bool:
    """Require exactly five final files and the frozen 80-control-step horizon."""
    leaf = Path(leaf)
    try:
        files = {child.name for child in leaf.iterdir() if child.is_file()}
        no_directories = all(child.is_file() for child in leaf.iterdir())
        requested = int(metadata["rollout"]["requested_control_steps"])
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return no_directories and files == FINALIZED_FILES and requested == ROLLOUT_CONTROL_STEPS


def remove_renderer_scratch(leaf: str | Path) -> None:
    """Remove only the Task 1 renderer's documented non-final inspection images."""
    leaf = Path(leaf)
    for path in (leaf / "step_000_reset.png", *leaf.glob("frame_*.png")):
        if path.is_file():
            path.unlink()


def replace_policy_leaf(staged_leaf: str | Path, final_leaf: str | Path) -> None:
    """Publish a staged leaf on the same filesystem, restoring the old leaf on failure."""
    staged_leaf, final_leaf = Path(staged_leaf), Path(final_leaf)
    final_leaf.parent.mkdir(parents=True, exist_ok=True)
    backup = final_leaf.with_name(f".{final_leaf.name}.backup-{uuid.uuid4().hex}")
    had_final = final_leaf.exists()
    if had_final:
        os.replace(final_leaf, backup)
    try:
        os.replace(staged_leaf, final_leaf)
    except BaseException:
        if had_final and backup.exists():
            os.replace(backup, final_leaf)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def make_policy_staging_root(output_root: str | Path) -> Path:
    """Create same-filesystem scratch beside, never inside, the library root."""
    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.policy-staging-", dir=output_root.parent
        )
    )


def format_validation_counters(
    rows: Sequence[Mapping[str, Any]], validation: Mapping[str, Any]
) -> str:
    """Format the six required counters after fail-closed validation succeeds."""
    artifacts = validation["artifacts"]
    counters = (
        ("policies", len(rows)),
        ("unique_checkpoint_sha256", len({row["checkpoint_sha256"] for row in rows})),
        (
            "unique_reset_state_digest",
            len({artifact["reset_state_digest"] for artifact in artifacts}),
        ),
        ("missing_artifacts", 0),
        ("invalid_video", 0),
        ("second_episode_contamination", 0),
    )
    return "\n".join(f"{name}={value}" for name, value in counters)


def _policy_artifact_is_valid(
    output_root: Path,
    row: Mapping[str, Any],
    fixed_reset: Mapping[str, Any],
    *,
    render_provenance: Mapping[str, Any],
) -> bool:
    """Check one complete policy leaf before allowing a resume skip."""
    leaf = policy_artifact_dir(output_root, row)
    try:
        metadata = json.loads((leaf / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            return False
        if not resume_leaf_shape_is_valid(leaf, metadata):
            return False
        for field in ("campaign", "arm", "training_seed", "checkpoint_sha256"):
            if str(metadata.get(field)) != str(row[field]):
                return False
        if metadata.get("reset_state_digest") != fixed_reset["reset_state_digest"]:
            return False
        if metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
            return False
        _validate_metadata_contract(metadata, row, leaf)
        if metadata.get("code_revision") != render_provenance["code_revision"]:
            return False
        if metadata.get("asset_revision") != render_provenance["asset_revision"]:
            return False
        if metadata.get("renderer_provenance") != dict(render_provenance):
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
            if metadata.get("artifacts", {}).get(filename) != _sha256(artifact):
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
    render_provenance: Mapping[str, Any],
    device: str = "cpu",
) -> bool:
    """Render one policy, returning True only when a fully validated leaf was reused."""
    output_root = Path(output_root).resolve()
    if _policy_artifact_is_valid(
        output_root,
        row,
        fixed_reset,
        render_provenance=render_provenance,
    ):
        return True
    if device != "cpu" or render_provenance.get("device") != "cpu":
        raise ValueError("the fixed-reset library is restricted to CPU rendering")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = make_policy_staging_root(output_root)
    staged_leaf = policy_artifact_dir(staging_root, row)
    staged_leaf.mkdir(parents=True)
    final_leaf = policy_artifact_dir(output_root, row)
    try:
        arguments = [
            "--checkpoint-file", row["_checkpoint_file"],
            "--campaign", row["campaign"],
            "--arm", row["arm"],
            "--training-seed", str(row["training_seed"]),
            "--checkpoint-sha256", row["checkpoint_sha256"],
            "--code-revision", render_provenance["code_revision"],
            "--asset-revision", render_provenance["asset_revision"],
            "--task", row["task"],
            "--out-dir", str(staged_leaf),
            "--steps", str(ROLLOUT_CONTROL_STEPS),
            "--fixed-reset-envelope", str(FIXED_RESET_PATH),
            "--metadata-provenance", "fixed-reset 56-policy library; CPU single-env reinference",
            "--device", device,
        ]
        run_renderer_process(arguments)
        metadata_path = staged_leaf / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_provenance"] = {
            "checkpoint_path": row["checkpoint_path"],
            "training_code_revision": row["training_code_revision"],
            "training_asset_revision": row["training_asset_revision"],
            "accepted_training_manifest_sha256": row[
                "accepted_training_manifest_sha256"
            ],
        }
        metadata["renderer_provenance"] = dict(render_provenance)
        write_metadata(metadata_path, metadata)
        remove_renderer_scratch(staged_leaf)
        if not _policy_artifact_is_valid(
            staging_root,
            row,
            fixed_reset,
            render_provenance=render_provenance,
        ):
            raise RuntimeError(f"renderer emitted an invalid policy artifact: {staged_leaf}")
        replace_policy_leaf(staged_leaf, final_leaf)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return False


def render_library(
    fq4_manifest: str | Path,
    fq3_manifest: str | Path,
    checkpoint_roots: Mapping[str, str | Path],
    output_root: str | Path,
    *,
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
    fixed_reset = load_fixed_reset(FIXED_RESET_PATH)
    render_provenance = capture_render_provenance()
    reused = 0
    try:
        for row in rows:
            reused += run_single_policy_renderer(
                row,
                fixed_reset,
                output_root,
                render_provenance=render_provenance,
                device=device,
            )
    finally:
        require_unchanged_render_provenance(render_provenance)
    validated = validate_policy_artifacts(output_root, rows)
    return {
        "policies": len(rows),
        "reused": reused,
        "validated": validated,
        "counters": format_validation_counters(rows, validated),
    }


@dataclass(frozen=True)
class Cfg:
    fq4_manifest: str
    fq3_manifest: str
    fq4_checkpoint_root: str
    fq3_checkpoint_root: str
    output_root: str = str(OUTPUT_ROOT)
    device: str = "cpu"


def main(cfg: Cfg) -> None:
    result = render_library(
        cfg.fq4_manifest,
        cfg.fq3_manifest,
        {"fq4x8": cfg.fq4_checkpoint_root, "fq3x8": cfg.fq3_checkpoint_root},
        cfg.output_root,
        device=cfg.device,
    )
    print(result["counters"])


if __name__ == "__main__":
    main(tyro.cli(Cfg))

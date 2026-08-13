"""Atomic, hash-pinned entry point for the frozen ``fq4x8`` Task-9 analysis."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np

from evaluation.analysis import first_strike_quality_campaign as quality_analysis
from evaluation.analysis import fq4x8_manifests as manifests
from evaluation.analysis import plot_first_strike_quality_campaign as quality_plot


render_quality_figures = quality_plot.render_quality_figures


EXPECTED_TRAINING_MANIFEST_SHA256 = (
    "fb55f214d6e0cb2da308e6580ef535d4823038bc8ab842a05ca4085ab346ec14"
)
EXPECTED_EVALUATION_MANIFEST_SHA256 = (
    "8679604440712d276996b8768842fa2358b8119c798ab94208c7a504a4f34136"
)
EXPECTED_SUMMARY_SHA256 = (
    "3e2469627ed0daa94c98a94eced580ca87d581d2a721bca97893d9ea03eeaa3a"
)
_PNG_BASENAMES = frozenset(
    {"paired_seed_effects.png", "aggregate_nail_plane_contact_map.png"}
)
_SUMMARY_INTEGER_FIELDS = (
    "training_seed",
    "num_envs",
    "episodes_per_env_sampled",
    "n_episodes_sampled",
    "reset_rng_seed",
    "observation_rng_seed",
    "action_rng_seed",
)


def _read_pinned(path: str | Path, *, expected_sha256: str) -> tuple[Path, bytes]:
    source = Path(path).resolve()
    payload = source.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen input SHA-256 mismatch for {source}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return source, payload


def _decode_utf8(payload: bytes, *, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error


def load_frozen_inputs(
    accepted_training_manifest: str | Path,
    accepted_evaluation_manifest: str | Path,
    summary: str | Path,
) -> dict:
    """Read once, hash-pin, decode, and cross-validate all Task-9 inputs."""

    specifications = (
        (
            "accepted_training_manifest",
            accepted_training_manifest,
            EXPECTED_TRAINING_MANIFEST_SHA256,
        ),
        (
            "accepted_evaluation_manifest",
            accepted_evaluation_manifest,
            EXPECTED_EVALUATION_MANIFEST_SHA256,
        ),
        ("summary", summary, EXPECTED_SUMMARY_SHA256),
    )
    sources: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    for label, path, expected_sha256 in specifications:
        source, payload = _read_pinned(path, expected_sha256=expected_sha256)
        sources[label] = source
        payloads[label] = payload

    training_rows = manifests.parse_training_manifest(
        _decode_utf8(
            payloads["accepted_training_manifest"],
            label="accepted training manifest",
        )
    )
    manifests.validate_training_manifest(training_rows)
    evaluation_rows = manifests.parse_evaluation_manifest(
        _decode_utf8(
            payloads["accepted_evaluation_manifest"],
            label="accepted evaluation manifest",
        )
    )
    manifests.validate_evaluation_manifest(
        evaluation_rows,
        training_rows,
        hashlib.sha256(payloads["accepted_training_manifest"]).hexdigest(),
    )
    summary_rows = list(
        csv.DictReader(
            io.StringIO(
                _decode_utf8(payloads["summary"], label="attempt-2 summary")
            )
        )
    )
    for index, row in enumerate(summary_rows):
        for field in _SUMMARY_INTEGER_FIELDS:
            try:
                value = row[field]
            except KeyError as error:
                raise ValueError(
                    f"attempt-2 summary row {index} is missing {field}"
                ) from error
            if not value.isascii() or not value.isdecimal():
                raise ValueError(
                    f"attempt-2 summary row {index}: {field} is not an exact integer"
                )
            row[field] = int(value)

    return {
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "summary_rows": summary_rows,
        "input_provenance": {
            label: {
                "path": str(sources[label]),
                "sha256": hashlib.sha256(payloads[label]).hexdigest(),
            }
            for label in sources
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing every destination inode.

    Linux uses ``renameat2(RENAME_NOREPLACE)`` and Darwin uses
    ``renamex_np(RENAME_EXCL)``.  Unsupported platforms and missing libc
    primitives fail closed; this function never falls back to ``os.rename``.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Linux":
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace publication requires libc renameat2"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            -100,  # AT_FDCWD on Linux
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif system == "Darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace publication requires libc renamex_np"
            ) from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(
            source_bytes,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise RuntimeError(
            f"atomic no-replace publication is unsupported on {system!r}"
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(destination),
    )


def run_task9(
    accepted_training_manifest: str | Path,
    accepted_evaluation_manifest: str | Path,
    summary: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run the analyzer once and atomically publish its canonical bundle."""

    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(
            f"output parent must be an existing directory: {destination.parent}"
        )

    loaded = load_frozen_inputs(
        accepted_training_manifest,
        accepted_evaluation_manifest,
        summary,
    )
    stage = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.stage-",
        )
    )
    published = False
    try:
        rendered = render_quality_figures(
            loaded["summary_rows"],
            loaded["evaluation_rows"],
            output_dir=stage,
        )
        if not isinstance(rendered, Mapping):
            raise ValueError("renderer result must be a mapping")
        analysis = rendered.get("analysis")
        if not isinstance(analysis, Mapping) or analysis.get("valid") is not True:
            raise ValueError("Task-9 analysis is invalid")
        artifact_paths = [Path(path) for path in rendered.get("artifacts", ())]
        if (
            len(artifact_paths) != 2
            or {path.name for path in artifact_paths} != _PNG_BASENAMES
            or any(path.parent != stage for path in artifact_paths)
        ):
            raise ValueError("renderer did not register exactly the two Task-9 PNGs")
        if (
            {path.name for path in stage.iterdir()} != _PNG_BASENAMES
            or any(not path.is_file() or path.is_symlink() for path in artifact_paths)
        ):
            raise ValueError("renderer output inventory is not canonical")

        analysis_path = stage / "analysis.json"
        analysis_path.write_text(
            json.dumps(
                analysis,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        core_paths = sorted(
            [analysis_path, *artifact_paths], key=lambda path: path.name
        )
        artifact_sha256 = {path.name: _sha256(path) for path in core_paths}
        manifest_path = stage / "task9_artifacts.sha256"
        manifest_path.write_text(
            "".join(
                f"{artifact_sha256[basename]}  {basename}\n"
                for basename in sorted(artifact_sha256)
            ),
            encoding="utf-8",
        )
        expected_inventory = set(artifact_sha256) | {manifest_path.name}
        if {path.name for path in stage.iterdir()} != expected_inventory:
            raise ValueError("canonical Task-9 bundle inventory drifted")
        if os.path.lexists(destination):
            raise FileExistsError(
                f"refusing to overwrite existing output: {destination}"
            )
        _publish_directory_noreplace(stage, destination)
        published = True
    finally:
        if not published and os.path.lexists(stage):
            shutil.rmtree(stage)

    return {
        "inputs": loaded["input_provenance"],
        "output": {
            "directory": str(destination.resolve()),
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": _sha256(
                destination / "task9_artifacts.sha256"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "executable": sys.executable,
            "analysis_module": str(Path(quality_analysis.__file__).resolve()),
            "renderer_module": str(Path(quality_plot.__file__).resolve()),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="run the immutable fq4x8 Task-9 analysis once",
        allow_abbrev=False,
    )
    parser.add_argument("--accepted-training-manifest", required=True)
    parser.add_argument("--accepted-evaluation-manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    provenance = run_task9(
        args.accepted_training_manifest,
        args.accepted_evaluation_manifest,
        args.summary,
        args.output_dir,
    )
    print(
        json.dumps(
            provenance,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

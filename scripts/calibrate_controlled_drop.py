"""Run and publish the fixed five-release controlled-drop calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _REPO_ROOT_STR]
sys.path.insert(0, _REPO_ROOT_STR)

from src.tasks.hammer.calibration.controlled_drop import run_primary_calibration
from src.tasks.hammer.calibration.controlled_drop_contract import ControlledDropResult


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parse the fixed primary calibration invocation."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--code-revision", required=True)
  parser.add_argument("--asset-revision", required=True)
  return parser.parse_args(argv)


def write_primary_result(path: Path, result: ControlledDropResult) -> None:
  """Publish one complete JSON result without replacing any existing path."""
  destination = Path(path)
  if os.path.lexists(destination):
    raise FileExistsError(destination)

  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=destination.parent,
      prefix=f".{destination.name}.",
      suffix=".tmp",
      delete=False,
    ) as temporary:
      temporary_path = Path(temporary.name)
      json.dump(
        result.to_json_dict(),
        temporary,
        sort_keys=True,
        indent=2,
        allow_nan=False,
      )
      temporary.write("\n")
      temporary.flush()
      os.fsync(temporary.fileno())
    os.link(temporary_path, destination)
    temporary_path.unlink()
    temporary_path = None
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
  """Run the fixed primary calibration once and publish only its result."""
  args = parse_args(argv)
  result = run_primary_calibration(
    device=args.device,
    expected_code_revision=args.code_revision,
    expected_asset_revision=args.asset_revision,
  )
  write_primary_result(args.output, result)
  impulses = [row.impulse_n_s for row in result.summary.trials]
  print(
    "controlled-drop calibration complete\n"
    f"output={args.output}\n"
    f"impulses_n_s={impulses!r}\n"
    f"i_ref_mean_n_s={result.summary.i_ref_mean_n_s!r}\n"
    f"device={result.execution.device}\n"
    f"code_revision={result.execution.code_revision}\n"
    f"asset_revision={result.execution.asset_revision}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

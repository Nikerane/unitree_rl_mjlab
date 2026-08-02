"""Fail-closed marker for the deferred C2 impulse-enforcement experiment.

The fixed-reset direct reference has not been qualified as a source of
non-identical over-limit events.  Repeating it cannot calibrate soft-CaT
enforcement, so this phase must stop before constructing an environment.
"""

from __future__ import annotations

import sys


DEFERRED_REASON = (
  "DEFERRED: direct reference not enforcement-qualified; "
  "imp_max_p must remain 0.0"
)


def main() -> int:
  print(DEFERRED_REASON, file=sys.stderr)
  return 2


if __name__ == "__main__":
  raise SystemExit(main())

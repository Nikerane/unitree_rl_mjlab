#!/usr/bin/env bash
# Open the Z1 hammer reward datasheet in your default browser (macOS / Linux).
# Usage:  ./viz/reward_explorer/open_sheet.sh   (or: bash viz/reward_explorer/open_sheet.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHEET="$HERE/reward_sheet.html"

if [ ! -f "$SHEET" ]; then
  echo "datasheet missing — regenerate it with:  python \"$HERE/build_sheet.py\"" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) open "$SHEET" ;;
  Linux)  xdg-open "$SHEET" >/dev/null 2>&1 || { echo "open manually: $SHEET"; exit 0; } ;;
  *)      echo "open manually: $SHEET"; exit 0 ;;
esac
echo "opened $SHEET"

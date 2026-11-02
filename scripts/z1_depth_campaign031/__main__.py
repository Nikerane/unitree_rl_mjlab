"""Retirement notice only. No training, evaluation or submission code."""
import sys

MESSAGE = (
    "Retired: the custom depth031 campaign launcher cannot launch work.\n"
    "Use the existing runner with the retained reward configuration.\n"
    "See docs/experiments/z1-depth-rewards.md.\n"
    "Original tracked source: Git 10c5bfe79c59ecfda16dfffae420d5e278d00977.\n"
    "Independent archive: /Volumes/NikeraneSSD/Z1-campaign-backups/2026-11-01-depth-reward031/retired-campaign031-cleanup.tar\n"
    "Historical source and tests: .superpowers/sdd/2026-11-01-z1-simplification-review/retired-campaign031.tar\n"
)


def main():
    help_only = sys.argv[1:] in (["--help"], ["-h"])
    print(MESSAGE, file=sys.stdout if help_only else sys.stderr)
    return 0 if help_only else 2


if __name__ == "__main__":
    raise SystemExit(main())

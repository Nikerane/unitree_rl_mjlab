#!/bin/bash
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002/admission-staged-integration002/autonomous005/controller-%j.log
set -euo pipefail
umask 077
# Root supplies the three approved hashes at its sole, later submission.
test "$#" -eq 3
for z1_hash in "$@"; do
  test "${#z1_hash}" -eq 64
  case "$z1_hash" in *[!0-9a-f]*) exit 64 ;; esac
done
z1_base=/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002/admission-staged-integration002/autonomous005
z1_entry="$z1_base/controller-source005/entry.py"
printf '%s  %s\n' "$1" "$z1_entry" | /usr/bin/sha256sum --check --status
# This request stays inactive until Task3's reviewed literal adapter is installed.
exec /usr/bin/python3 -B -I -S "$z1_entry" --activate \
  --manifest "$z1_base/controller-source005/source-manifest.json" --manifest-sha256 "$2" \
  --config "$z1_base/controller-config.json" \
  --delegation "$z1_base/controller-delegation.json" --delegation-sha256 "$3"

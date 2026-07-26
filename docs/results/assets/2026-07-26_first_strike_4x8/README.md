# First-strike 4×8 evidence assets

- `summary.csv` and `first_strike_campaign_analysis.json` are the frozen
  campaign statistics.
- The 32 raw sampled NPZ archives are retained on Vega at each
  `summary.csv::sampled_trace_path`, bound by that row's
  `sampled_trace_digest` and `sampled_trace_artifact_sha256`. They are not
  duplicated in Git because of their size; exact episode-level reconstruction
  depends on those bound archives.
- The report files at this directory's root are the retained
  `analysis_attempt2_7205237` outputs. They are audit history only: visual
  inspection found an x-y scale-anchor bug and misleading no-contact
  alignment in the trajectory HTML.
- `attempt3_33c6606/` is the corrected report generated from clean revision
  `33c6606351d58eb86d033fd4baa98161dbbaf3cc`. Use its HTML reports and
  `first_strike_fixed_reset_xz_grid.png` for interpretation. Its
  `SHA256SUMS` binds the exact files.

No tracked source file was copied to Vega. The reporting revision was deployed
by commit, push, and a fresh clean checkout; the generated report artifacts
were copied back for banking.

# Failed dose-matching bank (v2)

These files preserve the 2026-07-25 dose-matching qualification attempt as
immutable historical negative evidence. They are not inputs to the corrected
C / D-prime / F / E qualification.

Original SHA-256 values, verified unchanged after relocation:

| File | SHA-256 |
|---|---|
| `bank_manifest.json` | `5796559eb9679361206a7b485d703eca2e3be516226113d1004af0e899bb62fd` |
| `probe_raw.npz` | `e910c7e8136626e23b68f0c55bff53857c217ff55fc077723c19fdf75f00e1cb` |
| `probe_summary.json` | `7147bb2564d40641a7c609c5fedfbe9b9056ef3c8cb70df26d3af8c612852e91` |

The run's legacy-normalizer gate was defective: it removed the
`nail_driven` success termination and continued beyond the terminal reference
snapshot, observing `0.742736 N·s` instead of reproducing the configured
`0.6094 N·s` terminal value.

That horizon defect does not erase the separate negative result. The
preregistered dose transport genuinely failed for D: validation relative error
was `9.17%` overall and exceeded `10%` in all four strata (worst:
`43.67%`, `mx_maxoff`). This invalidated dose matching as a comparator design
and motivated the equal-weight, one-shot D-prime arm.

The large JSON/NPZ files remain local and intentionally untracked.

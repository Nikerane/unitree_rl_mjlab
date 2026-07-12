"""Impedance sweep driver: run impedance_one.py once per kp_scale, collect JSON."""
from __future__ import annotations
import subprocess, json, sys, os
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "impedance_one.py")
REPO = "/Users/nikerane/repos/unitree_rl_mjlab"
OUT = os.path.join(HERE, "impedance_results.json")
KPS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

def run(kp):
    cmd = [sys.executable, PROBE, "--kp_scale", str(kp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"ERROR": (r.stderr or "")[-400:], "kp_scale": kp}

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run, KPS))
    json.dump(results, open(OUT, "w"), indent=1)
    for r in sorted(results, key=lambda r: r.get("kp_scale", 0)):
        if "ERROR" in r:
            print("ERR kp", r["kp_scale"], r["ERROR"][:140]); continue
        print(f"kp={r['kp_scale']:<5} lock_ok={int(r['lock_ok'])} nail={r['nail_disp_mm']}mm "
              f"vach={r['achieved_impact_v']} imp_win={r['impact_win_substeps']} full_win={r['full_win_substeps']} "
              f"Λ/cap impact={r['worst_over_cap_IMPACT']}(j{r['worst_joint']}) full={r['worst_over_cap_FULL']} "
              f"accum={r['worst_over_cap_ACCUM']} peakF={r['peak_F']} bind={int(r['binds_impact'])}")

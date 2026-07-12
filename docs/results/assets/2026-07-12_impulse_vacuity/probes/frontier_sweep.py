"""Driver for the binding-frontier sweep: runs frontier_one.py once per (target, kp_scale, v)
point in parallel processes (one mjlab env per process; warp needs that), collects the JSON.

Curves:
  * reflecting x kp_scale=0 (ballistic/compliant lower-bound frontier) x velocity sweep
  * yielding   x kp_scale=0 (real nail -> never binds, it yields)      x velocity sweep
  * reflecting x kp_scale=1 (FIXED impedance)  x velocity sweep  (higher m_eff)
  * reflecting x kp_scale=5 (VIC-like stiffer) x velocity sweep  (higher still)
The x-axis in the figure is the MEASURED achieved impact velocity, not the command.
"""
from __future__ import annotations
import subprocess, json, sys, os
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "frontier_one.py")
REPO = "/Users/nikerane/repos/unitree_rl_mjlab"
OUT = os.path.join(HERE, "frontier_results.json")

VELS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
points = []
for tgt in ("reflecting", "yielding"):
    for v in VELS:
        points.append({"target": tgt, "kp_scale": 0.0, "v": v})
for kp in (1.0, 5.0):
    for v in VELS:
        points.append({"target": "reflecting", "kp_scale": kp, "v": v})


def run(p):
    cmd = [sys.executable, PROBE, "--v", str(p["v"]),
           "--target", p["target"], "--kp_scale", str(p["kp_scale"])]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    # JSON is the probe's single stdout print; find the last line that parses.
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"ERROR": (r.stderr or "")[-400:], **p}


if __name__ == "__main__":
    workers = int(os.environ.get("SWEEP_WORKERS", "4"))
    print(f"running {len(points)} points, {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, points))
    json.dump(results, open(OUT, "w"), indent=1)
    ok = [r for r in results if "ERROR" not in r]
    bad = [r for r in results if "ERROR" in r]
    print(f"done: {len(ok)} ok, {len(bad)} errored -> {OUT}")
    for r in bad:
        print("  ERR", r.get("target"), r.get("kp_scale"), r.get("v"), r["ERROR"][:120])

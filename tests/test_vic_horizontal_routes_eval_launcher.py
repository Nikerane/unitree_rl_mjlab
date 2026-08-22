"""Shell contract for the six-cell horizontal-route frozen evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import evaluate_vic_horizontal_routes as evaluator


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/vega_vic_horizontal_routes_p02_eval.sbatch"
ASSET_SHA = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
CELLS = (
  (0, "horizontal_routes_annealed", -1, "rminus", "ANNEALED_CHECKPOINT"),
  (1, "horizontal_routes_annealed", 0, "r0", "ANNEALED_CHECKPOINT"),
  (2, "horizontal_routes_annealed", 1, "rplus", "ANNEALED_CHECKPOINT"),
  (3, "horizontal_routes_persistent", -1, "rminus", "PERSISTENT_CHECKPOINT"),
  (4, "horizontal_routes_persistent", 0, "r0", "PERSISTENT_CHECKPOINT"),
  (5, "horizontal_routes_persistent", 1, "rplus", "PERSISTENT_CHECKPOINT"),
)
OUTPUTS = (
  "fixed_trace.npz",
  "training_like_seed_2_trace.npz",
  "training_like_seed_2026081701_trace.npz",
  "training_like_seed_2026081702_trace.npz",
  "summary.json",
)


def _git(path: Path, *args: str) -> str:
  result = subprocess.run(
    ["git", "-c", "user.email=routes@test", "-c", "user.name=Routes", *args],
    cwd=path,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _repo(path: Path) -> str:
  path.mkdir(parents=True)
  (path / "marker").write_text("fixture\n")
  _git(path, "init", "-q")
  _git(path, "add", "marker")
  _git(path, "commit", "-qm", "fixture")
  return _git(path, "rev-parse", "HEAD")


def _env(tmp_path: Path, cell: int) -> dict[str, str]:
  home = tmp_path / "home"
  code = tmp_path / "repos/code"
  assets = tmp_path / "repos/safe_impact_manipulation"
  fake_bin = home / "bin"
  fake_bin.mkdir(parents=True)
  real_git = shutil.which("git")
  real_sha = shutil.which("sha256sum")
  assert real_git and real_sha
  (fake_bin / "git").write_text(
    "#!/bin/sh\n"
    "case \" $* \" in\n"
    "  *' merge-base '*) exit 0 ;;\n"
    f"  *' cat-file -e {evaluator.TRAINING_CODE_REVISION}^{{commit}}'*) exit 0 ;;\n"
    f"  *'safe_impact_manipulation rev-parse HEAD'*) printf '%s\\n' {ASSET_SHA}; exit 0 ;;\n"
    "  *) exec \"$REAL_GIT\" \"$@\" ;;\n"
    "esac\n"
  )
  (fake_bin / "git").chmod(0o755)

  checkpoints = {}
  for role, variable in (
    ("horizontal_routes_annealed", "ANNEALED_CHECKPOINT"),
    ("horizontal_routes_persistent", "PERSISTENT_CHECKPOINT"),
  ):
    path = tmp_path / "checkpoints" / role / "model_499.pt"
    path.parent.mkdir(parents=True)
    path.write_text(role)
    checkpoints[variable] = path

  (fake_bin / "sha256sum").write_text(
    "#!/bin/sh\n"
    "first=${1:-}\n"
    "for variable in ANNEALED_CHECKPOINT PERSISTENT_CHECKPOINT; do\n"
    "  eval path=\\\"\\${$variable}\\\"\n"
    "  if [ \"$first\" = \"$path\" ]; then\n"
    "    [ \"${BAD_CHECKPOINT_HASH:-}\" = 1 ] && printf '%064d  %s\\n' 0 \"$first\" && exit 0\n"
    "    eval digest=\\\"\\${${variable}_FROZEN_SHA}\\\"\n"
    "    printf '%s  %s\\n' \"$digest\" \"$first\"; exit 0\n"
    "  fi\n"
    "done\n"
    "exec \"$REAL_SHA\" \"$@\"\n"
  )
  (fake_bin / "sha256sum").chmod(0o755)

  stubs = home / "runtime-stubs"
  stubs.mkdir()
  (stubs / "sitecustomize.py").write_text(
    "import os, sys, types\n"
    "from importlib import metadata\n"
    "metadata.version = {'mjlab':'1.4.0','mujoco':'3.8.1','mujoco-warp':'3.8.1'}.__getitem__\n"
    "torch=types.ModuleType('torch'); torch.cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_name=lambda i:os.environ.get('RUNTIME_GPU','NVIDIA A100-SXM4-40GB')); sys.modules['torch']=torch\n"
  )
  generator = home / "generate.py"
  generator.write_text(
    "import json, os, sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "out,role,sign,checkpoint,digest,code,asset=sys.argv[1:]; sign=int(sign); root=Path(out); root.mkdir()\n"
    "bad=os.environ.get('BAD_OUTPUT',''); scalar=(2,2); route=np.full(scalar,sign,dtype=np.int8)\n"
    "if bad=='wrong_route': route.fill(-sign if sign else 1)\n"
    "fields=dict(episode_id=np.zeros(scalar,dtype=np.int64),route_sign=route,strike_phase=np.full(scalar,.25),hammer_head_pos_w=np.zeros((*scalar,3)),assigned_reference_waypoint_w=np.zeros((*scalar,3)),straight_reference_waypoint_w=np.zeros((*scalar,3)),imitation_eligible=np.ones(scalar,dtype=bool),delta_impulse=np.zeros(scalar),delta_velocity=np.zeros(scalar),delta=np.zeros(scalar),lambda_per_joint=np.zeros((*scalar,6)),substep_rolling_per_joint=np.zeros((20,2,6)))\n"
    "if bad=='nonfinite': fields['strike_phase'][0,0]=np.nan\n"
    "for name in ('fixed_trace.npz','training_like_seed_2_trace.npz','training_like_seed_2026081701_trace.npz','training_like_seed_2026081702_trace.npz'): np.savez(root/name,**fields)\n"
    "task='Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-'+('Annealed' if role.endswith('annealed') else 'Persistent')\n"
    "def proto(seed,n,steps,mode): return {'seed':seed,'num_envs':n,'control_steps':steps,'policy_mode':mode,'auto_reset':mode=='sampled','initial_population_sha256':'a'*64,'rng_streams':{'reset':seed+10000019,'observation':seed+20000033,'action':seed+30000041},'cat_replay':{'imp_max_p_live':0.0},'training_config':{'imp_max_p':0.2,'imp_limit_n_m_s':[.369,.246,.738,.369,.246,.0164]},'forced_route_sign':sign,'actor_only_checkpoint_load':True}\n"
    "summary={'schema_version':1,'task':task,'checkpoint':{'role':role,'path':str(Path(checkpoint).resolve()),'sha256':digest},'code_revision':code,'asset_revision':asset,'route':{'label':{-1:'rminus',0:'r0',1:'rplus'}[sign],'sign':sign},'training_cap_identity_n_m_s':[.369,.246,.738,.369,.246,.0164],'protocol':{'actor_only_checkpoint_load':True,'live_imp_max_p':0.0,'fixed_seed':2026081202,'stochastic_seeds':[2,2026081701,2026081702]},'populations':{'fixed_mean':{'protocol':proto(2026081202,64,8,'mean')},'training_like_sampled':{str(s):{'protocol':proto(s,4096,24,'sampled')} for s in (2,2026081701,2026081702)}}}\n"
    "(root/'summary.json').write_text(json.dumps(summary,allow_nan=False))\n"
  )
  python = home / "repos/unitree_rl_mjlab/.venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text(
    "#!/bin/sh\n"
    "{ printf 'CALL'; for arg in \"$@\"; do printf '\\t%s' \"$arg\"; done; printf '\\n'; } >> \"$HOME/calls\"\n"
    "case \"${1:-}\" in\n"
    "  -) PYTHONPATH=\"$HOME/runtime-stubs:${PYTHONPATH:-}\" exec \"$REAL_PYTHON\" \"$@\" ;;\n"
    "  *scripts/evaluate_vic_horizontal_routes.py) role=''; sign=''; checkpoint=''; output=''; prev=''; for arg in \"$@\"; do [ \"$prev\" = --checkpoint-role ] && role=\"$arg\"; [ \"$prev\" = --route-sign ] && sign=\"$arg\"; [ \"$prev\" = --checkpoint ] && checkpoint=\"$arg\"; [ \"$prev\" = --output-dir ] && output=\"$arg\"; prev=\"$arg\"; done; if [ \"$role\" = horizontal_routes_annealed ]; then digest=$ANNEALED_CHECKPOINT_FROZEN_SHA; else digest=$PERSISTENT_CHECKPOINT_FROZEN_SHA; fi; exec \"$REAL_PYTHON\" \"$HOME/generate.py\" \"$output\" \"$role\" \"$sign\" \"$checkpoint\" \"$digest\" \"$EXPECTED_CODE_REVISION\" \"$EXPECTED_ASSET_REVISION\" ;;\n"
    "  *) exit 74 ;;\n"
    "esac\n"
  )
  python.chmod(0o755)
  code_sha = _repo(code)
  _repo(assets)
  env = {
    "PATH": f"{fake_bin}:{os.environ['PATH']}", "HOME": str(home),
    "RUN_ROOT": str(code), "ASSET_REPO": str(assets),
    "EXPECTED_CODE_REVISION": code_sha, "EXPECTED_ASSET_REVISION": ASSET_SHA,
    "REAL_GIT": real_git, "REAL_SHA": real_sha, "REAL_PYTHON": sys.executable,
    "ANNEALED_CHECKPOINT": str(checkpoints["ANNEALED_CHECKPOINT"]),
    "PERSISTENT_CHECKPOINT": str(checkpoints["PERSISTENT_CHECKPOINT"]),
    "ANNEALED_CHECKPOINT_FROZEN_SHA": evaluator.POLICY_SPECS["horizontal_routes_annealed"]["sha256"],
    "PERSISTENT_CHECKPOINT_FROZEN_SHA": evaluator.POLICY_SPECS["horizontal_routes_persistent"]["sha256"],
    "SLURM_ARRAY_JOB_ID": "810000", "SLURM_JOB_ID": str(1810000 + cell),
    "SLURM_ARRAY_TASK_ID": str(cell), "SLURM_ARRAY_TASK_COUNT": "6",
    "SLURM_ARRAY_TASK_MIN": "0", "SLURM_ARRAY_TASK_MAX": "5", "SLURM_ARRAY_TASK_STEP": "1",
  }
  return env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=30)


def test_launcher_executes_exact_six_policy_route_cells_and_manifests(tmp_path: Path):
  for cell, role, sign, label, variable in CELLS:
    env = _env(tmp_path / label / role, cell)
    result = _run(env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Z1_VIC_HORIZONTAL_ROUTES_P02_EVAL_PASS role={role} route={label}" in result.stdout
    calls = [line.split("\t")[1:] for line in (Path(env["HOME"]) / "calls").read_text().splitlines()]
    call = next(value for value in calls if value[0].endswith("evaluate_vic_horizontal_routes.py"))
    assert call[call.index("--checkpoint-role") + 1] == role
    assert call[call.index("--route-sign") + 1] == str(sign)
    assert call[call.index("--checkpoint") + 1] == env[variable]
    output = Path(call[call.index("--output-dir") + 1])
    assert sorted(path.name for path in output.iterdir()) == sorted(OUTPUTS)
    assert (output.parent / "SHA256SUMS").read_text() == "".join(
      f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  evaluation/{name}\n"
      for name in OUTPUTS
    )


@pytest.mark.parametrize(
  ("variable", "value"),
  (("SLURM_ARRAY_TASK_COUNT", "5"), ("SLURM_ARRAY_TASK_ID", "6"), ("WORLD_SIZE", "1")),
)
def test_launcher_rejects_wrong_array_or_distributed_context(tmp_path, variable, value):
  env = _env(tmp_path / variable, 0)
  env[variable] = value
  assert _run(env).returncode == 2


@pytest.mark.parametrize("bad", ("wrong_route", "nonfinite"))
def test_launcher_rejects_bad_route_telemetry(tmp_path: Path, bad: str):
  env = _env(tmp_path / bad, 2)
  env["BAD_OUTPUT"] = bad
  result = _run(env)
  assert result.returncode == 2
  assert "Z1_VIC_HORIZONTAL_ROUTES_P02_EVAL_PASS" not in result.stdout


def test_launcher_rejects_checkpoint_hash_mismatch(tmp_path: Path):
  env = _env(tmp_path, 0)
  env["BAD_CHECKPOINT_HASH"] = "1"
  result = _run(env)
  assert result.returncode == 2
  assert "role/checkpoint SHA-256 mismatch" in result.stdout

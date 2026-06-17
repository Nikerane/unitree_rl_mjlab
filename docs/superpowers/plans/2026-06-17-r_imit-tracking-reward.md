# r_imit Weak-Annealed Tracking Reward (A-TRACK arm) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Transient artifact:** this is an execution plan; delete after the work lands (the *design* lives in `docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`, the 2026-06-17 changelog).

**Goal:** Add a weak, annealed, ante-impact reference-tracking reward term (`r_imit`) behind an `imitation` flag, so a new **A-TRACK** arm (= A-BASE + `r_imit`) can be trained and compared to A-BASE on Vega.

**Architecture:** `r_imit = exp(−‖p_head − p*(φ)‖²/σ²) · 1[pre-contact]`, where `p*(φ)` is the existing `SingleStrikeReference` waypoint. The term is a stateful `ManagerTermBase` with a per-episode "has contacted" latch. Its weight (w₀=0.1) anneals to 0 by training step 6000 via mjlab's native `reward_curriculum` (which mutates the live `term_cfg.weight`, so `validate_rewards.py`'s dynamic read stays truthful). The term is added only when `imitation=True`; `imitation=False` is byte-identical to today's A-BASE. A second gym task `Unitree-Z1-Hammer-Track` registers the imitation-enabled cfg.

**Tech Stack:** Python, PyTorch, mjlab 1.4.0 (managers: reward, curriculum), MuJoCo-Warp. Tests: pytest (stub-env unit tests, no MuJoCo) + `validate_rewards.py` (scripted end-to-end).

**Key facts (verified against the code):**
- Reward term pattern: `src/tasks/hammer/mdp/rewards.py::ImpactProgressTerm` (ManagerTermBase: `__init__(cfg, env)`, `reset(env_ids)`, `__call__(env, ...) -> (B,)`).
- Reference access: `get_strike_reference(env)` → `ref.update(head_w, nail_top_w, env.episode_length_buf)` → `phi`; `ref.waypoint(phi)` → `p*` shape (B,3). `waypoint(0) == ref._head0` (the anchor = head position at episode_step==0).
- Head site `robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)`; nail-top site `nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)`; contact `sensor.data.found` shape (B,P).
- Per-robot head site is injected in `config/z1/env_cfgs.py` (`HAMMER_HEAD_SITE_NAME`); nail-top site is set in the cfg definition.
- `common_step_counter` increments once per control step → `T_anneal = 250 iters × 24 steps/iter = 6000`; full run = 12000.
- `reward_dict()` in `validate_rewards.py` returns the **weighted** per-term reward (e.g. completion == weight 100 at success).

---

### Task 1: `ImitationPriorTerm` reward term + unit tests

**Files:**
- Create: `tests/test_imitation_reward.py`
- Modify: `src/tasks/hammer/mdp/rewards.py` (add class + a module-top import)

- [ ] **Step 1: Write the failing test**

Create `tests/test_imitation_reward.py`:

```python
"""Unit tests for the weak-annealed tracking prior (ImitationPriorTerm, plan T2).

    r_imit = exp(-||p_head - p*(phi)||^2 / sigma^2) * 1[pre-contact]

Runs against a tiny stub env (no MuJoCo/Warp). The SingleStrikeReference is pure
torch, so get_strike_reference() attaches a real instance to the stub env.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.hammer.mdp.rewards import ImitationPriorTerm

_ROBOT_CFG = SimpleNamespace(name="robot", site_ids=[0])
_NAIL_CFG = SimpleNamespace(name="nail_block", site_ids=[0])
_PARAMS = dict(
    sensor_name="hammer_nail_contact",
    robot_cfg=_ROBOT_CFG,
    nail_cfg=_NAIL_CFG,
    sigma=0.05,
)


class _StubSensor:
    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(found=torch.zeros(num_envs, 1))

    def set_found(self, value: bool) -> None:
        self.data.found.fill_(float(value))


def _make_env(num_envs: int = 1):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    nail = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(num_envs, 1, 3)))
    sensor = _StubSensor(num_envs)
    scene = {"robot": robot, "nail_block": nail, "hammer_nail_contact": sensor}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    return env, robot, nail, sensor


def _set(robot, nail, *, head, nail_top=(0.0, 0.0, 0.10)):
    robot.data.site_pos_w[:, 0, :] = torch.tensor(head)
    nail.data.site_pos_w[:, 0, :] = torch.tensor(nail_top)


def test_at_reference_anchor_is_one():
    """First call after reset: head == anchor == waypoint(0) -> Gaussian=1, pre-contact -> 1."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0          # fresh episode -> reference anchors here
    r = term(env, **_PARAMS)
    assert float(r) == pytest.approx(1.0, abs=1e-4)


def test_far_from_reference_decays_to_zero():
    """Head far from the waypoint on a later step -> Gaussian ~ 0."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    term(env, **_PARAMS)                   # anchor
    _set(robot, nail, head=(1.0, 1.0, 1.0))  # far away
    env.episode_length_buf[:] = 1          # no re-anchor
    sensor.set_found(False)
    r = term(env, **_PARAMS)
    assert float(r) < 0.01


def test_ante_impact_latch_zeroes_from_first_contact():
    """From the first contact onward the term is 0, regardless of distance."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)  # pre-contact
    # Contact begins; even sitting on the reference, the term must be 0.
    env.episode_length_buf[:] = 1
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0
    # Contact released, but the latch persists for the rest of the episode.
    env.episode_length_buf[:] = 2
    sensor.set_found(False)
    assert float(term(env, **_PARAMS)) == 0.0


def test_latch_resets_per_episode():
    """reset() clears the contact latch so a new episode tracks again."""
    env, robot, nail, sensor = _make_env()
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    term(env, **_PARAMS)
    env.episode_length_buf[:] = 1
    sensor.set_found(True)
    assert float(term(env, **_PARAMS)) == 0.0  # latched
    term.reset(None)
    sensor.set_found(False)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0              # re-anchor
    assert float(term(env, **_PARAMS)) == pytest.approx(1.0, abs=1e-4)


def test_returns_per_env_shape():
    env, robot, nail, sensor = _make_env(num_envs=3)
    term = ImitationPriorTerm(cfg=None, env=env)
    _set(robot, nail, head=(0.0, 0.0, 0.20))
    env.episode_length_buf[:] = 0
    r = term(env, **_PARAMS)
    assert tuple(r.shape) == (3,)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_imitation_reward.py -q`
Expected: FAIL — `ImportError: cannot import name 'ImitationPriorTerm'`.

- [ ] **Step 3: Implement the term**

In `src/tasks/hammer/mdp/rewards.py`, add this import after the existing imports (top of file, after line 11 `from mjlab.managers.scene_entity_config import SceneEntityCfg`):

```python
from src.tasks.hammer.mdp.references import get_strike_reference
```

Then append this class at the end of the file (after `ImpactProgressTerm`):

```python
class ImitationPriorTerm(ManagerTermBase):
  """Weak ante-impact tracking prior (plan T2): exp(-||p_head - p*(phi)||^2 / sigma^2) * 1[pre-contact].

  Rewards the hammer head for following the scripted SingleStrikeReference waypoint
  p*(phi), but ONLY before the first hammer->nail contact of the episode (ante-impact
  latch), so it shapes the approach/wind-up and never the impact. Position-only,
  task-space; no velocity imitation (Biemond/TAC: ill-posed through contact). Intended
  to run at a small weight that ANNEALS to 0 via mjlab's reward_curriculum, so the policy
  stays free to deviate and beat the reference (online RL, not DeepMimic).

  Stateful: a per-env "has contacted this episode" latch, reset per episode.

  KNOWN RISK (watch-item, not yet guarded): the latch bounds accumulation only once
  contact occurs. A policy that hovers near the wind-up apex without contacting keeps
  earning ~weight/step. Mitigated by (a) the anneal to 0 by step 6000, and (b) the +100
  completion bonus that terminates the episode (striking dominates hovering). The
  deviation-norm / press-watchdog training metrics surface it; add a per-episode cap or
  a phi-descent gate only if observed (augment-not-replace).
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._contacted: torch.Tensor = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self._contacted.fill_(False)
    else:
      self._contacted[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    nail_cfg: SceneEntityCfg = _DEFAULT_NAIL_CFG,
    sigma: float = 0.05,
  ) -> torch.Tensor:
    """Returns shape (B,)."""
    robot: Entity = env.scene[robot_cfg.name]
    nail: Entity = env.scene[nail_cfg.name]
    sensor = env.scene[sensor_name]

    head_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    nail_top_w = nail.data.site_pos_w[:, nail_cfg.site_ids].squeeze(1)

    ref = get_strike_reference(env)
    phi = ref.update(head_w, nail_top_w, env.episode_length_buf)
    p_star = ref.waypoint(phi)

    dist_sq = torch.sum((head_w - p_star) ** 2, dim=-1)
    gauss = torch.exp(-dist_sq / sigma**2)

    # Ante-impact latch: zero from the first contact of the episode onward.
    found = (sensor.data.found > 0).any(-1)
    self._contacted = self._contacted | found
    gate = (~self._contacted).to(head_w.dtype)
    return gauss * gate
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_imitation_reward.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tasks/hammer/mdp/rewards.py tests/test_imitation_reward.py
git commit -m "feat(reward): add ImitationPriorTerm (weak ante-impact tracking prior, T2)"
```

---

### Task 2: Wire the `imitation` flag — reward term + anneal curriculum + head-site

**Files:**
- Modify: `src/tasks/hammer/hammer_env_cfg.py` (add `imitation` param; add `r_imit` reward + curriculum)
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (thread the flag; wire head site)
- Modify: `tests/test_configs.py` (add the wiring test)

- [ ] **Step 1: Write the failing test**

In `tests/test_configs.py`, add these imports near the top (with the other imports):

```python
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
```

Add this test class at the end of the file:

```python
class TestImitationFlag:
    def test_base_has_no_r_imit_or_curriculum(self):
        base = z1_hammer_env_cfg(imitation=False)
        assert "r_imit" not in base.rewards
        assert not base.curriculum  # A-BASE unchanged: empty curriculum

    def test_track_adds_only_r_imit(self):
        base = z1_hammer_env_cfg(imitation=False)
        track = z1_hammer_env_cfg(imitation=True)
        # Exactly one new reward term, nothing else changed.
        assert set(track.rewards) - set(base.rewards) == {"r_imit"}
        assert track.rewards["r_imit"].weight == pytest.approx(0.1)

    def test_track_anneal_curriculum_present(self):
        track = z1_hammer_env_cfg(imitation=True)
        assert "r_imit_anneal" in track.curriculum
        stages = track.curriculum["r_imit_anneal"].params["stages"]
        assert stages[0]["weight"] == pytest.approx(0.1)
        assert stages[-1]["weight"] == pytest.approx(0.0)
        assert stages[-1]["step"] == 6000  # 250 iters * 24 steps/iter

    def test_track_head_site_wired(self):
        track = z1_hammer_env_cfg(imitation=True)
        assert track.rewards["r_imit"].params["robot_cfg"].site_names == (HAMMER_HEAD_SITE_NAME,)

    def test_track_play_mode_clears_curriculum(self):
        # Play/validation mode keeps r_imit but drops the anneal (weight fixed at 0.1).
        track_play = z1_hammer_env_cfg(play=True, imitation=True)
        assert "r_imit" in track_play.rewards
        assert not track_play.curriculum
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_configs.py::TestImitationFlag -q`
Expected: FAIL — `z1_hammer_env_cfg() got an unexpected keyword argument 'imitation'`.

- [ ] **Step 3a: Add the flag + term + curriculum to `make_hammer_env_cfg`**

In `src/tasks/hammer/hammer_env_cfg.py`, add these imports (after line 23 `from mjlab.managers.termination_manager import TerminationTermCfg`):

```python
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
```

Change the function signature (line 35) from:

```python
def make_hammer_env_cfg() -> ManagerBasedRlEnvCfg:
```
to:
```python
def make_hammer_env_cfg(imitation: bool = False) -> ManagerBasedRlEnvCfg:
```

Immediately **before** the `return ManagerBasedRlEnvCfg(` line (currently line 238), insert:

```python
  # --- T2 weak-annealed tracking prior (A-TRACK arm) ---
  # imitation=False -> byte-identical A-BASE. imitation=True -> add r_imit + its anneal.
  curriculum: dict = {}
  if imitation:
    rewards["r_imit"] = RewardTermCfg(
      func=hammer_mdp.ImitationPriorTerm,
      weight=0.1,  # w_I0; budget rule: cumulative w0*Sum r_imit < 0.35 * completion
      params={
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": SceneEntityCfg("robot", site_names=()),  # head site, per-robot
        "nail_cfg": SceneEntityCfg("nail_block", site_names=("nail_top",)),
        "sigma": 0.05,
      },
    )
    # Linear-ish decay to 0 by step 6000 (= 250 iters * 24 steps/iter). Shape immaterial
    # (Freitag); reward_curriculum mutates the live term weight so validate_rewards' read
    # stays truthful. common_step_counter increments once per control step.
    curriculum["r_imit_anneal"] = CurriculumTermCfg(
      func=reward_curriculum,
      params={
        "reward_name": "r_imit",
        "stages": [
          {"step": 0, "weight": 0.1},
          {"step": 1200, "weight": 0.08},
          {"step": 2400, "weight": 0.06},
          {"step": 3600, "weight": 0.04},
          {"step": 4800, "weight": 0.02},
          {"step": 6000, "weight": 0.0},
        ],
      },
    )
```

Then in the `return ManagerBasedRlEnvCfg(...)` call, change `curriculum={},` (line 250) to:

```python
    curriculum=curriculum,
```

- [ ] **Step 3b: Thread the flag through `z1_hammer_env_cfg` + wire the head site**

In `src/tasks/hammer/config/z1/env_cfgs.py`:

Change the signature (line 22) from:
```python
def z1_hammer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
```
to:
```python
def z1_hammer_env_cfg(play: bool = False, imitation: bool = False) -> ManagerBasedRlEnvCfg:
```

Change line 24 from:
```python
  cfg = make_hammer_env_cfg()
```
to:
```python
  cfg = make_hammer_env_cfg(imitation=imitation)
```

After the `impact_progress` site-wiring block (after line 81), insert:

```python
  # --- Wire r_imit reward head site name (A-TRACK arm only; mirrors approach) ---
  if imitation:
    cfg.rewards["r_imit"].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)
```

(The existing `play` block already sets `cfg.curriculum = {}`, which correctly drops the
anneal in play/validation mode while leaving `r_imit` at its fixed 0.1 weight — no change
needed there.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_configs.py::TestImitationFlag -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tasks/hammer/hammer_env_cfg.py src/tasks/hammer/config/z1/env_cfgs.py tests/test_configs.py
git commit -m "feat(env): imitation flag wires r_imit + anneal curriculum (A-TRACK arm)"
```

---

### Task 3: Register the `Unitree-Z1-Hammer-Track` gym task

**Files:**
- Modify: `src/tasks/hammer/config/z1/__init__.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_configs.py`, add to `TestImitationFlag`:

```python
    def test_track_task_registered(self):
        import gymnasium as gym
        import src.tasks.hammer.config.z1  # noqa: F401 - triggers registration
        assert "Unitree-Z1-Hammer-Track" in gym.registry
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_configs.py::TestImitationFlag::test_track_task_registered -q`
Expected: FAIL — task id not in registry.

(If `register_mjlab_task` uses a registry other than `gym.registry`, adjust the assertion
to match how `Unitree-Z1-Hammer` is found — read `register_mjlab_task` in
`mjlab.tasks.registry` and mirror its lookup. The intent: the Track id is registered.)

- [ ] **Step 3: Add the registration**

In `src/tasks/hammer/config/z1/__init__.py`, after the existing `register_mjlab_task(...)` block, append:

```python
register_mjlab_task(
    task_id="Unitree-Z1-Hammer-Track",
    env_cfg=z1_hammer_env_cfg(imitation=True),
    play_env_cfg=z1_hammer_env_cfg(play=True, imitation=True),
    rl_cfg=z1_hammer_ppo_runner_cfg(),
    runner_cls=HammerOnPolicyRunner,
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n unitree_mjlab python -m pytest tests/test_configs.py::TestImitationFlag -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/hammer/config/z1/__init__.py tests/test_configs.py
git commit -m "feat(env): register Unitree-Z1-Hammer-Track (A-TRACK arm)"
```

---

### Task 4: Extend `validate_rewards.py` with Phase K (r_imit end-to-end)

**Files:**
- Modify: `docs/research/reward-design/validate_rewards.py`

- [ ] **Step 1: Build the validation env with imitation enabled**

Change line 101 from:
```python
  cfg = z1_hammer_env_cfg(play=True)
```
to:
```python
  cfg = z1_hammer_env_cfg(play=True, imitation=True)
```

(Phases A–J read terms by name and are unaffected by the extra term. In play mode the
curriculum is cleared, so `r_imit` weight is fixed at 0.1.)

- [ ] **Step 2: Add Phase K** immediately before the `# --- Summary table ---` block (before line 300):

```python
  # --- Phase K: imitation prior r_imit (T2) ---
  # K1: right after reset the head == the reference anchor (phi=0 -> waypoint=head0),
  #     so the Gaussian is 1 and (pre-contact) r_imit == its weight.
  # K2: driving down, from the first contact onward the ante-impact latch zeroes it.
  # K3: budget rule -- cumulative weighted r_imit over the strike < 0.35 * completion.
  print("\n--- Phase K: imitation prior (anchored, ante-impact latch, budget) ---")
  env.reset()
  W_IMIT = get_weights(env)["r_imit"]
  r = recompute_rewards(env)
  assert_close(r["r_imit"], W_IMIT,
               f"K1: r_imit should be ~{W_IMIT} (Gaussian=1 at the reference anchor)",
               tol=max(0.01, W_IMIT * TOL_FRAC))
  print(f"  K1 PASS  r_imit={r['r_imit']:.4f} at the reference anchor (weight {W_IMIT})")

  env.reset()
  k_sensor = env.scene["hammer_nail_contact"]
  contacted = False
  budget = 0.0
  for step in range(40):
    env.step(down_action)
    r = reward_dict(env)
    budget += r["r_imit"]
    found = bool((k_sensor.data.found > 0).any())
    if found or contacted:
      contacted = True
      assert_zero(r["r_imit"],
                  f"K2.step{step}: r_imit must be 0 from first contact onward (ante-impact latch)")
    if env.episode_length_buf[0].item() == 0:
      break  # episode reset after the strike drove the nail home
  if not contacted:
    print("\n[FAIL] K2: no contact within 40 steps; cannot verify the ante-impact latch")
    sys.exit(1)
  print(f"  K2 PASS  r_imit latched to 0 from first contact; pre-contact budget={budget:.4f}")

  budget_cap = 0.35 * W_COMPLETION
  if budget >= budget_cap:
    print(f"\n[FAIL] K3: imitation budget {budget:.4f} >= 0.35*completion ({budget_cap:.1f})")
    sys.exit(1)
  print(f"  K3 PASS  imitation budget {budget:.4f} < 0.35*completion ({budget_cap:.1f})")
  summary.append(("K. Imitation prior", {}))
```

Also update the module docstring phase list (after the Phase J line, ~line 20) by adding:
```
    K. Imitation prior (T2)       -> r_imit ~= weight at the anchor, 0 from first contact,
                                     cumulative budget < 0.35 * completion
```

- [ ] **Step 3: Run validate_rewards to verify all phases (incl. K) pass**

Run: `conda run -n unitree_mjlab python docs/research/reward-design/validate_rewards.py 2>&1 | grep -E "PASS|FAIL|ALL PHASES"`
Expected: `ALL PHASES PASSED`, including `K1/K2/K3 PASS`.

- [ ] **Step 4: Commit**

```bash
git add docs/research/reward-design/validate_rewards.py
git commit -m "test(validate): add Phase K (r_imit anchor / ante-impact latch / budget)"
```

---

### Task 5: Full local pre-training gate (both arms)

**Files:** none (verification only).

- [ ] **Step 1: Run the unit suite**

Run: `conda run -n unitree_mjlab python -m pytest -q`
Expected: PASS (all prior tests + the new imitation + config tests). Record the count.

- [ ] **Step 2: Run validate_rewards (A-TRACK env; covers Phases A–K)**

Run: `conda run -n unitree_mjlab python docs/research/reward-design/validate_rewards.py 2>&1 | tail -5`
Expected: `ALL PHASES PASSED`.

- [ ] **Step 3: Run the sensor + setup checks (A-BASE)**

Run:
```bash
conda run -n unitree_mjlab python docs/research/reward-design/verify_contact_sensor.py 2>&1 | tail -2
conda run -n unitree_mjlab python docs/research/reward-design/verify_reward_setup.py 2>&1 | tail -3
```
Expected: ContactSensor verified; "Reward setup verified. Safe to start training." (`r_imit` is NOT silent under a random policy — it fires pre-contact — so it needs no LIVENESS_EXEMPT entry; if `verify_reward_setup` flags it, it is run on the A-BASE env which has no `r_imit`, so this should not occur.)

- [ ] **Step 4: Commit (if anything changed) / otherwise note green**

No code change expected. If the gate is green, proceed. If `verify_reward_setup` is desired on the Track env too, run it against `Unitree-Z1-Hammer-Track` and confirm `r_imit` fires.

---

### Task 6: Push, then submit A-BASE + A-TRACK on Vega (500 iters, 3 seeds)

**Files:**
- Modify (optional): `scripts/slurm/train_array.sbatch` only if a per-arm task-id override is needed (see below).

- [ ] **Step 1: Push the branch**

```bash
git push origin hammer-z1
```

- [ ] **Step 2: Sync Vega to the new commit**

```bash
ssh vega 'cd ~/repos/unitree_rl_mjlab && git pull --ff-only origin hammer-z1 && git log --oneline -1'
```

- [ ] **Step 3: Confirm `train_array.sbatch` can select the task id + iters**

`train_array.sbatch` runs `scripts/train.py Unitree-Z1-Hammer ... --agent.max-iterations $ITERS`. To
train A-TRACK, the task id must be overridable. Read `train_array.sbatch`; if the task id is
hard-coded, add a `TASK="${TASK:-Unitree-Z1-Hammer}"` env var and substitute it into the
`train.py` invocation (one-line change; do NOT touch reward/env code). Commit + push that
infra change if made.

- [ ] **Step 4: Submit both arms (6 one-GPU tasks)**

```bash
ssh vega 'cd ~/repos/unitree_rl_mjlab && \
  ITERS=500 RUN=a_base  TASK=Unitree-Z1-Hammer       sbatch --array=0-2 scripts/slurm/train_array.sbatch && \
  ITERS=500 RUN=a_track TASK=Unitree-Z1-Hammer-Track sbatch --array=0-2 scripts/slurm/train_array.sbatch && \
  squeue -u $USER'
```
Expected: 6 jobs queued/running (a_base seeds 0–2, a_track seeds 0–2), one A100 each.

- [ ] **Step 5: Monitor + pull the result**

Watch `squeue -u $USER`; on completion compare success rate, mean episode length, and the
per-term episode rewards (incl. `r_imit` decaying to ~0 by iter 250) between `a_base` and
`a_track`. Pull a checkpoint/rollout back to answer: does the tracking prior speed up /
change the strike, or does the policy still slam/press? Record in
`docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` + `OPEN_QUESTIONS.md`.

---

## Self-review notes

- **Spec coverage:** weak+annealed (Task 1 term + Task 2 curriculum), T2-only / keep impact_progress (Task 2 adds only `r_imit`; `test_track_adds_only_r_imit` enforces it), position-only ante-impact latch (Task 1 + tests), mjlab reward_curriculum anneal to 0 by step 6000 (Task 2 + `test_track_anneal_curriculum_present`), A-BASE byte-identical when off (`test_base_has_no_r_imit_or_curriculum`), 500 iters / 3 seeds / parallel (Task 6), threshold 0.027 already in place.
- **Deferred (not in this plan, by scope decision):** the bespoke degenerate-tracking metrics (deviation-norm, beat-the-reference, press-watchdog) and T3/T4/T5. The first A-TRACK-vs-A-BASE read uses success rate / episode length / per-term episode rewards / rollout inspection. Add the metrics if the first run needs finer "anchor vs cage" diagnosis (augment-not-replace).
- **Known risk (carried into the term docstring):** hover-at-apex can accumulate `r_imit` without contacting; mitigated by anneal + completion dominance; budget assertion (K3) validates the non-pathological strike; watch the training curves.
- **Type consistency:** `ImitationPriorTerm.__call__(env, sensor_name, robot_cfg, nail_cfg, sigma)` matches the params dict in Task 2 and the stub `_PARAMS` in Task 1. `reward_curriculum` + `CurriculumTermCfg` imports verified against the installed mjlab.

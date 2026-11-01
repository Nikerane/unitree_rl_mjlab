"""Offline source census and exact proposed controller allocation contract."""
import ast
from pathlib import Path

import pytest

from scripts.z1_depth_campaign031 import lifecycle, manifest, worker


CANONICAL = Path(__file__).resolve().parents[1] / 'docs/research/reward-design/validate_rewards.py'


def public_reset_census(source):
    """Count the current canonical success path; reject unreviewed reset flow.

    This deliberately narrow AST check never imports/executes the scientific
    module. Only direct full public resets and literal finite loops are known.
    Native in-step _reset_idx/autoresets are outside this public-call census.
    """
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    assert len(mains) == 1, 'one canonical main required'
    main = mains[0]
    assert not any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) for node in ast.walk(main)), 'unknown main exit'
    configs = {}
    environments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and target.attr == 'num_envs':
                assert parents[node] is main, 'conditional world count'
                assert isinstance(target.value, ast.Attribute) and target.value.attr == 'scene'
                assert isinstance(target.value.value, ast.Name)
                name = target.value.value.id
                assert name not in configs, 'world count reassigned'
                worlds = ast.literal_eval(node.value)
                assert type(worlds) is int and worlds > 0
                configs[name] = worlds
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ManagerBasedRlEnv':
                assert parents[node] is main and isinstance(target, ast.Name), 'unknown constructor flow'
                assert isinstance(node.value.args[0], ast.Name), 'unknown constructor configuration'
                assert target.id not in environments, 'environment reassigned'
                environments[target.id] = configs[node.value.args[0].id]
    resets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != 'reset':
            continue
        call = parents[node]
        assert isinstance(call, ast.Call) and call.func is node and not call.args and not call.keywords, 'unknown/partial public reset'
        assert isinstance(node.value, ast.Name) and node.value.id in environments, 'unknown reset receiver'
        statement = parents[call]
        assert isinstance(statement, ast.Expr) and statement.value is call, 'reset must be direct'
        multiplicity = 1
        owner = parents[statement]
        while owner is not main:
            assert isinstance(owner, ast.For), 'unknown reset control flow'
            assert not owner.orelse and not any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(owner)), 'reset loop may exit early'
            values = ast.literal_eval(owner.iter)
            assert isinstance(values, (tuple, list)), 'unknown reset loop multiplicity'
            multiplicity *= len(values)
            owner = parents[owner]
        resets.append((node.value.id, multiplicity, environments[node.value.id]))
    return environments, resets


def test_recipe_reset_worlds_match_actual_canonical_loop_census():
    # Reverting the manifest to ten fails against actual source multiplicity.
    environments, resets = public_reset_census(CANONICAL.read_text())
    assert environments == {'env': 1, 'imp_env': 1}
    assert sum(m for env, m, _ in resets if env == 'env' and m == 1) == 8
    assert [(env, m) for env, m, _ in resets if m > 1] == [('env', 2)]
    assert [(m, worlds) for env, m, worlds in resets if env == 'imp_env'] == [(1, 1)]
    gate = manifest.qualification_recipe()['gates'][0]
    assert gate['constructor_worlds'] == sum(environments.values())
    assert gate['reset_worlds'] == sum(m * worlds for _, m, worlds in resets)


@pytest.mark.parametrize('replacement', [
    '  if condition:\n    env.reset()',
    '  while condition:\n    env.reset()',
    '  for item in unknown_values:\n    env.reset()',
    '  for item in (1, 2):\n    env.reset()\n    break',
    '  env.reset(env_ids=[0])',
    '  saved_reset = env.reset\n  saved_reset()',
])
def test_census_rejects_unknown_reset_control_flow(replacement):
    changed = CANONICAL.read_text().replace('  env.reset()', replacement, 1)
    with pytest.raises((AssertionError, ValueError)):
        public_reset_census(changed)


def test_census_follows_changed_literal_loop_multiplicity():
    changed = CANONICAL.read_text().replace('(0.063, 0.10)', '(0.063, 0.10, 0.12)')
    _, resets = public_reset_census(changed)
    assert sum(m * worlds for _, m, worlds in resets) == 12


def controller_row():
    return dict(Account='d2026d06-166-users', QOS='normal', Partition='cpu',
                NumCPUs='2', NumNodes='1', TimeLimit='18:00:00',
                AllocTRES='cpu=2,mem=2G,node=1,billing=2')


def test_real_controller_resource_guard_accepts_exact_two_cpu_shape():
    lifecycle.require_resource_shape(controller_row(), 'depth031-controller')


@pytest.mark.parametrize('changes', [
    dict(NumCPUs='1', AllocTRES='cpu=1,mem=2G,node=1'),
    dict(NumCPUs='3', AllocTRES='cpu=3,mem=2G,node=1'),
    dict(AllocTRES='cpu=1,mem=2G,node=1'),
    dict(AllocTRES='cpu=3,mem=2G,node=1'),
    dict(AllocTRES='cpu=2,mem=8G,node=1'),
    dict(TimeLimit='09:00:00'),
])
def test_real_controller_resource_guard_rejects_inexact_shape(changes):
    with pytest.raises(ValueError):
        lifecycle.require_resource_shape(dict(controller_row(), **changes), 'depth031-controller')


def test_case_shape_keeps_exact_two_cpu_eight_gib_135_minutes():
    row = dict(controller_row(), TimeLimit='02:15:00', AllocTRES='cpu=2,mem=8G,node=1,billing=8')
    lifecycle.require_resource_shape(row, 'depth031-eval-0')


def test_controller_plan_and_limits_reserve_36_allocated_cpu_hours():
    value = manifest.template()
    value['python'] = dict(path='/pinned/python')
    plan = worker.submission_plan(value)
    for shape in (value['limits'], plan):
        assert (shape['controller_cpus'], shape['controller_memory_gib'], shape['controller_seconds']) == (2, 2, 18 * 3600)
    assert value['limits']['maximum_controller_cpu_hours'] == plan['controller_cpu_hours_bound'] == 36
    assert plan['controller_cpu_hours_bound'] == plan['controller_cpus'] * plan['controller_seconds'] / 3600
    assert plan['case_cpu_hours_bound'] == 24 * 2 * 135 / 60 == 108
    assert plan['case_cpu_hours_bound'] + plan['controller_cpu_hours_bound'] == 144
    assert value['limits']['eval_concurrency'] * value['limits']['eval_cpus'] + plan['controller_cpus'] == 16

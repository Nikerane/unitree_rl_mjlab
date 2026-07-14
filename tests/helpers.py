"""Shared test scaffolding.

``stub(cls, **attrs)`` — the loud-failure replacement for bare ``object.__new__`` fakes
(adversarial-review R0.3, 2026-07-14). The suite's stub factories deliberately bypass
``__init__`` (no real env), hand-setting the instance fields. That pattern breaks SILENTLY
whenever ``__init__`` gains a field: the stub keeps passing until some code path reads the
missing attribute — or worse, never reads it in tests but does in training. This has fired
twice (``SubstepImpulseAccumulator._window``, ``ContactRowImpulseAccumulator._enabled``) and
one stub was gapped in-tree (``CatSoftHook`` fakes missing all five ``_imp_*`` fields).

``stub`` walks ``cls.__init__``'s AST for ``self.X = ...`` assignments and refuses to build
the object unless every such field is provided — turning the silent-break mode into an
immediate red test naming the missing fields. Extra attrs (fakes the class never assigns,
e.g. pre-resolved ``SceneEntityCfg`` stand-ins) are allowed.

Note ``ManagerTermBase.__init__`` (``self._env``) is a *super* call and thus not part of the
walked source — stubs don't set ``_env``, and no stubbed method reads it. ``setattr(env, ...)``
stash calls are not ``self.X`` assignments and are correctly ignored.
"""

from __future__ import annotations

import ast
import inspect
import textwrap


def _init_assigned_fields(cls) -> set[str]:
  """Every attribute name ``cls.__init__`` assigns on ``self`` (AST walk, own body only)."""
  src = textwrap.dedent(inspect.getsource(cls.__init__))
  tree = ast.parse(src)
  fields: set[str] = set()

  def collect(target: ast.expr) -> None:
    if (
      isinstance(target, ast.Attribute)
      and isinstance(target.value, ast.Name)
      and target.value.id == "self"
    ):
      fields.add(target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
      for t in target.elts:
        collect(t)

  for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
      for t in node.targets:
        collect(t)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
      collect(node.target)
  return fields


def stub(cls, **attrs):
  """``object.__new__(cls)`` with every ``__init__``-assigned field required in ``attrs``."""
  missing = _init_assigned_fields(cls) - attrs.keys()
  if missing:
    raise AssertionError(
      f"stub({cls.__name__}) is missing __init__ field(s) {sorted(missing)}: the real __init__ "
      f"assigns them, so this stub has drifted from the class (the `_window`/`_enabled` silent-"
      f"break bug class). Set them explicitly to the class defaults."
    )
  obj = object.__new__(cls)
  for k, v in attrs.items():
    setattr(obj, k, v)
  return obj

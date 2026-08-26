"""Public behavior tests for the trained-policy reference-line overlay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import tyro

import scripts.play as play_script
from src.tasks.hammer.mdp.references import SingleStrikeReference


class _Folder:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Checkbox:
    def __init__(self, value: bool):
        self.value = value
        self.callback = None

    def on_update(self, callback):
        self.callback = callback
        return callback


class _Gui:
    def __init__(self):
        self.checkbox = None

    def add_folder(self, _label):
        return _Folder()

    def add_checkbox(self, _label, initial_value):
        self.checkbox = _Checkbox(initial_value)
        return self.checkbox


class _LineHandle:
    def __init__(self, points):
        self.points = points
        self.visible = True


class _SceneApi:
    def __init__(self):
        self.calls = []
        self.handle = None

    def add_line_segments(self, name, **kwargs):
        self.calls.append((name, kwargs))
        self.handle = _LineHandle(kwargs["points"])
        return self.handle


class _Server:
    def __init__(self):
        self.gui = _Gui()
        self.scene = _SceneApi()


def _reference() -> SingleStrikeReference:
    ref = SingleStrikeReference(1, "cpu", overshoot=0.05, horizontal_detour_m=0.02)
    head = torch.tensor([[0.50, 0.00, 0.20]])
    nail = torch.tensor([[0.50, 0.00, 0.102]])
    ref.update(head, nail, torch.zeros(1, dtype=torch.long))
    ref.set_route_signs(torch.tensor([1]))
    return ref


def _viewer(monkeypatch, ref, policy=None):
    monkeypatch.setattr(
        play_script.ViserPlayViewer,
        "setup",
        lambda self: setattr(
            self,
            "_scene",
            SimpleNamespace(env_idx=0, _scene_offset=np.zeros(3)),
        ),
    )
    env = SimpleNamespace(cfg=SimpleNamespace(viewer=SimpleNamespace()))
    server = _Server()
    viewer = play_script.ReferenceLineViewer(
        env,
        policy or (lambda obs: obs),
        reference=ref,
        viser_server=server,
    )
    return viewer, server


def test_trained_playback_shows_a_thin_red_reference_line_by_default(monkeypatch):
    assert play_script.PlayConfig().show_reference_line is True
    viewer, server = _viewer(monkeypatch, _reference())

    viewer.setup()

    assert len(server.scene.calls) == 1
    name, style = server.scene.calls[0]
    assert name == "/reference_path"
    assert style["colors"] == (230, 30, 15)
    assert style["line_width"] == 1.5
    assert style["points"].shape == (64, 2, 3)
    assert server.gui.checkbox.value is True


def test_reference_line_refreshes_after_a_new_anchor_without_stepping(monkeypatch):
    class Policy:
        calls = 0

        def __call__(self, obs):
            self.calls += 1
            return obs

    ref = _reference()
    policy = Policy()
    viewer, server = _viewer(monkeypatch, ref, policy)
    viewer.setup()
    original = server.scene.handle.points.copy()

    shifted_head = torch.tensor([[0.50, 0.04, 0.20]])
    shifted_nail = torch.tensor([[0.50, 0.04, 0.102]])
    ref.reset()
    ref.update(shifted_head, shifted_nail, torch.zeros(1, dtype=torch.long))
    changed = viewer.refresh_reference_line()

    assert changed is True
    assert policy.calls == 0
    assert not np.array_equal(server.scene.handle.points, original)
    assert np.allclose(server.scene.handle.points[..., 1], 0.04)


def test_reference_line_gui_toggle_hides_and_restores_the_line(monkeypatch):
    viewer, server = _viewer(monkeypatch, _reference())
    viewer.setup()

    server.gui.checkbox.value = False
    server.gui.checkbox.callback(None)
    assert server.scene.handle.visible is False
    server.gui.checkbox.value = True
    server.gui.checkbox.callback(None)
    assert server.scene.handle.visible is True


def test_cli_can_explicitly_disable_the_default_reference_line():
    cfg = tyro.cli(
        play_script.PlayConfig,
        args=["--no-show-reference-line"],
        default=play_script.PlayConfig(),
    )

    assert cfg.show_reference_line is False


def test_viser_viewer_factory_uses_overlay_by_default_and_plain_viewer_when_disabled(
    monkeypatch,
):
    ref = _reference()
    raw = SimpleNamespace(_strike_reference=ref)
    env = SimpleNamespace(unwrapped=raw)
    calls = []

    monkeypatch.setattr(
        play_script,
        "ReferenceLineViewer",
        lambda *args, **kwargs: calls.append(("reference", args, kwargs)) or "red",
    )
    monkeypatch.setattr(
        play_script,
        "ViserPlayViewer",
        lambda *args, **kwargs: calls.append(("plain", args, kwargs)) or "plain",
    )

    assert play_script.build_viser_play_viewer(env, "policy", True) == "red"
    assert calls[-1][2]["reference"] is ref
    assert play_script.build_viser_play_viewer(env, "policy", False) == "plain"

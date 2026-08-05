"""Wave-2 pooled-table contract.

Two things must hold before any Wave-2 number is reported:
  1. the pooled generator reproduces the frozen Wave-1 columns byte-for-byte;
  2. gate completion is counted at contact onset, not over the full rollout.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "evaluation/results/2026-08-03_wave2_waypoint/build_pooled_table.py"
FROZEN_CSV = (
    REPO
    / "evaluation/results/2026-08-02_wave1_waypoint/tables/wave1_six_policy_comparison.csv"
)
FROZEN_SHA = "4ab58b67658ef6c899a21ffbd200d4c568327352c72da0dd1c386437dda86268"


def _load():
    spec = importlib.util.spec_from_file_location("build_pooled_table", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gates_at_contact_onset_ignores_gates_swept_during_follow_through():
    """A gate crossed after the nail is struck is follow-through, not guidance."""
    mod = _load()
    gate_index = np.array([0, 1, 2, 2, 3, 4, 5, 6])
    contact = np.array([False, False, False, True, True, True, True, True])

    assert mod.gates_at_contact_onset(gate_index, contact) == 2
    assert int(gate_index.max()) == 6


def test_gates_at_contact_onset_uses_the_whole_rollout_without_contact():
    """A policy that never contacts has no onset, so its window is everything."""
    mod = _load()
    gate_index = np.array([0, 1, 2, 3])
    contact = np.zeros(4, dtype=bool)

    assert mod.gates_at_contact_onset(gate_index, contact) == 3


def test_gates_at_contact_onset_includes_the_onset_sample_itself():
    """Pre-contact is defined inclusive of the first contact-onset sample."""
    mod = _load()
    gate_index = np.array([0, 0, 6, 6])
    contact = np.array([False, False, True, True])

    assert mod.gates_at_contact_onset(gate_index, contact) == 6


def test_precontact_window_is_reset_through_onset_inclusive():
    """The window definition is shared by every pre-contact metric."""
    mod = _load()
    contact = np.array([False, False, False, True, True])

    assert mod.precontact_slice(contact) == slice(0, 4)
    assert mod.precontact_slice(np.zeros(5, dtype=bool)) == slice(0, 5)


@pytest.mark.skipif(
    not (REPO / "evaluation/results/2026-08-02_wave1_waypoint/videos").is_dir(),
    reason="frozen Wave-1 artifacts not present",
)
def test_pooled_generator_reproduces_the_frozen_wave1_projection_byte_for_byte(tmp_path):
    """Extending 6 -> 18 rows must not perturb a single Wave-1 cell."""
    mod = _load()
    rows = mod.build_rows(mod.WAVE1_ORDER, campaign="wave1")
    projection = tmp_path / "projection.csv"
    mod.write_csv(projection, rows, columns=mod.FROZEN_COLUMNS)

    assert hashlib.sha256(projection.read_bytes()).hexdigest() == FROZEN_SHA
    assert projection.read_bytes() == FROZEN_CSV.read_bytes()


def test_frozen_column_list_matches_the_frozen_csv_header():
    """The projection is only meaningful if it spans the frozen header exactly."""
    mod = _load()
    header = FROZEN_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert list(mod.FROZEN_COLUMNS) == header


def test_pooled_columns_extend_the_frozen_ones_without_reordering():
    """New columns are appended; a pre-existing column never moves or changes."""
    mod = _load()

    assert list(mod.POOLED_COLUMNS[: len(mod.FROZEN_COLUMNS)]) == list(mod.FROZEN_COLUMNS)
    assert "gates_at_contact_onset" in mod.POOLED_COLUMNS
    assert "campaign" in mod.POOLED_COLUMNS


def test_precontact_path_ratio_is_defined_without_contact(tmp_path):
    """The straight label's third clause must be computable for every policy.

    A policy can cross all six gates and stop short of the nail -- the
    guidance-succeeds/task-fails cell this study exists to detect. The frozen
    `traj_path_len_ratio` leaves that blank, so a superset column is added
    rather than the frozen one changed.
    """
    mod = _load()
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]])
    contact = np.zeros(3, dtype=bool)
    entry = np.zeros(3)

    ratio = mod.precontact_path_ratio(pos, contact, entry)

    assert ratio == pytest.approx(2.0 / np.sqrt(2.0))


def test_precontact_path_ratio_matches_the_frozen_ratio_when_contact_exists():
    """Where the frozen column is defined, the superset column must agree."""
    mod = _load()
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [9.0, 9.0, 9.0]])
    contact = np.array([False, False, True, True])
    entry = np.zeros(3)

    assert mod.precontact_path_ratio(pos, contact, entry) == pytest.approx(1.0)


def test_precontact_path_ratio_is_nan_only_when_the_chord_is_degenerate():
    """A zero chord cannot form a ratio; that is the single undefined case."""
    mod = _load()
    pos = np.array([[0.0, 0, 0], [0.0, 0, 0]])
    contact = np.zeros(2, dtype=bool)

    assert np.isnan(mod.precontact_path_ratio(pos, contact, np.zeros(3)))


def test_build_rows_shares_one_precontact_window_with_the_tested_helper():
    """Every pre-contact metric must use the interval the tests actually cover."""
    mod = _load()
    rows = mod.build_rows(mod.WAVE1_ORDER, campaign="wave1")

    for row, (arm, seed) in zip(rows, mod.WAVE1_ORDER, strict=True):
        trace = dict(
            np.load(
                mod.WAVE1_ROOT / "videos" / f"wave1_{mod.SHORT[arm]}_seed{seed}"
                / "trace.npz"
            )
        )
        contact = trace["substep_contact"].astype(bool)
        window = mod.precontact_slice(contact)
        assert row["precontact_window_substeps"] == window.stop
        assert row["gates_at_contact_onset"] == int(
            trace["substep_gate_index"][window].max()
        )


def test_only_campaigns_predating_the_device_field_are_exempt():
    """The exemption is a closed historical list, never a default for new campaigns.

    wave1/fq4x8/fq3x8 artifacts were rendered before ``execution_device`` existed, so
    requiring it would reject them retroactively. Every campaign from wave2 onward MUST
    record the device directly. Stated positively on purpose: the earlier subtractive
    form (``all campaigns - {"wave2"}``) silently classified every newly registered
    campaign as pre-wave1, which would have exempted wave3 from recording its device.
    """
    from evaluation.analysis.fixed_reset_video_library import (
        EXECUTION_DEVICE_CAMPAIGN_EXEMPT,
        TASK_BY_CAMPAIGN_ARM,
    )

    assert EXECUTION_DEVICE_CAMPAIGN_EXEMPT == frozenset({"wave1", "fq4x8", "fq3x8"})
    registered = {campaign for campaign, _ in TASK_BY_CAMPAIGN_ARM}
    assert {"wave2", "wave3", "presentation3"} <= registered
    assert not ({"wave2", "wave3", "presentation3"} & EXECUTION_DEVICE_CAMPAIGN_EXEMPT)


def test_join_identity_is_checked_not_assumed(tmp_path):
    """A mis-named render directory must not silently join the wrong policy.

    Both halves of a row carry campaign/arm/seed/checkpoint/reset-digest. If the
    generator only trusts the directory name, one hand-named render out of twelve
    joins one policy's trajectory to another's provenance and nothing raises.
    """
    mod = _load()
    good = {
        "campaign": "wave2", "arm": "G", "training_seed": 5,
        "checkpoint_sha256": "a" * 64, "reset_state_digest": "b" * 64,
    }

    mod.assert_join_identity(good, dict(good), campaign="wave2", arm="G", seed=5,
                             checkpoint_sha256="a" * 64, leaf="leaf")

    swapped = dict(good, training_seed=6)
    with pytest.raises(ValueError, match="identity"):
        mod.assert_join_identity(good, swapped, campaign="wave2", arm="G", seed=5,
                                 checkpoint_sha256="a" * 64, leaf="leaf")
    with pytest.raises(ValueError, match="identity"):
        mod.assert_join_identity(good, dict(good), campaign="wave2", arm="G", seed=5,
                                 checkpoint_sha256="c" * 64, leaf="leaf")


def test_join_identity_rejects_a_different_fixed_reset(tmp_path):
    """Two halves from different resets are not the same physical rollout."""
    mod = _load()
    good = {
        "campaign": "wave2", "arm": "P", "training_seed": 4,
        "checkpoint_sha256": "a" * 64, "reset_state_digest": "b" * 64,
    }
    other = dict(good, reset_state_digest="c" * 64)

    with pytest.raises(ValueError, match="identity"):
        mod.assert_join_identity(good, other, campaign="wave2", arm="P", seed=4,
                                 checkpoint_sha256="a" * 64, leaf="leaf")


def test_write_csv_refuses_a_row_missing_a_declared_column(tmp_path):
    """A blank cell from a typo'd key would read as a measured value."""
    mod = _load()
    out = tmp_path / "t.csv"

    with pytest.raises(ValueError, match="missing"):
        mod.write_csv(out, [{"arm": "C0"}], columns=("arm", "gates_at_contact_onset"))

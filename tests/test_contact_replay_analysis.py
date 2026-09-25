import json

import pytest
import torch

from docs.results.analyze_contact_replay import analyze_condition


def _condition_artifact(path, *, observed_samples, success_sample, onset_sample=None):
    path.mkdir()
    samples = max(observed_samples, 1)
    valid = torch.zeros((samples, 1), dtype=torch.bool)
    valid[:observed_samples] = True
    onset = torch.zeros_like(valid)
    if onset_sample is not None:
        onset[onset_sample - 1, 0] = True
    zeros = torch.zeros((samples, 1, 6))
    data = {
        "valid": valid,
        "accepted_onset": onset,
        "native_success_sample": torch.tensor([success_sample or 0]),
        "joint_load_nms": zeros.clone(),
        "rolling_joint_load_nms": zeros.clone(),
        "control_step_peak_load_nms": zeros.clone(),
        "qfrc_constraint_nm": zeros.clone(),
        "saturated": torch.zeros_like(zeros, dtype=torch.bool),
        "off_contact_baseline_nm": zeros.clone(),
    }
    torch.save(data, path / "contact-trace.pt")
    (path / "contact-trace.json").write_text(json.dumps({"physics_dt_s": .00025}))
    measure = {
        "observed_samples": observed_samples,
        "native_success_sample": success_sample,
        "loads": {"episode": {"peak_nms": [0.] * 6, "exceeded": [False] * 6},
                  "cap_nms": [1.] * 6},
        "common_window": {},
    }
    if onset_sample is not None:
        measure["time_to_contact_s"] = onset_sample * .00025
    return [{"world": 0, "measurement": measure, "goal_mps": 1.5,
             "measured_mps": 1.4, "selection_count": 0}]


def test_zero_valid_samples_rejected_before_load_reduction(tmp_path):
    rows = _condition_artifact(tmp_path / "condition", observed_samples=0, success_sample=None)
    with pytest.raises(ValueError, match="zero valid samples"):
        analyze_condition(tmp_path / "condition", rows)


@pytest.mark.parametrize("success", [-1, 3])
def test_native_success_outside_observed_samples_rejected(tmp_path, success):
    rows = _condition_artifact(tmp_path / "condition", observed_samples=2, success_sample=success)
    with pytest.raises(ValueError, match="native success sample"):
        analyze_condition(tmp_path / "condition", rows)


@pytest.mark.parametrize("success", [None, 2])
def test_missing_or_onset_free_native_success_remains_separately_labeled(tmp_path, success):
    rows = _condition_artifact(tmp_path / "condition", observed_samples=2, success_sample=success)
    record = analyze_condition(tmp_path / "condition", rows)[0]
    assert record["native_success_sample"] == success
    assert record["onset_sample"] is None
    assert record["gain_intervals"] == {}

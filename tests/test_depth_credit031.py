"""Hand-derived native depth-credit contracts; no simulator construction."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def ledger(n=1, *, progress=True, completion=True):
    path = Path(__file__).parents[1] / "src/tasks/hammer/mdp/depth_credit031.py"
    assert path.exists(), "pure depth-credit ledger implementation missing"
    spec = importlib.util.spec_from_file_location("depth_credit031", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DepthCreditLedger031(
        n, "cpu", progress_enabled=progress, completion_enabled=completion
    )


def collect(owner, horizon=None, reset=None):
    n = owner.num_envs
    return owner.collect(
        horizon_complete=torch.tensor(horizon or [False] * n, dtype=torch.bool),
        native_reset=torch.tensor(reset or [False] * n, dtype=torch.bool),
    )


def values(receipt, key, expected):
    torch.testing.assert_close(
        receipt[key], torch.tensor(expected, dtype=receipt[key].dtype), rtol=0, atol=1e-7
    )


def test_literal_depths_pay_only_new_potential_and_rebound_cannot_repay():
    owner = ledger()
    for depth, potential, paid in [
        (0., 0., 0.), (.004, 0., 0.), (.017, .5, .5),
        (.030, 1., .5), (.001, 1., 0.), (.017, 1., 0.), (.040, 1., 0.),
    ]:
        owner.observe(torch.tensor([depth]))
        receipt = collect(owner)
        values(receipt, "progress_potential", [potential])
        values(receipt, "progress_paid", [paid])
        values(receipt, "progress_paid_total", [potential])
        values(receipt, "completion_paid", [0.])
    values(receipt, "current_depth_m", [.032])
    values(receipt, "peak_depth_m", [.032])
    assert receipt["attained"].tolist() == [True]


@pytest.mark.parametrize("progress,completion", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_unit_budgets_and_off_components(progress, completion):
    owner = ledger(progress=progress, completion=completion)
    owner.observe(torch.tensor([.030]))
    early = collect(owner)
    owner.observe(torch.tensor([.002]))
    final = collect(owner, [True], [True])
    values(early, "progress_paid", [float(progress)])
    values(final, "progress_paid", [0.])
    values(final, "progress_paid_total", [float(progress)])
    values(final, "completion_paid", [float(completion)])
    values(final, "completion_paid_total", [float(completion)])
    values(final, "completion_forfeited", [0.])
    assert final["attained"].tolist() == [True]


def test_crossing_after_50ms_between_policy_boundaries_survives_rebound_to_200ms():
    owner = ledger()
    # 4kHz samples; 20ms collections. Crossing at 70.25ms is later than the
    # accepted-strike deadline and absent in every policy-boundary sample.
    progress = completion = 0.
    for tick in range(1, 801):
        owner.observe(torch.tensor([.030 if tick == 281 else .004]))
        if tick % 80 == 0:
            receipt = collect(owner, [tick == 800], [tick == 800])
            progress += receipt["progress_paid"].item()
            completion += receipt["completion_paid"].item()
            if tick < 800:
                assert completion == 0.
    assert progress == completion == 1.
    values(receipt, "current_depth_m", [.004])
    values(receipt, "peak_depth_m", [.030])


def test_timeout_forfeits_completion_once_but_simultaneous_horizon_wins():
    owner = ledger(3)
    owner.observe(torch.tensor([.030, .030, .017]))
    receipt = collect(owner, [False, True, False], [True, True, True])
    values(receipt, "completion_due", [0., 1., 0.])
    values(receipt, "completion_paid", [0., 1., 0.])
    values(receipt, "completion_forfeited", [1., 0., 0.])
    assert receipt["completion_closed"].tolist() == [True, True, True]
    owner.observe(torch.tensor([.030, .030, .030]))
    later = collect(owner, [True, True, True], [True, True, True])
    values(later, "completion_paid", [0., 0., 0.])
    values(later, "completion_forfeited", [0., 0., 0.])
    values(later, "progress_paid", [0., 0., 0.])


def test_subset_external_reset_discards_only_selected_completion_and_starts_new_generation():
    owner = ledger(3)
    owner.observe(torch.tensor([.030, .030, .017]))
    first = collect(owner)
    owner.reset(torch.tensor([1]))
    values(owner.last_snapshot, "progress_paid", [1., 1., .5])
    owner.observe(torch.tensor([.001, .004, .017]))
    after = collect(owner)
    values(after, "generation", [0, 1, 0])
    values(after, "progress_paid_total", [1., 0., .5])
    values(after, "peak_depth_m", [.030, .004, .017])
    values(after, "externally_forfeited_completion", [0., 1., 0.])
    values(after, "resets_since_collect", [0, 1, 0])
    assert first["attained"].tolist() == [True, True, False]
    owner.observe(torch.tensor([.001, .030, .030]))
    final = collect(owner, [True] * 3, [True] * 3)
    values(final, "completion_paid", [1., 1., 1.])
    owner.reset()
    values(owner.last_snapshot, "completion_paid", [1., 1., 1.])
    next_receipt = collect(owner)
    values(next_receipt, "generation", [1, 2, 1])
    values(next_receipt, "externally_forfeited_completion", [0., 0., 0.])
    values(next_receipt, "progress_paid_total", [0., 0., 0.])


def test_external_reset_without_collection_cannot_pay_completion():
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    owner.reset()
    receipt = collect(owner, [True], [True])
    values(receipt, "completion_paid", [0.])
    values(receipt, "externally_forfeited_completion", [1.])
    values(receipt, "progress_paid", [0.])


def test_receipts_are_defensive_clones_and_duplicate_collection_is_idempotent():
    owner = ledger()
    sample = torch.tensor([.030])
    owner.observe(sample)
    sample.zero_()
    first = collect(owner, [True], [True])
    for tensor in first.values():
        tensor.zero_()
    cached = collect(owner, [True], [True])
    values(cached, "progress_paid", [1.])
    values(cached, "completion_paid", [1.])
    values(cached, "peak_depth_m", [.030])
    snapshot = owner.last_snapshot
    snapshot["progress_paid"].zero_()
    values(owner.last_snapshot, "progress_paid", [1.])
    owner.reset()
    values(cached, "completion_paid_total", [1.])
    values(owner.last_snapshot, "completion_paid", [1.])


def test_changed_masks_without_observation_are_rejected_before_payment():
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    collect(owner)
    with pytest.raises(ValueError):
        collect(owner, [True], [True])
    owner.observe(torch.tensor([.030]))
    values(collect(owner, [True], [True]), "completion_paid", [1.])


def test_horizon_without_native_reset_is_rejected_without_consuming_credit():
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    with pytest.raises(ValueError):
        collect(owner, [True], [False])
    receipt = collect(owner, [True], [True])
    values(receipt, "progress_paid", [1.])
    values(receipt, "completion_paid", [1.])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_substep_is_latched_until_collect_even_after_a_valid_sample(bad):
    owner = ledger()
    owner.observe(torch.tensor([bad]))
    owner.observe(torch.tensor([.030]))
    with pytest.raises(ValueError):
        collect(owner, [True], [True])
    owner.reset()
    owner.observe(torch.tensor([.017]))
    values(collect(owner), "progress_paid", [.5])


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("observe synchronized a tensor scalar to the host")
        return func(*args, **(kwargs or {}))


def test_observe_uses_no_host_scalar_reads_or_rng_and_retains_no_autograd_graph():
    owner = ledger()
    rng_before = torch.random.get_rng_state().clone()
    with NoHostScalar():
        owner.observe(torch.tensor([.030], requires_grad=True))
        owner.observe(torch.tensor([.004]))
    receipt = collect(owner, [True], [True])
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert all(not value.requires_grad for value in receipt.values())
    values(receipt, "completion_paid", [1.])


@pytest.mark.parametrize("bad", [[.01], torch.zeros(1, 1), torch.zeros(2), torch.ones(1, dtype=torch.int64)])
def test_malformed_depth_is_rejected_without_changing_history(bad):
    owner = ledger()
    with pytest.raises(ValueError):
        owner.observe(bad)
    values(collect(owner), "peak_depth_m", [0.])


@pytest.mark.parametrize("which", ["horizon_complete", "native_reset"])
@pytest.mark.parametrize("bad", [[False], torch.zeros(1), torch.zeros(2, dtype=torch.bool)])
def test_malformed_boundary_masks_do_not_consume_credit(which, bad):
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    masks = dict(horizon_complete=torch.ones(1, dtype=torch.bool), native_reset=torch.ones(1, dtype=torch.bool))
    masks[which] = bad
    with pytest.raises(ValueError):
        owner.collect(**masks)
    values(collect(owner, [True], [True]), "completion_paid", [1.])


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([1]), torch.tensor([0, 0]), torch.tensor([0.]), torch.tensor([[0]]), [0]])
def test_malformed_reset_ids_do_not_discard_credit(ids):
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    with pytest.raises(ValueError):
        owner.reset(ids)
    values(collect(owner, [True], [True]), "completion_paid", [1.])


def test_empty_subset_reset_is_noop():
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    owner.reset(torch.tensor([], dtype=torch.int64))
    receipt = collect(owner, [True], [True])
    values(receipt, "generation", [0])
    values(receipt, "completion_paid", [1.])


def test_finite_depth_clamps_to_physical_zero_and_32mm_limits():
    owner = ledger(3)
    owner.observe(torch.tensor([-.1, .030, .1]))
    receipt = collect(owner, [True] * 3, [True] * 3)
    values(receipt, "current_depth_m", [0., .030, .032])
    values(receipt, "peak_depth_m", [0., .030, .032])
    values(receipt, "progress_paid", [0., 1., 1.])
    values(receipt, "completion_paid", [0., 1., 1.])


def test_initial_scene_depth_is_credited_baseline_without_progress_payment():
    owner = ledger(3)
    owner.reset(initial_depth_m=torch.tensor([.017, .030, .1]))
    initial = collect(owner)
    values(initial, "current_depth_m", [.017, .030, .032])
    values(initial, "progress_potential", [.5, 1., 1.])
    values(initial, "progress_paid", [0., 0., 0.])
    values(initial, "progress_paid_total", [0., 0., 0.])
    assert initial["attained"].tolist() == [False, True, True]
    owner.observe(torch.tensor([.030, .001, .001]))
    final = collect(owner, [True] * 3, [True] * 3)
    values(final, "progress_paid", [.5, 0., 0.])
    values(final, "progress_paid_total", [.5, 0., 0.])
    values(final, "completion_paid", [1., 1., 1.])


def test_subset_initial_depth_uses_selected_order_and_preserves_other_worlds():
    owner = ledger(3)
    owner.observe(torch.tensor([.030, .030, .030]))
    collect(owner)
    owner.reset(torch.tensor([2, 0]), initial_depth_m=torch.tensor([.017, -.1]))
    receipt = collect(owner)
    values(receipt, "peak_depth_m", [0., .030, .017])
    values(receipt, "progress_potential", [0., 1., .5])
    values(receipt, "progress_paid_total", [0., 1., 0.])
    values(receipt, "generation", [1, 0, 1])
    values(receipt, "externally_forfeited_completion", [1., 0., 1.])


@pytest.mark.parametrize("bad", [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.zeros(2), torch.tensor([0]), [.017]])
def test_bad_initial_depth_does_not_reset_or_discard_existing_credit(bad):
    owner = ledger()
    owner.observe(torch.tensor([.030]))
    with pytest.raises(ValueError):
        owner.reset(initial_depth_m=bad)
    receipt = collect(owner, [True], [True])
    values(receipt, "completion_paid", [1.])
    values(receipt, "generation", [0])


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_environment_count_is_rejected(n):
    with pytest.raises(ValueError):
        ledger(n)


@pytest.mark.parametrize("which", ["progress_enabled", "completion_enabled"])
def test_component_switches_require_booleans(which):
    owner = ledger()
    kwargs = dict(progress_enabled=True, completion_enabled=True)
    kwargs[which] = 1
    with pytest.raises(ValueError):
        type(owner)(1, "cpu", **kwargs)

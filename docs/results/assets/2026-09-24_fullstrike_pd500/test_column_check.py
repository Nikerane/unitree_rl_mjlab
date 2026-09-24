import pytest
import torch
import gain_screen as subject


def test_matching_columns_accept_different_integer_storage():
    subject.validate_arm_columns(torch.tensor([0,1,2,3,4,5],dtype=torch.int64),
                                 torch.tensor([0,1,2,3,4,5],dtype=torch.int32))


def test_different_column_order_is_rejected():
    with pytest.raises(AssertionError):
        subject.validate_arm_columns(torch.tensor([0,1,2]),torch.tensor([0,2,1]))

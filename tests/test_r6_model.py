
import numpy as np
import torch
import pytest

from r6.common import LABELS
from r6.train import TemporalGRU, apply_cue_dropout, group_split
from r6.inference import validate_checkpoint_contract


def test_project_class_contract():
    assert LABELS == ["alert", "fatigue", "distraction"]


def test_gru_architecture_and_output():
    model = TemporalGRU(input_size=11, hidden_size=64, num_classes=3)

    x = torch.randn(4, 75, 11)
    output = model(x)

    assert model.gru.num_layers == 1
    assert model.gru.hidden_size == 64
    assert output.shape == (4, len(LABELS))
    assert torch.isfinite(output).all()


def test_cue_dropout():
    x = torch.ones(4, 75, 11)

    torch.manual_seed(42)
    result = apply_cue_dropout(x, p=1.0)

    assert torch.all(result[:, :, 0] == 0)
    assert torch.all(result[:, :, 1:6] == 0)
    assert torch.all(result[:, :, 6:10] == 0)
    assert torch.all(result[:, :, 10] == 1)


def test_subject_wise_split_isolation():
    groups = np.repeat(
        [f"S{i}" for i in range(1, 11)], 30
    )

    y = np.tile(
        np.repeat([0, 1, 2], 10), 10
    )

    train, val, test = group_split(groups, y, seed=42)

    train_subjects = set(groups[train])
    val_subjects = set(groups[val])
    test_subjects = set(groups[test])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_all_classes_present_in_splits():
    groups = np.repeat(
        [f"S{i}" for i in range(1, 11)], 30
    )

    y = np.tile(
        np.repeat([0, 1, 2], 10), 10
    )

    train, val, test = group_split(groups, y, seed=42)

    expected = set(range(len(LABELS)))

    assert set(y[train]) == expected
    assert set(y[val]) == expected
    assert set(y[test]) == expected


# Checkpoint validation tests

def valid_checkpoint():
    return {
        "labels": ["alert", "fatigue", "distraction"],
        "num_classes": 3,
        "input_size": 11,
        "feature_names": [f"feature_{i}" for i in range(11)],
        "window": 75,
        "temperature": 1.0,
    }


def test_valid_checkpoint_contract():
    validate_checkpoint_contract(valid_checkpoint())


def test_reject_old_checkpoint_labels():
    ckpt = valid_checkpoint()
    ckpt["labels"] = ["alert", "low_vigilant", "fatigue"]

    with pytest.raises(ValueError, match="do not match"):
        validate_checkpoint_contract(ckpt)


def test_reject_incorrect_class_count():
    ckpt = valid_checkpoint()
    ckpt["num_classes"] = 2

    with pytest.raises(ValueError, match="class count"):
        validate_checkpoint_contract(ckpt)


def test_reject_incorrect_input_size():
    ckpt = valid_checkpoint()
    ckpt["input_size"] = 10

    with pytest.raises(ValueError, match="input size"):
        validate_checkpoint_contract(ckpt)


def test_reject_invalid_temperature():
    ckpt = valid_checkpoint()
    ckpt["temperature"] = 0

    with pytest.raises(ValueError, match="temperature"):
        validate_checkpoint_contract(ckpt)

import numpy as np
import pytest

from anomaly.data.splits import split_train_val


def test_disjoint_and_complete():
    tr, va = split_train_val(50, 0.2, seed=0)
    assert len(va) == 10
    assert set(tr).isdisjoint(va)
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(50))


def test_deterministic():
    a = split_train_val(50, 0.2, seed=3)
    b = split_train_val(50, 0.2, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_different_seeds_give_different_splits():
    a = split_train_val(50, 0.2, seed=0)[1]
    b = split_train_val(50, 0.2, seed=1)[1]
    assert not np.array_equal(a, b)


def test_neither_side_is_ever_empty():
    tr, va = split_train_val(2, 0.9)
    assert len(tr) == 1 and len(va) == 1


def test_bad_fraction():
    with pytest.raises(ValueError):
        split_train_val(50, 1.5)

import numpy as np
from .moda import compute_moda


def test_perfect_box_accuracy():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    b = a.copy()
    moda = compute_moda(a, b)
    assert moda.value == 1


def test_no_predictions():
    a = np.array([])
    b = np.zeros((2, 4))
    b[:, 2:] += 1
    moda = compute_moda(a, b)
    assert moda.value == 0


def test_no_ground_truth():
    a = np.zeros((2, 4))
    a[:, 2:] += 1
    b = np.array([])
    moda = compute_moda(a, b)
    assert moda.value == 0

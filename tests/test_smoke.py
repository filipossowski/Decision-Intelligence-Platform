import numpy as np
from pymcdm.methods import TOPSIS


def test_topsis_runs():
    matrix = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 2]])
    weights = np.array([0.4, 0.3, 0.3])
    types = np.array([1, 1, -1])  
    prefs = TOPSIS()(matrix, weights, types)
    assert len(prefs) == 3
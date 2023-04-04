import numpy as np
from mylib.tree.my_decision_tree.mean_regression import mean_regression


def mean_squared_error(chi: np.array) -> float:
    target = chi[:, -1]
    return np.sum((target - mean_regression(chi)) ** 2) / len(target)

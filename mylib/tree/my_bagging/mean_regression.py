import numpy as np


def mean_regression(x: np.array) -> float:
    return np.sum(x) / len(x)

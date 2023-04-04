import numpy as np


def mean_regression(chi: np.array) -> float:
    target = chi[:, -1]
    return np.sum(target) / len(target)

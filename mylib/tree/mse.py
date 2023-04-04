import numpy as np


def mean_squared_error(chi: np.array) -> float:
    target = chi[:, -1]
    mean = np.mean(target)
    return np.sum((target - mean) ** 2) / len(target)

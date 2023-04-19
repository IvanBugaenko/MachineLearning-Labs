import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    res = y_true * np.log(y_pred)
    return np.where(res == -np.inf, 0, res)

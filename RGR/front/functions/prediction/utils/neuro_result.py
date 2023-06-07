import numpy as np


def neuro_result(X: np.ndarray) -> np.ndarray:
    return np.array([np.argmax(pred) for pred in X], dtype=bool)

import numpy as np


def cross_entropy_function(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    return -np.sum(X_true * np.log(X_pred))


def cross_entropy_derivative(X_true: float, X_pred: float) -> float:
    return -X_true / X_pred

import numpy as np


def MSE_function(X_true: float, X_pred: float) -> float:
    return (X_pred - X_true) ** 2


def MSE_derivative(X_true: float, X_pred: float) -> float:
    return 2 * (X_pred - X_true)

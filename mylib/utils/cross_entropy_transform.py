import numpy as np
from mylib.neural_network.apply import apply


def cross_entropy_transform(y_true: np.ndarray, n: int) -> np.ndarray:
    return apply(y_true, lambda x: np.array( [1 if i == x[0] else 0 for i in range(n)] ))

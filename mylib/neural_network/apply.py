import numpy as np
from typing import Callable


def apply(array: np.ndarray, function: Callable) -> np.ndarray:
    res = np.array(list(map(function, array)))
    return res

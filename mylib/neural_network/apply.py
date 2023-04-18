import numpy as np
from typing import Callable


def apply(array: np.ndarray, function: Callable) -> np.ndarray:
    f = function
    res = np.array(list(map(f, array)))
    return res

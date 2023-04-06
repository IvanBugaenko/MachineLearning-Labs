import numpy as np


def bootstrap(X: np.array, y: np.array, n: int) -> tuple:
    chi = np.c_[X, y]
    i = 0
    while i < n:
        fold = chi[np.random.choice(np.arange(len(chi)), 2)]
        i += 1
        yield fold[:, :-1], fold[:, -1]
        
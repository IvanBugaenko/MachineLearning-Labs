import numpy as np


def split_chi(chi: np.array, j: int, predicate_value: float):
    return chi[ chi[:, j] < predicate_value], chi[ chi[:, j] >= predicate_value]

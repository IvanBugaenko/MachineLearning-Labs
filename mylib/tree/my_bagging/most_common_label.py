import numpy as np
from scipy.stats import mode

def most_common_label(x: np.array) -> object:
    return mode(x).mode[0]

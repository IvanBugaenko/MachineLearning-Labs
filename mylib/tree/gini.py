import numpy as np
from mylib.tree.classes_prior_probability import classes_prior_probability


def gini(chi: np.array) -> float:
    target = chi[:, -1]
    priori = classes_prior_probability(target)
    gini = 0
    for c, p in priori:
        

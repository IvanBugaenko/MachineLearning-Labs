import numpy as np
from mylib.tree.my_decision_tree.classes_prior_probability import classes_prior_probability


def gini(chi: np.array) -> float:
    priori = classes_prior_probability(chi)
    gini = 0
    for c, p in priori.items():
        gini += p ** 2
    
    return 1 - gini

import numpy as np
from mylib.tree.my_decision_tree.node import Node
from scipy.stats import mode


def classes_prior_probability(chi: np.array) -> dict:

    info = np.unique(chi[:, -1], return_counts=True)

    classes_and_counts = []

    for i in range(len(info[0])):
        classes_and_counts.append((info[0][i], info[1][i]))

    n = chi.shape[0]

    priori = {}

    for cl, co in classes_and_counts:
        priori.update({cl: co / n})

    return priori

a = np.array([[1, 2, 3, 1],
              [4, 5, 6, 1],
              [7, 8, 9, 0]])

print(a[np.random.choice(np.arange(len(a)), 2)])

a, b, c = (1, (2, 3))
print(a, b, c)
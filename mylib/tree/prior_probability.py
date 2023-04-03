import numpy as np

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

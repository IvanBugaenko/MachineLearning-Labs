import numpy as np
from mylib.tree.my_bagging.bootstrap import bootstrap


def create_bagging_algorithm(estimator, n_estimators: int, X: np.array, y: np.array) -> list:

    estimators = [estimator() for i in range(n_estimators)]

    i = 0

    for X_boot, y_boot in bootstrap(X, y, n_estimators):
        estimators[i] = estimators[i].fit(X_boot, y_boot)
        i += 1

    return estimators

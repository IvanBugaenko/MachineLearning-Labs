import numpy as np
from typing import Any
from scipy.stats import mode


class MyKNN:
    X_train: np.array
    y_train: np.array


    def __init__(self, k: int = 7):
        self.k = k


    def fit(self, X_train: Any, y_train: Any) -> None:
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)


    def predict(self, X_test: Any) -> np.array:

        answer = []

        for x in np.array(X_test):
            
            d = np.array(np.apply_along_axis(np.linalg.norm, 1, self.X_train - x))
            
            d_c = np.c_[d, self.y_train]

            choise = np.array(sorted(d_c, key=lambda x: x[0]))

            choise = choise[:self.k, 1]

            answer.append(np.around(mode(choise)[0][0]))

        return np.array(answer)
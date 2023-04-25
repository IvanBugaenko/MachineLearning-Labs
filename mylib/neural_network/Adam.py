import numpy as np
from typing import List, Dict, Callable
from mylib.neural_network.dense import Dense


class Adam:

    def __init__(self, learning_rate: int = 0.01, batch_size: int = 1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.batch_size = batch_size


    def compile(self, loss_function: Dict[str, Callable]) -> None:
        self.loss_function = loss_function


    def __forward_propagation(self, X: np.ndarray, layers: List[Dense]) -> np.ndarray:
        res = X
        for layer in layers:
            res = np.around(layer.forward_propagation(res), 5)

        return res


    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, layers: List[Dense]) -> List[Dense]:
        temp_layers = layers
        chi: np.array = np.c_[X_train, y_train]
        n = len(np.unique(y_train))

        np.random.shuffle(chi)
        
        for i in range(len(chi) // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            X = chi[start:end, :-1]

            y = chi[start:end, -1].reshape(end - start, 1)

            y = self.loss_function["transform"](y, n)
            h = self.__forward_propagation(X, temp_layers)

            initial_dE_dt = np.around(np.mean(self.loss_function["derivative"](y, h), axis=0), 5)

            temp_layers = self.__update(initial_dE_dt, temp_layers)

        return temp_layers

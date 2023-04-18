import numpy as np
from typing import List
from mylib.neural_network.dence import Dense
import random


class Sequential:

    def __init__(self, layers: List[Dense]) -> None:
        self.layers = layers
        self.loss_functions = {
            "mse": {
                "function": lambda X_true, X_pred: 0.5 * (X_pred - X_true) ** 2,
                "derivative": lambda X_true, X_pred: X_pred - X_true
            },
            "cross_entropy": {
                "function": lambda X_true, X_pred: -np.sum(X_true * np.log(X_pred)),
                "derivative": lambda X_true, X_pred: -X_true / X_pred # TODO: Перепроверить производную
            }
        }


    def compile(self, optimizer: object, loss: str) -> None:
        """
        interface IOptimizer:
        {
            void Step();
            void optimize(); // 
        }
        """
        self.optimizer = optimizer # TODO: Оптимизаторы
        self.loss = loss

        for i in range(len(self.layers)):
            if self.layers[i].n:
                self.layers[i].initialize_weights()
                continue
            self.layers[i].n = self.layers[i - 1].m
            self.layers[i].initialize_weights()


    def __backward_propagation(self):
        ...


    def __forward_propagation(self, X: np.ndarray) -> np.ndarray:
        res = X
        for layer in self.layers:
            res = layer.forward_propagation(res)

        return res
    

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 1, batch_size: int = 1) -> None:
        chi: np.ndarray = np.c_[X_train, y_train]
        for _ in range(epochs):
            np.random.shuffle(chi)
            for i in range(len(chi) // batch_size):
                start = i * batch_size
                end = (i + 1) * batch_size

                batch = chi[start:end]

                error = 0

                for obj in batch:
                    error += self.loss_functions[self.loss]["function"](
                        self.__forward_propagation(obj[:-1]), obj[-1]
                    )
            

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__forward_propagation(X)

import numpy as np
from typing import List
from mylib.neural_network.dence import Dense
from mylib.neural_network.cross_entropy_transform import cross_entropy_transform
from mylib.neural_network.cross_entropy import cross_entropy


class Sequential:

    def __init__(self, layers: List[Dense]) -> None:
        self.layers = layers
        self.loss_functions = {
            "mse": {
                "function": lambda y_true, y_pred: (0.5 * (np.linalg.norm(y_true - y_pred, axis=1)) ** 2).reshape((len(y_pred), 1)),
                "derivative": lambda y_true, y_pred: y_true - y_pred,
                "transform": lambda *args: args[0]
            },
            "cross_entropy": {
                "function": cross_entropy,
                "derivative": lambda y_true, y_pred: -y_true / y_pred,       # TODO: Перепроверить производную
                "transform": cross_entropy_transform        # принимает у и количество классов n
            }
        }


    def compile(self, optimizer: object, loss: str) -> None:
        
        self.optimizer = optimizer
        self.loss = loss

        for i in range(len(self.layers)):
            if self.layers[i].n:
                self.layers[i].initialize_weights()
                continue
            self.layers[i].n = self.layers[i - 1].m
            self.layers[i].initialize_weights()


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 1) -> None:
        self.optimizer.compile(self.layers, self.loss_functions[self.loss])

        for _ in range(epochs):
            self.layers = self.optimizer.optimize(X_train, y_train)


    def predict(self, X: np.ndarray) -> np.ndarray:
        res = X
        for layer in self.layers:
            res = layer.forward_propagation(res)

        return res

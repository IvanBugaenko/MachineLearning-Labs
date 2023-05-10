import numpy as np
from typing import Dict, Callable, List
from mylib.lab5.layers.layer import Layer
from mylib.lab5.optimizers.optimizer import Optimizer
from mylib.utils.cross_entropy_transform import cross_entropy_transform


class Sequential:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

        self.loss_functions: Dict[str, Callable] = {
            "mse": {
                "function": lambda y_true, y_pred: 0.5 * (y_true - y_pred) ** 2,
                "derivative": lambda y_true, y_pred: y_true - y_pred,
                "transform": lambda *args: args[0]
            },
            "categorical_crossentropy": {
                "function": lambda y_true, y_pred: -np.sum(y_true * np.log(y_pred)),
                "derivative": lambda y_true, y_pred: y_true - y_pred,
                "transform": cross_entropy_transform
            }
        }

    def compile(self, optimizer: Optimizer, loss: str) -> None:
        self.optimizer = optimizer
        self.loss = loss

        previous_layer = self.layers[0]
        self.layers[0].initialize_weights()

        for layer in self.layers[1:]:
            layer.compile(previous_layer)
            previous_layer = layer
            layer.initialize_weights()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 1) -> None:
        self.optimizer.compile(self.loss_functions[self.loss])
        self.layers = self.optimizer.optimize(
            X_train, y_train, self.layers, epochs)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        res = X
        for layer in self.layers:
            res = layer.forward_propagation(res)

        return res

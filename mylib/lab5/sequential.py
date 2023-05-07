import numpy as np
from typing import List


class Sequential:

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.loss_functions = {
            "mse": {
                "derivative": lambda y_true, y_pred: y_true - y_pred,
                "transform": lambda *args: args[0]
            },
            "cross_entropy": {
                "derivative": lambda y_true, y_pred: y_true - y_pred,
                "transform": cross_entropy_transform
            }
        }
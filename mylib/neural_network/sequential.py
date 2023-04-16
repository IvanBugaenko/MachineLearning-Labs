import numpy as np
from typing import List
from mylib.neural_network.loss_functions.cross_entropy import cross_entropy_derivative, cross_entropy_function
from mylib.neural_network.loss_functions.mse import MSE_derivative, MSE_function
from mylib.neural_network.dence import Dense


class Sequential:

    def __init__(self, layers: List[Dense]) -> None:
        self.layers = layers
        self.dependencies = {
            "loss": {
                "mse": {
                    "function": MSE_function,
                    "derivative": MSE_derivative
                },
                "cross_entropy": {
                    "function": cross_entropy_function,
                    "derivative": cross_entropy_derivative
                }
            },
            "optimizator": {
                "SGD": ... # TODO: реализовать стохастический градиентный спуск и ADAM! 
            }
        }


    def compile(self) -> None:
        ...


    def fit(self) -> None:
        ...


    def predict(self) -> np.ndarray:
        ...


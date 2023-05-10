from mylib.lab5.layers.layer import Layer
import numpy as np
from typing import Dict, Callable
from scipy.special import expit
from mylib.utils.apply import apply


class Dense(Layer):
    def __init__(self, units: int, activation: str, input_shape: np.ndarray = None) -> None:
        super().__init__(input_shape)
        self.output_shape = (units,)

        self.x = None
        self.t: float = None

        self.W: np.ndarray = None
        self.b: np.ndarray = None

        self.activation: str = activation

        self.activation_functions: Dict[str, Callable] = {
            "relu": {
                "function": lambda t: np.maximum(0, t),
                "derivative": lambda t: (t >= 0) * 1
            },
            "sigmoid": {
                "function": lambda t: expit(t),
                "derivative": lambda t: expit(t) * (1 - expit(t))
            },
            "tanh": {
                "function": lambda t: np.tanh(t),
                "derivative": lambda t: 1 - (np.tanh(t)) ** 2
            },
            "linear": {
                "function": lambda t: t,
                "derivative": lambda t: np.ones(t.shape)
            },
            "softmax": {
                "function": lambda t: apply(t, lambda x: np.exp(x) / np.sum(np.exp(x))),
            }
        }

    def compile(self, previous_layer: Layer) -> None:
        self.input_shape = previous_layer.output_shape

    def initialize_weights(self):
        self.W = np.random.randn(self.output_shape[0], self.input_shape[0])
        self.b = np.zeros((1, self.output_shape[0]))

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        t = X @ self.W.T + self.b
        h = self.activation_functions[self.activation]["function"](t)

        self.t = np.mean(t, axis=0)
        self.x = np.mean(X, axis=0)

        return h

    def backward_propagation(self) -> tuple:
        print(1)

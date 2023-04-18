import numpy as np
from mylib.neural_network.apply import apply


class Dense:
    
    def __init__(self, units: int, activation: str, input_shape: int = None):
        """
        m - количество нейронов на выходе из слоя
        n - количество нейронов на вход в слой

        W - матрица весов размерности m x n
        b - вектор-столбец смещений размерности m x 1

        x - входные значения
        t - линейная комбинация весов и входных значений + смещение
        f - функция активации
        h = f(t)

        dE_dh - исходное данное значение вектора-градиента по выходу слоя (вычисляется на основе предыдущего)
        dE_dt - градиент по вычисляемым значениям
        dE_dx - градиент по входу слоя (для протаскивания вперед)

        dE_dW - градиент для весов
        dE_db - градиент для смещения
        """

        self.m: int = units # 100%
        self.n: int = input_shape # 100%

        self.W: np.ndarray = None # 100%
        self.b: np.ndarray = None # 100%

        self.x = None # 100%
        self.t: float = None # 100%
        self.activation: str = activation # 100%
        self.h: float = None #! ???

        self.dE_dh: np.ndarray = None #! ???
        self.dE_dt: np.ndarray = None

        self.dE_dx: np.ndarray = None #! ???
        self.dE_dW: np.ndarray = None #! ???
        self.dE_db: np.ndarray = None #! ???

        self.activation_functions = {
            "ReLU": {
                "function": lambda t: apply(t, lambda x: max(0, x)),
                "derivative": lambda t: apply(t, lambda x: 1 if x >= 0 else 0)
            },
            "sigmoid": {
                "function": lambda t: 1 / (1 + np.exp(-t)),
                "derivative": lambda t: np.exp(t) / ((np.exp(t) + 1) ** 2)
            },
            "tanh": {
                "function": lambda t: (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t)),
                "derivative": lambda t: 4 / ((np.exp(t) + np.exp(-t)) ** 2)
            },
            "linear": {
                "function": lambda t: np.exp(t) / np.sum(np.exp(t)),
                "derivative": 1
            },
            "softmax": {
                "function": lambda t: (1 / np.sum(np.exp(t))) * np.exp(t),
                "derivative": lambda t, y: t - y
            }
        }


    def backward_propagation(self, dE_dh: np.ndarray):
        ...



    def forward_propagation(self, X: np.ndarray) -> np.ndarray:

        scalar_prod = X @ self.W.T

        h = apply(scalar_prod, lambda t: (
            self.activation_functions[self.activation]["function"](t + self.b)
        ))

        self.t = np.mean(scalar_prod, axis=0)
        self.x = np.mean(X, axis=0)

        return h


    def initialize_weights(self):
        self.W = np.random.sample((self.m, self.n))
        self.b = np.random.sample(self.m)

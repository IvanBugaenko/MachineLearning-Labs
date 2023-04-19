import numpy as np
from mylib.neural_network.apply import apply
from scipy.special import expm1
from scipy.special import expit



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


        self.activation_functions = {
            "ReLU": {
                "function": lambda t: apply(t, lambda x: max(0, x)),
                "derivative": lambda t: apply(t, lambda x: 1 if x >= 0 else 0)
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
                "derivative": lambda t: 1
            },
            "softmax": {
                "function": lambda t: apply(t, lambda x: (1 / np.sum(np.around(expm1(x) + 1, 5))) * np.around(expm1(x) + 1, 5)),
                "derivative": lambda t: t
            }
        }


    def backward_propagation(self, dE_dh: np.ndarray) -> tuple:
        dE_dt = dE_dh * self.activation_functions[self.activation]["derivative"](self.t)
        dE_dW = np.outer(dE_dt, self.x)
        dE_db = dE_dt
        dE_dx = dE_dt @ self.W
        return dE_dW, dE_db, dE_dx


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

import numpy as np


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

        dE_dh - исходное данное значение вектора-градиента по выходу слоя (вычисляется на основе предыдущего), размерность 
        dE_dt - градиент по вычисляемым значениям
        dE_dx - градиент по входу слоя (для протаскивания вперед)

        dE_dW - градиент для весов
        dE_db - градиент для смещения
        """

        self.m: int = units # 100%
        self.n: int = input_shape # 100%

        self.W: np.ndarray = None # 100%
        self.b: np.ndarray = None # 100%

        self.x = None
        self.t: float = None
        self.activation: str = activation # 100%
        self.h: float = None

        self.dE_dh: np.ndarray = None
        self.dE_dt: np.ndarray = None
        self.dE_dx: np.ndarray = None

        self.dE_dW: np.ndarray = None
        self.dE_db: np.ndarray = None

        self.activation_functions = {
            "ReLU": {
                "function": lambda t: max(0, t),
                "derivative": lambda t: 1 if t >= 0 else 0
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
                "function": lambda t: t ,
                "derivative": 1
            },
            "softmax": {
                # TODO: Softmax
            }
        }


    def backward_propagation(self):
        ...


    # TODO: Переделать для softmax декватного перемножения + запись промежуточных значений
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.t = self.W @ x
        return np.array(
            list(map(
                self.activation_functions[self.activation]["function"], self.W @ x.T
            ))
        ) + self.b


    def initialize_weights(self):
        self.W = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)

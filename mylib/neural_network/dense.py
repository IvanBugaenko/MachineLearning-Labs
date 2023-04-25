import numpy as np
from mylib.neural_network.apply import apply
from scipy.special import expm1
from scipy.special import exp2, expit



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

        self.m: int = units
        self.n: int = input_shape 

        self.W: np.ndarray = None 
        self.b: np.ndarray = None 

        self.x = None 
        self.t: float = None 
        self.activation: str = activation

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
                "function": lambda t: apply(t, lambda x: exp2(x) / np.sum(exp2(t))),
                "derivative": None
            }
        }


    def backward_propagation(self, dE_dh: np.ndarray) -> tuple:
        dE_dt = np.around(np.around(dE_dh, 5) * np.around(self.activation_functions[self.activation]["derivative"](self.t), 5), 5)
        dE_dW = np.around(np.outer(dE_dt, self.x), 5)
        dE_db = dE_dt
        dE_dx = np.around(np.around(dE_dt, 5) @ np.around(self.W, 5), 5)
        print(dE_dx)
        return dE_dW, dE_db, dE_dx


    def forward_propagation(self, X: np.ndarray) -> np.ndarray:

        scalar_prod = np.around(np.around(X, 5) @ np.around(self.W.T, 5), 5)

        t = np.around(apply(scalar_prod, lambda x: x + self.b), 5)

        h = np.around(apply(t, self.activation_functions[self.activation]["function"]), 5)

        self.t = np.mean(t, axis=0)
        self.x = np.mean(X, axis=0)

        return h


    def initialize_weights(self):
        self.W = np.around(np.random.sample((self.m, self.n)), 5)
        self.b = np.around(np.random.sample(self.m), 5)

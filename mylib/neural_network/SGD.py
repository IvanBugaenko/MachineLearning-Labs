import numpy as np
from typing import List, Dict, Callable
from mylib.neural_network.dence import Dense


class SGD:
    """
    interface IOptimizer:
        optimize(); // возвращает готовенький слой
        compile(); // инициализирует начальное состояние оптимизатора

        init(); // инициализация специфичных параметров оптимизатора
    """
    
    def __init__(self, learning_rate: int = 0.01, batch_size: int = 1) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def __forward_propagation(self, X: np.ndarray) -> np.ndarray:
        res = X
        for layer in self.layers:
            res = layer.forward_propagation(res)

        return res
    

    def __backward_propagation(self, initial_dE_dh: np.ndarray) -> None:
        dE_dh = initial_dE_dh
            
        for i in range(len(self.layers) - 1, -1, -1):
            dE_dW, dE_db, dE_dx = self.layers[i].backward_propagation(dE_dh)
            self.layers[i].W += self.learning_rate * dE_dW
            self.layers[i].b += self.learning_rate * dE_db
            dE_dh = dE_dx


    def compile(self, layers: List[Dense], loss_function: Dict[str, Callable]) -> None:
        self.layers = layers
        self.loss_function = loss_function
        
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> List[Dense]:
        chi: np.array = np.c_[X_train, y_train]
        n = len(np.unique(y_train))

        np.random.shuffle(chi)
        
        for i in range(len(chi) // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            X = chi[start:end, :-1]

            y = chi[start:end, -1].reshape(end - start, 1)

            y = self.loss_function["transform"](y, n)
            h = self.__forward_propagation(X)      # предсказание НС для всего батча: |batch| x m, m - количество нейронов на выходе

            initial_dE_dh = np.mean(self.loss_function["derivative"](y, h), axis=0)

            self.__backward_propagation(initial_dE_dh)

            return self.layers

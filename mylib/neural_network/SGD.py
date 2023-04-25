import numpy as np
from typing import List, Dict, Callable
from mylib.neural_network.dense import Dense


class SGD:
    
    def __init__(self, learning_rate: int = 0.01, batch_size: int = 1) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def __forward_propagation(self, X: np.ndarray, layers: List[Dense]) -> np.ndarray:
        res = X
        for layer in layers:
            res = np.around(layer.forward_propagation(res), 5)

        return res
    

    def __update(self, initial_dE_dh: np.ndarray, layers: List[Dense]) -> None:
        dE_dh = np.around(initial_dE_dh, 5)
        temp_layers = layers
            
        for i in range(len(temp_layers) - 1, -1, -1):
            if i == len(temp_layers) - 1:
                dE_dW = np.around(np.outer(np.around(dE_dh, 5), np.around(temp_layers[i].x, 5)), 5)
                dE_db = dE_dh
                dE_dx = np.around(dE_dh @ temp_layers[i].W, 5)
                temp_layers[i].W -= np.around(self.learning_rate * dE_dW, 5)
                temp_layers[i].b -= np.around(self.learning_rate * dE_db, 5)
                dE_dh = dE_dx
                continue

            dE_dW, dE_db, dE_dx = temp_layers[i].backward_propagation(dE_dh)
            temp_layers[i].W -= np.around(self.learning_rate * dE_dW, 5)
            temp_layers[i].b -= np.around(self.learning_rate * dE_db, 5)
            dE_dh = dE_dx
        
        return temp_layers


    def compile(self, loss_function: Dict[str, Callable]) -> None:
        self.loss_function = loss_function
        
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, layers: List[Dense]) -> List[Dense]:
        temp_layers = layers
        chi: np.array = np.c_[X_train, y_train]
        n = len(np.unique(y_train))

        np.random.shuffle(chi)
        
        for i in range(len(chi) // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            X = chi[start:end, :-1]

            y = chi[start:end, -1].reshape(end - start, 1)

            y = self.loss_function["transform"](y, n)
            h = self.__forward_propagation(X, temp_layers)

            initial_dE_dt = np.around(np.mean(self.loss_function["derivative"](y, h), axis=0), 5)

            temp_layers = self.__update(initial_dE_dt, temp_layers)

        return temp_layers

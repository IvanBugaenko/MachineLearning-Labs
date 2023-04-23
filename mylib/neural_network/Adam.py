import numpy as np
from typing import List, Dict, Callable
from mylib.neural_network.dense import Dense


class Adam:

    def __init__(self, learning_rate: int = 0.01, batch_size: int = 1) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def compile(self, layers: List[Dense], loss_function: Dict[str, Callable]) -> None:
        self.layers = layers
        self.loss_function = loss_function


    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> List[Dense]:
        
        
        
        return self.layers

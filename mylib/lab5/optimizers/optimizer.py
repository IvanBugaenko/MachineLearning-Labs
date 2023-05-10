import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod


class Optimizer:
    def __init__(self, learning_rate: int = 0.01, batch_size: int = 1):
        self.learning_rate: int = learning_rate
        self.batch_size: int = batch_size

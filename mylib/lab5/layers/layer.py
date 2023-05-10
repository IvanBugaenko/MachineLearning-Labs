import numpy as np
from typing import Any
# from abc import ABC, abstractmethod


# class Layer(ABC):
class Layer:

    def __init__(self, input_shape: tuple = None) -> None:
        self.input_shape: tuple = input_shape
        self.output_shape: tuple = None

    # @abstractmethod
    def compile(self) -> Any:
        pass

    # @abstractmethod
    def initialize_weights(self) -> Any:
        pass

    # @abstractmethod
    def forward_propagation(self) -> Any:
        pass

    # @abstractmethod
    def backward_propagation(self) -> Any:
        pass

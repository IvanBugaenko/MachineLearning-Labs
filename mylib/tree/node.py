import numpy as np


class Node:

    def __init__(self) -> None:
        self.left_node = None
        self.right_node = None
        self.score = None
        self.value = None
        self.t = None
        self.index = None

    def __call__(self):
        print(1, end=' ')

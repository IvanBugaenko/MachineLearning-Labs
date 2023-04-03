import numpy as np
from mylib.tree.node import Node
from mylib.tree.build_tree import build_tree


class MyDecisionTree:

    def __init__(self, tree_type: str, max_depth: int = 3) -> None:
        self.tree_type: str = tree_type
        self.max_depth: int = max_depth


    def fit(self, X: np.array, y: np.array) -> None:
        self.tree: Node = build_tree(tree_type, np.c[X, y], self.max_depth)

        return self


    def __tree_answer(self, x: np.array) -> object:
        ...


    def predict(self, X: np.array) -> np.array:
        answer = []
        for obj in X:
            answer.append(self.__tree_answer(obj))

        return answer

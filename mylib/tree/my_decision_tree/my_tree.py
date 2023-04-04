import numpy as np
from mylib.tree.my_decision_tree.node import Node
from mylib.tree.my_decision_tree.build_tree import build_tree
from mylib.tree.my_decision_tree.popusk import popusk
from mylib.tree.my_decision_tree.dependensies import functions


class MyDecisionTree:

    def __init__(self, tree_type: str, max_depth: int = 3) -> None:
        self.tree_type: str = tree_type
        self.max_depth: int = max_depth


    def fit(self, X: np.array, y: np.array) -> None:
        self.tree: Node = build_tree(self.tree_type, np.c_[X, y], self.max_depth)

        return self


    def __tree_answer(self, x: np.array) -> object:
        return functions[self.tree_type]["predict"](popusk(self.tree, x))


    def predict(self, X: np.array) -> np.array:
        answer = []
        for obj in X:
            answer.append(self.__tree_answer(obj))

        return np.array(answer)

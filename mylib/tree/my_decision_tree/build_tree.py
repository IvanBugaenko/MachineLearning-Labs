import numpy as np
from mylib.tree.my_decision_tree.node import Node
from mylib.tree.my_decision_tree.dependensies import functions
from mylib.tree.my_decision_tree.best_split import best_split


def build_tree(tree_type: str, chi: np.array, max_depth: int, depth=0) -> Node:
    
    node = Node()

    node.value = functions[tree_type]["value"](chi)

    if depth >= max_depth or functions[tree_type]["score"] == 0 or len(chi) == 1:
        return node

    index, predicate_value, left_dataset, right_dataset = best_split(chi, tree_type)

    node.predicate_value = predicate_value
    node.index = index

    node.left_node = build_tree(tree_type, left_dataset.copy(), max_depth, depth + 1)
    node.right_node = build_tree(tree_type, right_dataset.copy(), max_depth, depth + 1)

    return node

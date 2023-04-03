import numpy as np
from mylib.tree.node import Node
from mylib.tree.dependensies import functions


def build_tree(tree_type: str, chi: np.array, max_depth: int, depth=0) -> Node:
    
    node = Node()

    if depth < max_depth:
        score, index, t, left_dataset, right_dataset = best_split(chi)
        
        node.value = functions[tree_type]["value"](chi)
        node.t = t
        node.score = score
        node.index = index
        node.left_node = build_tree(tree_type, left_dataset, max_depth, depth + 1)
        node.right_node = build_tree(tree_type, right_dataset, max_depth, depth + 1)

    return node

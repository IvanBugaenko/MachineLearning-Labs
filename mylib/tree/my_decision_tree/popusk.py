import numpy as np
from mylib.tree.my_decision_tree.node import Node


def popusk(node: Node, x: np.array) -> object:
    
    if(node.left_node and node.right_node):
            if(x[node.index] < node.predicate_value):
                return popusk(node.left_node, x)
            return popusk(node.right_node, x)
    
    return node.value

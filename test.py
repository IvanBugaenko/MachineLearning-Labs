import numpy as np
from mylib.tree.my_decision_tree.node import Node
from scipy.stats import mode



a = np.array([[1, 2, 3, 1],
              [4, 5, 6, 1],
              [7, 8, 9, 0]])

# print(a[np.random.choice(np.arange(len(a)), 2)])

a = np.array([1, 2, 3, 1])

# print(np.array(list(map(lambda x: x + 1, a))))

a = np.zeros((2, 3))
print(a)
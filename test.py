import numpy as np
from mylib.tree.my_decision_tree.node import Node
from scipy.stats import mode
import random


a = np.array([[1, 2, 3, 1],
              [4, 5, 6, 1],
              [7, 8, 9, 0]])

# print(a[np.random.choice(np.arange(len(a)), 2)])

# a = np.array([1, 2, 3, 1])
# b = np.array([4, 5, 6, 1])

# print(np.array(list(map(lambda x: x + 1, a))))


f = lambda x, y: x + y

np.random.shuffle(a)

# print(a @ b.T)

# print(np.exp(2))

# f = lambda x: 

# print(f(4))

# a = np.array([1, 2, 3])
# b = np.array([4, 5])
# print(np.zeros((1, 4)))
# print(b.transpose())
print(a)


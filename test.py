import numpy as np
import random
from mylib.neural_network.apply import apply


relu = lambda t: apply(t, lambda x: max(0, x))
softmax = lambda t: (1 / np.sum(np.exp(t))) * np.exp(t)



h0 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
W1 = np.array([[2, -3, 4],
               [-5, 1, 1]])
b1 = np.array([3, 4])
W2 = np.array([[1, 1],
               [-1, -2],
               [3, 7],
               [-6, 1]])
b2 = np.array([1, -1, 1, -1])


t1 = h0 @ W1.T
h1 = apply(t1, lambda t: (
            relu(t + b1)
        ))

t2 = h1 @ W2.T
h2 = apply(t2, lambda t: (
            softmax(t + b2)
        ))

print(h2)

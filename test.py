import numpy as np
import random
from mylib.neural_network.apply import apply
from scipy.special import expit
from scipy.linalg import tanhm


relu = lambda t: apply(t, lambda x: max(0, x))
softmax = lambda t: (1 / np.sum(np.exp(t))) * np.exp(t)

y = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0]
])

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

l = lambda y_true, y_pred: -y_true / y_pred

dE_dh2 = np.mean(l(y, h2), axis=0)

f2 = lambda t: t

dE_dt2 = dE_dh2 * f2(np.mean(t2, axis=0))

dE_dW2 = np.outer(dE_dt2, np.mean(h1, axis=0))

dE_dx2 = dE_dt2 * np.mean(W2)
# print(dE_dx2)


a = np.array([1, 2])
b = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7]
])

print(tanhm(b))
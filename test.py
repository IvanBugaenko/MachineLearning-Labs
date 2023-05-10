import numpy as np
import random
from mylib.neural_network.apply import apply
from scipy.special import expit, exp2
from scipy.linalg import tanhm


def relu(t): return apply(t, lambda x: max(0, x))
def softmax(t): return (1 / np.sum(np.exp(t))) * np.exp(t)


y = np.array([
    [
        [
            [1, 2, 6, 5],
            [3, 4, 6, 5]
        ],
        [
            [5, 6, 6, 6],
            [7, 8, 6, 7]
        ],
        [
            [9, 1, 6, 3],
            [5, 6, 7, 1]
        ]
    ],

])
# print(y.shape)
# print(np.pad(y, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0))
# print(np.zeros((4,)))
# print(type((4,)))
# print(list(map(lambda x: x + 1, [3])))

print(y.reshape(-1, 1))


def f(t): return apply(t, lambda x: np.exp(x) / np.sum(np.exp(x)))


a = np.array(
    [
        [1, 2, 3],
        [0, 0, 1]
    ]
)

b = np.array([[1, 1, 1, 1]])
print(np.sum(f(a), axis=1))

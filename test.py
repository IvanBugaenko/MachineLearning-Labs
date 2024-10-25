import numpy as np
import random
from mylib.neural_network.apply import apply
from scipy.special import expit, exp2
from scipy.linalg import tanhm


# def relu(t): return apply(t, lambda x: max(0, x))
# def softmax(t): return (1 / np.sum(np.exp(t))) * np.exp(t)


# y = np.array([
#     [
#         [
#             [1, 2, 6, 5],
#             [3, 4, 6, 5]
#         ],
#         [
#             [5, 6, 6, 6],
#             [7, 8, 6, 7]
#         ],
#         [
#             [9, 1, 6, 3],
#             [5, 6, 7, 1]
#         ]
#     ],

# ])



# def f(t): return apply(t, lambda x: np.exp(x) / np.sum(np.exp(x)))


a = np.array(
    [
        [1, 2, 4],
        [0, 0, 1],
        [1, 1, 1],
        [3, 5, 7],
        [1, 2, 3],
        [9, 0, 4],
        [4, 5, 6],
        [1, 2, 8]
    ]
)

b = np.array([8, 8, 8])

# print(np.cov(a).shape)
val, vec = np.linalg.eig(np.cov(a.T))
print(sorted(zip(val, vec)))
# print(np.c_[[v.T for v in a]])

print(bool(None))

n = 2

cov = np.cov(a.T)
eig_values, eig_vectors = np.linalg.eig(cov)
W = np.c_[[eig_vector[1] for eig_vector in sorted(zip(eig_values, eig_vectors), reverse=True)[:n]]]
print(a @ W.T)

# centroids = np.array(
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ]
# )

# b = np.array([1, 1, 0, 0, 0, 1, 0, 1])

# chi = np.c_[a, b]

# print(chi[chi[:, -1] == 1])

# print(np.mean(chi[chi[:, -1] == 1], axis=0))

# for cluster in range(2):
#     print(np.mean(chi[chi[:, -1] == cluster], axis=0)[:-1])
#     # centroids[cluster] = np.mean([chi[:, -1] == cluster], axis=0)[:, :-1]

# # print(np.all(f(c, c1) < 11))

# # print(np.mean(a[a[:, -1] == 4], axis=0))




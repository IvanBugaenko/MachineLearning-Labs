import numpy as np
import pandas as pd

# a = np.array([
#     [1, 0, 1, 1],
#     [7, 2, 3, 4],
#     [14, 2, 3, 2]
# ])

# l = lambda x: {
#             print(x),
#             print(x + 1)
#         }

# b = np.array(a)

# def statistics(classes: np.array, chi: np.array) -> dict:
        
#         stats = {}

#         for c in classes:
#             stats.update({
#                 c:
#                 {
#                     "mean": chi[np.in1d(chi[:, -1], np.array([c]))][:, :-1].mean(axis=0),
#                     "std": chi[np.in1d(chi[:, -1], np.array([c]))][:, :-1].std(axis=0)
#                 }
#             })

#         return stats

# print(a[np.in1d(a[:, -1], np.array([2]))][:, :-1].mean(axis=0))

# print(statistics(np.array([1, 2, 4]), a))






A = pd.DataFrame(data=[1, 2, 1, 1, 2, 1, 2, 2, 1])
print(A)
print()
B = A.groupby([0])[0].count()
print(B.loc[1])
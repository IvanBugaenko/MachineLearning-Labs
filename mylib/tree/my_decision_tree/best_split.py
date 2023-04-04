import numpy as np
from mylib.tree.my_decision_tree.split_chi import split_chi
from mylib.tree.my_decision_tree.dependensies import functions


def best_split(chi: np.array, tree_type: str) -> tuple:
    best_score: float = float('inf')
    best_index: int = 0
    best_predicate_value: float = 0
    best_left_chi: np.array = np.array([])
    best_right_chi: np.array = np.array([])

    X = chi[:,:-1]

    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            predicate_value = float(X[i][j])

            left_chi, right_chi = split_chi(chi, j, predicate_value)

            score_function = functions[tree_type]["score"]

            N_left: int = len(left_chi)
            N_right: int = len(right_chi)
            N =  N_left + N_right

            if N_left == 0 or N_right == 0:
                continue

            score: float = float(
                score_function(left_chi) * (len(left_chi) / N) + 
                score_function(right_chi) * (len(right_chi) / N)
            )

            if score < best_score:
                best_score = score
                best_index = j
                best_predicate_value = predicate_value
                best_left_chi = left_chi.copy()
                best_right_chi = right_chi.copy()

    return best_index, best_predicate_value, best_left_chi.copy(), best_right_chi.copy()

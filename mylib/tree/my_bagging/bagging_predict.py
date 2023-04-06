import numpy as np
from mylib.tree.my_bagging.dependencies import functions


def bagging_predict(algorithm: list, x: np.array, bagging_type: str) -> object:

    vote = []
    
    for est in algorithm:
        data = np.array([x])
        pred = est.predict(data)
        vote.append(pred[0])

    return functions[bagging_type]["predict"](np.array(vote))

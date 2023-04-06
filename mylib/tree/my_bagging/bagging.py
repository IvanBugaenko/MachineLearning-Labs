import numpy as np
from mylib.tree.my_bagging.create_bagging_algorithm import create_bagging_algorithm
from mylib.tree.my_bagging.bagging_predict import bagging_predict


class MyBagging():

    def __init__(self, estimator, bagging_type: str, n_estimators: int = 10) -> None:
        
        self.estimator = estimator
        self.bagging_type = bagging_type
        self.n_estimators = n_estimators


    def fit(self, X: np.array, y: np.array):

        self.n = len(X)

        self.algorithm: list = create_bagging_algorithm(self.estimator, self.n_estimators, X, y)

        return self


    def __bagging_vote(self, x: np.array) -> object:
        return bagging_predict(self.algorithm, x, self.bagging_type)
    

    def predict(self, X: np.array) -> np.array:

        answer = []
        for x in X:
            answer.append(self.__bagging_vote(x))

        return np.array(answer)

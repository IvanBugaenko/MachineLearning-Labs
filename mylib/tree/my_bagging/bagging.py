import numpy as np


class Bagging():

    def __init__(self, estimator, n_estimators: int = 10) -> None:
        
        self.estimator = estimator
        self.n_estimators = n_estimators


    def fit(self, X: np.array, y: np.array):

        self.n = len(X)

        self.alghorithm = create_bagging_alghorithm(self.estimator, self.n_estimators, X)

        return self
    

    def __bagging_vote(self, x: np.array) -> object:
        
    

    def predict(self, X: np.array) -> np.array:

        answer = []
        for x in X:
            answer.append(self.__bagging_vote(x))

        return np.array(answer)

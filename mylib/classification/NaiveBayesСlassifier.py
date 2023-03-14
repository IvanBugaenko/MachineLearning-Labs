import numpy as np
from math import pi, sqrt


class MyNaiveBayesClassifer:

    X_train: np.array
    y_train: np.array
    classes: np.array 
    priori: dict = {}
    statistics: dict = {}


    def fit(self, X: np.array, y: np.array):

        self.X_train = X
        self.y_train = y

        self.classes = np.unique(y)

        self.priori = self.classes_prior_probability(y)

        chi = np.c_[X, y]

        self.statistics = self.statistics(self.classes, chi)

        return self


    @staticmethod
    def classes_prior_probability(y: np.array) -> dict:

        info = np.unique(y, return_counts=True)

        classes_and_counts = []

        for i in range(len(info[0])):
            classes_and_counts.append((info[0][i], info[1][i]))

        n = y.shape[0]

        priori = {}

        for cl, co in classes_and_counts:
            priori.update({cl: co / n})

        return priori
    

    @staticmethod
    def statistics(classes: np.array, chi: np.array) -> dict:
        
        stats = {}

        for c in classes:
            stats.update({
                c:
                {
                    "mean": chi[np.in1d(chi[:, -1], np.array([c]))][:, :-1].mean(axis=0),
                    "std": chi[np.in1d(chi[:, -1], np.array([c]))][:, :-1].std(axis=0)
                }
            })

        return stats


    def classificator(self, x: np.array) -> object:

        choise = []

        for c in self.classes:
            p = np.prod(
                (1/sqrt(2 * pi)) *
                  self.statistics[c]["std"] *
                    np.exp(
                      -((x - self.statistics[c]["mean"]) ** 2) /  
                        self.statistics[c]["std"] ** 2
                    )
            ) * self.priori[c]
            
            choise.append((p, c))

        return sorted(choise, reverse=True)[0][-1] 


    def predict(self, X: np.array) -> np.array:
        return np.array([self.classificator(x) for x in X])

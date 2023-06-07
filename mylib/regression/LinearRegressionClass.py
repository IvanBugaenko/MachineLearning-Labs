import numpy as np
from scipy.optimize import minimize


class MyLinearRegression_L2:
    def __init__(self, alpha: float, eps: float) -> None:
        self.alpha = alpha
        self.eps = eps


    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        self.y_train = np.array(y_train)
        self.l = X_train.shape[0]
        self.d = X_train.shape[1] + 1
        self.coef = np.zeros(self.d)
        self.optimizator()

        return self


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.dot(np.c_[np.ones(X_test.shape[0]), X_test], self.coef)
    

    def optimizator(self, k=0.00001) -> None:
        w0 = self.coef
        t = 1
        w1 = w0 - (k / t) * self.nabla(w0)
        while t <= 100:
            t += 1
            w0, w1 = w1, w1 - (k / t) * self.nabla(w1)
        self.coef = w1
        # self.coef = minimize(self.loss_function, self.coef, method='SLSQP').x


    def nabla(self, w) -> np.ndarray:
        grad = []
        for i in range(self.d):
            a = 0
            for j in range(self.l):
                a += ((self.y_train[j] - np.dot(w, self.X_train[j]))* self.X_train[j][i])
            grad.append((-2 / self.l) * (a - self.alpha * w[i]))
        return np.array(grad)

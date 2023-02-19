import numpy as np
from scipy import optimize as opt


class My_LinearRegression:
    coef: np.array
    m: int
    X_train: np.array
    y_train: np.array

    def __init__(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.m = X_train.shape[1]
        self.coef = np.zeros(self.m)

    def fit(self):
        '''
        Функция, подбирающая значения коэффициентов
        '''
        self.coef = opt.least_squares(self.loss_function, self.coef).x
        return self

    def predict(self, X_test):
        '''
        Функция по вычислению регрессии (выдача предсказания, исходя из коэффициентов)
        '''
        return np.matmul(X_test, self.coef)
    
    def loss_function(self, c: np.array):
        '''
        Функция потерь
        '''
        return np.sum((np.array(self.y_train) - np.matmul(self.X_train, c)) ** 2)
    
import numpy as np
from math import sqrt


def MAE(y_test, y_predict):
    return np.sum(abs(np.array(y_test) - np.array(y_predict))) / len(y_test)


def MSE(y_test, y_predict):
    return np.sum((np.array(y_test) - np.array(y_predict)) ** 2) / len(y_test)


def RMSE(y_test, y_predict):
    return sqrt(MSE(y_test, y_predict))


def MAPE(y_test, y_predict):
    return np.mean(abs( (np.array(y_test) - np.array(y_predict)) / np.array(y_test) ))


def R_2(y_test, y_predict):
    return 1 - ( MSE(y_test, y_predict) / ( np.sum((np.array(y_test) - np.array(y_test).mean()) ** 2) / len(y_test) ) )


def all_metrics(y_test, y_predict):
    print(f'MAE: {MAE(y_test, y_predict)}')
    print(f'MSE: {MSE(y_test, y_predict)}')
    print(f'RMSE: {RMSE(y_test, y_predict)}')
    print(f'MAPE: {MAPE(y_test, y_predict)}')
    print(f'R^2: {R_2(y_test, y_predict)}')
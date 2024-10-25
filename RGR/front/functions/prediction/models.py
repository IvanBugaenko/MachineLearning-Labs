import joblib


pipe_knn = joblib.load('..\preprocessing\pipe_knn.pkl')
pipe_bagging = joblib.load('..\preprocessing\pipe_bagging.pkl')
pipe_dence = joblib.load('..\preprocessing\pipe_dence.h5')


models = {
    "KNN": pipe_knn,
    "Bagging": pipe_bagging,
    "Нейронная сеть": pipe_dence
}

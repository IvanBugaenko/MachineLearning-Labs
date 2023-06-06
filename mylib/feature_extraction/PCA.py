import numpy as np


class MyPCA:
    def __init__(self, n_components, is_scaled=False) -> None:
        self.n_components = n_components
        self.is_scaled = is_scaled


    def fit(self, X: np.ndarray) -> None:
        cov = np.cov(X.T) if self.is_scaled else np.cov(((X - np.mean(X, axis=0))/np.std(X, axis=0)).T)
        eig_values, eig_vectors = np.linalg.eig(cov)
        self.W = np.c_[[eig_vector[1] for eig_vector in sorted(zip(eig_values, eig_vectors), reverse=True)[:self.n_components]]]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X if self.is_scaled else (X - np.mean(X, axis=0))/np.std(X, axis=0)
        return Z @ self.W.T


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        _ = self.fit(X)
        return X @ self.W.T

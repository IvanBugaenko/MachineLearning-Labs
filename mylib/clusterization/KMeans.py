import numpy as np


class KMeans:
    def __init__(self, k: int, eps: float, max_iter=300, metrics: str = "euclid") -> None:
        self.dependencies = {
            "euclid": lambda X, centroid: np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        }

        self.k = k
        self.eps = eps
        self.max_iter = max_iter
        self.metrics = metrics

    def fit(self, X: np.ndarray) -> object:
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        while i := 1 <= self.max_iter:
            d = []
            for centroid in centroids:
                distances = self.dependencies[self.metrics](X, centroid)
                d.append(distances)

            labels = np.argmin(np.vstack(d).T, axis=1)

            chi = np.c_[X, labels]

            new_centroids = np.copy(centroids)

            for cluster in range(self.k):
                centroids[cluster] = np.mean(chi[chi[:, -1] == cluster], axis=0)[:-1]

            if np.all(self.dependencies[self.metrics](new_centroids, centroids) < self.eps):
                break

            i += 1

        self.labels_ = labels

        return self
        
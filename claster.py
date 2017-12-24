import numpy as np
from scipy.stats import mode


class KMeans(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = [[] for _ in range(n_clusters)]
        self.centers = []

    def fit(self, x):
        self.centers = np.random.random((self.n_clusters, len(x.T)))
        centers_old = self.centers + 1  # just to be sure they are different

        while not np.array_equal(centers_old, self.centers):
            centers_old = self.centers
            self.clusters = [[] for _ in range(self.n_clusters)]
            for point in range(len(x)):
                # find distances to all cluster centers
                distances = np.linalg.norm(self.centers - x[point, :], axis=1)
                # add point to neartest cluster
                self.clusters[np.argmin(distances)].append(point)

            self.centers = []
            for cluster in self.clusters:
                center = []
                sub_x = x[cluster]
                for feature in range(len(sub_x.T)):
                    center.append(np.mean(sub_x[:, feature]))

                self.centers.append(center)

    def predict(self, x):
        x = np.array(x)
        result = []
        for point in range(len(x)):
            distances = np.linalg.norm(self.centers - x[point, :], axis=1)
            result.append(np.argmin(distances))

        return result


class KNNClf(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def _get_neighbors(self, point):
        distances = np.linalg.norm(self.x - point, axis=1)
        return np.argsort(distances)[:self.n_neighbors]

    def predict(self, x):
        result = []
        for point in x:
            neighbors = self._get_neighbors(point)
            result.append(int(mode(self.y[neighbors])[0]))

        return result


class KNNReg(KNNClf):
    def predict(self, x):
        result = []
        for point in x:
            neighbors = self._get_neighbors(point)
            result.append(np.mean(self.y[neighbors]))

        return result

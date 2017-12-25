import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from cluster import KMeans
from cluster import KNNClf


def test_kmeans_init():
    clf = KMeans(3)
    assert clf.n_clusters == 3
    assert len(clf.clusters) == 3
    assert clf.centers == []


def test_kmeans_fit():
    dataset = datasets.load_iris()
    scaler = StandardScaler()
    x = scaler.fit_transform(dataset.data)

    clf = KMeans(3)
    clf.fit(x)
    assert len(clf.centers) == 3
    assert clf.n_clusters == 3
    assert np.array(clf.centers).shape == (3, 4)
    assert (set(clf.centers[0])
            != set(clf.centers[1])
            != set(clf.centers[2])
            != set(clf.centers[0]))


def test_kmean_predict():
    np.random.seed(42)
    dataset = datasets.load_iris()
    scaler = StandardScaler()
    x = scaler.fit_transform(dataset.data)
    y = dataset.target
    clf = KMeans(3)
    clf.fit(x)
    centers = clf.centers
    y2_pred = clf.predict(x[20:30])
    assert len(x[20:30]) == len(y2_pred)
    y_pred = clf.predict(x)
    assert set(y_pred) == set(y)
    assert clf.n_clusters == 3
    assert np.array_equiv(clf.centers, centers)

    for cluster in range(clf.n_clusters):
        for point_n in np.random.choice(clf.clusters[cluster],
                                        size=min(20, len(clf.clusters[cluster])),
                                        replace=False):
            point = x[point_n]
            dist_internal = np.linalg.norm(clf.centers[cluster] - point)
            dist_external = min(np.linalg.norm(clf.centers[(cluster+1) % 3] - point),
                                np.linalg.norm(clf.centers[(cluster+2) % 3] - point))
            assert dist_internal <= dist_external


def test_knn_clf_init():
    clf = KNNClf(3)
    assert clf.n_neighbors == 3
    assert clf.x is None
    assert clf.y is None


def test_knn_clf_fit():
    dataset = datasets.load_iris()
    scaler = StandardScaler()
    x = scaler.fit_transform(dataset.data)
    y = dataset.target
    clf = KNNClf(3)
    clf.fit(x, y)
    assert np.array_equiv(clf.x, x)
    assert np.array_equiv(clf.y, y)
    assert clf.n_neighbors == 3


def test_knn_clf_get_neighbors():
    np.random.seed(42)
    dataset = datasets.load_iris()
    scaler = StandardScaler()
    x = scaler.fit_transform(dataset.data)
    y = dataset.target
    clf = KNNClf(3)
    clf.fit(x, y)
    points = x[np.random.choice(x.shape[0], 20, replace=False)]
    for point in points:
        neighbors = clf._get_neighbors(point)
        assert len(neighbors) == clf.n_neighbors
        dist_neighbors = max(np.linalg.norm(x[neighbors] - point, axis=1))
        for other in x[np.random.choice(x.shape[0], 20, replace=False)]:
            if other not in x[neighbors]:
                dist_other = np.linalg.norm(other - point)
                assert dist_neighbors <= dist_other

#
# def test_KNN_predict(): clf = KNN(3)
#     clf.fit(x, y)
#     y_pred = clf.predict(x)
#     assert y_pred == clf.y

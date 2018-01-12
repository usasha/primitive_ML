import numpy as np

from trees import TreeNode


class IsolationTree(object):
    def __init__(self):
        self.root = TreeNode()

    def split(self, x, y, node):
        if len(np.unique(y)) == 1 or x is None:
            return
        if len(x) < 1:
            pass

        variance = 1
        while variance < 2:
            feature = np.random.randint(len(x.T))
            variance = len(np.unique(x[:, feature]))
        low_bound = x[:, feature].min() + 1e-10  # np.random.uniform interval is half open
        high_bound = x[:, feature].max() - 1e-10
        threshold = np.random.uniform(low_bound, high_bound)

        left = x[:, feature] < threshold
        right = ~left

        node.feature = feature
        node.threshold = threshold

        node.left = TreeNode()
        self.split(x[left], y[left], node.left)
        node.right = TreeNode()
        self.split(x[right], y[right], node.right)

    def fit(self, x, y):
        self.split(x, y, self.root)

    def find_distance(self, point, current_dist=0, node=None):
        if node is None:
            node = self.root
        if (node.left is None) and (node.right is None):
            return current_dist

        if point[node.feature] < node.threshold:
            return self.find_distance(point, current_dist+1, node.left)
        else:
            return self.find_distance(point, current_dist+1, node.right)


class IsolationForest(object):
    def __init__(self, n_estimators=20, outliers_percent=0.01):
        self.n_estimators = n_estimators
        self.trees = []
        self.outliers_percent = outliers_percent

    def fit(self, x, y):
        for i in range(self.n_estimators):
            sub = np.random.randint(0, len(x), len(x))

            new_tree = IsolationTree()
            new_tree.fit(x[[sub]], y[[sub]])
            self.trees.append(new_tree)

    def predict(self, x):
        outliers_n = int(len(x) * self.outliers_percent)
        distance_means = []
        for point in x:
            point_distances = []
            for tree in self.trees:
                point_distances.append(np.array(tree.find_distance(point)))
            distance_means.append(np.mean(point_distances))
        furthest = np.argsort(distance_means)[::-1][:outliers_n]

        result = np.zeros(len(x))
        result[furthest] = 1

        return result

import numpy as np

from trees import TreeNode


class IsolationTree(object):
    def __init__(self):
        self.root = TreeNode()

    def split(self, x, y, node):
        if len(np.unique(y)) == 1:
            return

        feature = np.random.randint(len(x.T))
        feature_min = x[:, feature].min()
        feature_max = x[:, feature].max()
        threshold = np.random.uniform(feature_min, feature_max)

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

    def find_distance(self, point, current_dist, node):
        if (node.left is None) and (node.right is None):
            return current_dist

        if point[node.feature] < node.threshold:
            return self.find_distance(point, current_dist+1, node.left)
        else:
            return self.find_distance(point, current_dist+1, node.right)


class IsolationForest(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x, y):
        pass

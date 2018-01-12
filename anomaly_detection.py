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
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x, y):
        pass

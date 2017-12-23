import numpy as np


class TreeNode(object):
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.predict = None
        self.left = None
        self.right = None


class DecisionTree(object):
    def __init__(self):
        self.root = TreeNode(None, None)
        pass

    def _inf_criteria(self, y):
        if len(y):
            return np.dot((y - y.mean()), (y - y.mean())) / len(y)
        else:
            return 0

    def split(self, x, y):
        errors = {}
        for feature in range(len(x) + 1):
            for threshold in np.unique(x[:, feature]):
                left = x[:, feature] < threshold
                inf_left = self._inf_criteria(y[left])
                inf_right = self._inf_criteria(y[~left])
                err = (len(y[left]) / len(y) * inf_left
                       + len(y[~left]) / len(y) * inf_right)
                errors[err] = {'feature': feature, 'threshold': threshold}

        best_split = min(errors.keys())
        return best_split

    def fit(self, x, y):
        self.root = TreeNode(1, 0.5)
        self.root.left = TreeNode(None, None)
        self.root.left.predict = 111
        self.root.right = TreeNode(None, None)
        self.root.right.predict = 777

    def predict(self, x):
        result = []
        for point in x:
            print('>', point)
            tree = self.root
            while not tree.predict:
                if point[tree.feature] < tree.threshold:
                    tree = tree.left
                else:
                    tree = tree.right
            result.append(tree.predict)
        return result

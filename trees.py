import numpy as np


class TreeNode(object):
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.predict = None
        self.left = None
        self.right = None


class DecisionTree(object):
    def __init__(self):
        self.root = TreeNode()
        pass

    def _inf_criteria(self, y):
        if len(y):
            return np.dot((y - y.mean()), (y - y.mean())) / len(y)
        else:
            return 0

    def split(self, x, y, node):
        if len(np.unique(y)) == 1:
            node.predict = np.mean(y)
            return

        errors = {}
        for feature in range(len(x.T)):
            for threshold in np.unique(x[:, feature]):
                left = x[:, feature] < threshold
                inf_left = self._inf_criteria(y[left])
                inf_right = self._inf_criteria(y[~left])
                err = (len(y[left]) / len(y) * inf_left
                       + len(y[~left]) / len(y) * inf_right)
                errors[err] = {'feature': feature, 'threshold': threshold}

        best_split = errors[min(errors.keys())]
        left = x[:, best_split['feature']] < best_split['threshold']
        right = ~left

        node.feature = best_split['feature']
        node.threshold = best_split['threshold']
        node.left = TreeNode()
        self.split(x[left], y[left], node.left)
        node.right = TreeNode()
        self.split(x[right], y[right], node.right)

    def fit(self, x, y):
        self.split(x, y, self.root)

    def predict(self, x):
        result = []
        for point in x:
            tree = self.root
            while not tree.predict:
                if point[tree.feature] < tree.threshold:
                    tree = tree.left
                else:
                    tree = tree.right
            result.append(tree.predict)
        return result

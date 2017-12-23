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

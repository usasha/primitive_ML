import numpy as np

class LinearRegression(object):
    def __init__(self, learn_rate):
        self.weights = []
        self.learn_rate = learn_rate

    def _compute_gradient(self, X, Y):
        return 2 * (X.T.dot(X.dot(self.weights) - Y)) / len(Y)

    def fit(self, X, Y):
        intercept = np.ones((len(X), 1))
        X = np.append(intercept, X, 1)

        self.weights = np.zeros((len(X.T)))
        grad = np.ones((len(X.T),1))

        while np.linalg.norm(grad) > 0.001:
            grad = self._compute_gradient(X, Y)
            self.weights -= grad * self.learn_rate

    def predict(self, X):
        intercept = np.ones((len(X),1))
        X = np.append(intercept, X, 1)

        return X.dot(self.weights)

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as LR


class LinearRegression(object):
    def __init__(self, learn_rate):
        self.weights = []
        self.learn_rate = learn_rate

    def fit(self, X, Y):
        intercept = np.ones((len(X), 1))
        X = np.append(intercept, X, 1)

        self.weights = np.zeros((len(X.T)))
        grad = np.ones((len(X.T),1))

        while np.linalg.norm(grad) > 0.001:
        # for i in range(10000):
            grad = 2 * (X.T.dot(X.dot(self.weights) - Y)) / len(Y)
            self.weights -= grad * self.learn_rate
            # self.learn_rate *= 0.9

    def predict(self, X):
        intercept = np.ones((len(X),1))
        X = np.append(intercept, X, 1)

        return X.dot(self.weights)


# dataset = datasets.load_boston()
dataset = datasets.load_diabetes()
scaler = StandardScaler()
data = scaler.fit_transform(dataset.data)
target = dataset.target

lr = LinearRegression(0.01)
lr.fit(data[:-100], target[:-100])
result = lr.predict(data[-100:])


scores = mean_squared_error(target[-100:], result[-100:])
print(scores)
print("----------------")
lr2 = LR()
lr2.fit(data[:-100], target[:-100])
result = lr2.predict(data[-100:])
scores = mean_squared_error(target[-100:], result[-100:])
print(scores)
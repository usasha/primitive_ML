import numpy as np
import abc


class NaiveBayesBase(abc.ABC):
    def __init__(self):
        self.probabilities = []
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    @abc.abstractmethod
    def _get_proba(self, feature, point, category):
        pass

    def predict(self, x):
        result = []
        for point in x:
            probabilities = []
            for category in np.unique(self.y):
                category_prob = 1
                for feature in range(len(point)):
                    category_prob *= self._get_proba(feature, point, category)

                probabilities.append(category_prob)
            result.append(np.unique(self.y)[np.argmax(probabilities)])
        return result


class NaiveBayesMultinominal(NaiveBayesBase):
    def _get_proba(self, feature, point, category):
        p_a = (self.y == category).sum() / len(self.y)
        p_ba = ((self.x[self.y == category][:, feature] == point[feature]).sum()
                / len(self.x[self.y == category]))
        p_b = (self.x[:, feature] == point[feature]).sum() / len(self.x) + 1e-100

        return p_a * p_ba / p_b


class NaiveBayesGaussian(NaiveBayesBase):
    def _get_proba(self, feature, point, category):
        std = np.std(self.x[self.y == category])
        u = np.mean(self.x[self.y == category])
        prob = (1 / np.sqrt(2 * np.pi * std ** 2)
                * np.exp(-(point[feature] - u) ** 2 / 2 * std ** 2))

        return prob

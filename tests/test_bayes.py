import pytest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import sys
from sklearn import datasets
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from bayes import NaiveBayesMultinominal
from bayes import NaiveBayesGaussian


def test_multinominal_init():
    clf = NaiveBayesMultinominal()
    assert clf.x is None
    assert clf.y is None
    assert clf.probabilities == []


def test_multinominal_fit():
    dataset = datasets.load_digits()
    data = dataset.data
    target = dataset.target

    clf = NaiveBayesMultinominal()
    clf.fit(data, target)
    assert np.array_equal(clf.x, data)
    assert np.array_equal(clf.y, target)


def test_multinominal_get_proba():
    dataset = datasets.load_digits()
    x = dataset.data
    y = dataset.target

    clf = NaiveBayesMultinominal()
    clf.fit(x, y)
    class_0 = clf._get_proba(2, x[0], 0)
    class_1 = clf._get_proba(2, x[0], 1)
    class_2 = clf._get_proba(2, x[0], 2)
    assert class_0 > class_1 > class_2


def test_multinominal_predict():
    dataset = datasets.load_digits()
    x = dataset.data
    y = dataset.target

    clf = NaiveBayesMultinominal()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    precision = precision_score(y, y_pred, average='micro')
    recall = recall_score(y, y_pred, average='micro')

    assert precision > 0.8
    assert recall > 0.8


def test_gauss_init():
    clf = NaiveBayesGaussian()
    assert clf.x is None
    assert clf.y is None
    assert clf.probabilities == []


def test_gauss_fit():
    dataset = datasets.load_digits()
    data = dataset.data
    target = dataset.target

    clf = NaiveBayesGaussian()
    clf.fit(data, target)
    assert np.array_equal(clf.x, data)
    assert np.array_equal(clf.y, target)


def test_gauss_get_proba():
    dataset = datasets.load_iris()
    x = dataset.data
    y = dataset.target

    clf = NaiveBayesGaussian()
    clf.fit(x, y)
    class_0 = clf._get_proba(0, x[0], 0)
    class_1 = clf._get_proba(0, x[0], 1)
    class_2 = clf._get_proba(0, x[0], 2)
    assert class_0 < class_1 < class_2


def test_gauss_predict():
    dataset = datasets.load_iris()
    x = dataset.data
    y = dataset.target

    clf = NaiveBayesGaussian()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    precision = precision_score(y, y_pred, average='micro')
    recall = recall_score(y, y_pred, average='micro')

    assert precision > 0.7
    assert recall > 0.7

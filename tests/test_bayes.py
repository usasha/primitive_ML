import pytest
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
    data = dataset.data
    target = dataset.target

    clf = NaiveBayesMultinominal()
    clf.fit(data, target)
    assert clf._get_proba(2, data[0], 0) == pytest.approx(0.01168, 0.001)
    assert clf._get_proba(2, data[0], 1) == pytest.approx(0.00612, 0.001)
    assert clf._get_proba(2, data[0], 2) == pytest.approx(0.003895, 0.001)

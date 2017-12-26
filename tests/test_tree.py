import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from trees import TreeNode
from trees import DecisionTree


def test_tree_node_init():
    node = TreeNode(2)
    assert node.max_depth == 2
    assert node.feature is None
    assert node.threshold is None
    assert node.predict is None
    assert node.left is None
    assert node.right is None


def test_decision_tree_init():
    tree = DecisionTree(3)
    assert type(tree.root) is TreeNode
    assert tree.root.max_depth == 3
    assert tree.root.feature is None
    assert tree.root.threshold is None
    assert tree.root.predict is None
    assert tree.root.left is None
    assert tree.root.right is None


def test_decision_tree_inf_criteria():
    tree = DecisionTree(3)
    result = tree._inf_criteria(np.array([1,1,1,1,3,11,2]))
    assert result == pytest.approx(11.551, 0.001)
    assert tree._inf_criteria(np.array([])) == 0
    

def test_decision_tree_split():
    tree = DecisionTree(3)
    x = np.array([[1,2],[2,3],[4,5]])
    y = np.array([0,1,1])
    tree.split(x, y, tree.root)
    assert tree.root.left.predict == 0
    assert tree.root.right.predict == 1
    assert tree.root.left.left is None
    assert tree.root.left.right is None
    assert tree.root.right.left is None
    assert tree.root.right.right is None
    assert tree.root.max_depth == 3
    assert tree.root.left.max_depth == 2
    assert tree.root.right.max_depth == 2
    assert tree.root.feature == 1
    assert tree.root.threshold == 3
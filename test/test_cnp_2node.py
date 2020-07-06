import unittest

import numpy as np

from .utils import two_node_tree


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = two_node_tree()

    def test_sinkhorn_single_0(self):
        self.graph.get_node('VarNode_000').constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.get_node('VarNode_000').isconstrained = True
        self.graph.constrained_names.append('VarNode_000')
        self.graph.run_bp()
        node_equal = np.isclose(
            self.graph.get_node('VarNode_001').marginal(),
            np.array([0.375, 0.625])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.get_node('VarNode_000').marginal(),
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))

    def test_sinkhorn_single_1(self):
        self.graph.get_node('VarNode_001').constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.get_node('VarNode_001').isconstrained = True
        self.graph.constrained_names.append('VarNode_001')
        self.graph.run_bp()
        node_equal = np.isclose(
            self.graph.get_node('VarNode_001').marginal(),
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.get_node('VarNode_000').marginal(),
            np.array([23.0 / 76, 53.0 / 76])
        )
        self.assertTrue(all(node_equal))

    def test_sinkhorn_2(self):
        self.graph.get_node('VarNode_000').constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.get_node('VarNode_000').isconstrained = True
        self.graph.get_node('VarNode_001').constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.get_node('VarNode_001').isconstrained = True
        self.graph.constrained_names.append('VarNode_000')
        self.graph.constrained_names.append('VarNode_001')
        self.graph.run_bp()
        node_equal = np.isclose(
            self.graph.get_node('VarNode_000').marginal(),
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.get_node('VarNode_001').marginal(),
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))

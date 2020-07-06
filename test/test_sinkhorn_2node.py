import unittest

import numpy as np

from .utils import two_node_tree


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = two_node_tree()

    def test_sinkhorn_single_0(self):
        self.graph.varnode_recorder['VarNode_000'].constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.varnode_recorder['VarNode_000'].isconstrained = True
        self.graph.constrained_names.append('VarNode_000')
        self.graph.sinkhorn()
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_000'].sinkhorn,
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_001'].sinkhorn,
            np.array([0.375, 0.625])
        )
        self.assertTrue(all(node_equal))

    def test_sinkhorn_single_1(self):
        self.graph.varnode_recorder['VarNode_001'].constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.varnode_recorder['VarNode_001'].isconstrained = True
        self.graph.constrained_names.append('VarNode_001')
        self.graph.sinkhorn()
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_001'].sinkhorn,
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_000'].sinkhorn,
            np.array([23.0 / 76, 53.0 / 76])
        )
        self.assertTrue(all(node_equal))

    def test_sinkhorn_2(self):
        self.graph.varnode_recorder['VarNode_000'].constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.varnode_recorder['VarNode_000'].isconstrained = True
        self.graph.varnode_recorder['VarNode_001'].constrained_marginal \
            = np.array([0.5, 0.5])
        self.graph.varnode_recorder['VarNode_001'].isconstrained = True
        self.graph.constrained_names.append('VarNode_000')
        self.graph.constrained_names.append('VarNode_001')
        self.graph.sinkhorn()
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_000'].sinkhorn,
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(
            self.graph.varnode_recorder['VarNode_001'].sinkhorn,
            np.array([0.5, 0.5])
        )
        self.assertTrue(all(node_equal))

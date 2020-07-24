import unittest

import numpy as np
from cbp.graph.coef_policy import bp_policy, factor_policy

from .utils import two_node_tree


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = two_node_tree()
        self.graph.coef_policy = bp_policy

    def test_tree_bp(self):
        self.graph.tree_bp()
        node_equal = np.isclose(np.array([0.2, 0.8]),
                                self.graph.get_node("VarNode_000").marginal())
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(np.array([0.24, 0.76]),
                                self.graph.get_node("VarNode_001").marginal())
        self.assertTrue(all(node_equal))

    def test_marginal_brutal_force(self):
        self.graph.exact_marginal()
        node_equal = np.isclose(np.array([0.2, 0.8]),
                                self.graph.get_node("VarNode_000").bfmarginal)
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(np.array([0.24, 0.76]),
                                self.graph.get_node("VarNode_001").bfmarginal)
        self.assertTrue(all(node_equal))

    def test_marginal_bp(self):
        self.graph.run_bp()
        node_equal = np.isclose(np.array([0.2, 0.8]),
                                self.graph.get_node("VarNode_000").marginal())
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(np.array([0.24, 0.76]),
                                self.graph.get_node("VarNode_001").marginal())
        self.assertTrue(all(node_equal))

    def test_cnp_coef(self):
        self.graph.bake()
        self.graph.get_node("VarNode_000").node_coef = 1.0 / 4
        self.graph.get_node("VarNode_001").node_coef = 1.0 / 4
        self.graph.get_node("FactorNode_000").node_coef = 1.0 / 2
        self.graph.get_node("FactorNode_000").set_i_alpha(
            "VarNode_000", 1.0 / 4)
        self.graph.get_node("FactorNode_000").set_i_alpha(
            "VarNode_001", 1.0 / 4)
        for node in self.graph.nodes:  # update new cnp coef
            node.cal_cnp_coef()
        self.graph.norm_product_bp()
        node_equal = np.isclose(np.array([0.2, 0.8]),
                                self.graph.get_node("VarNode_000").marginal(),
                                atol=1e-3, rtol=1e-2)
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(np.array([0.24, 0.76]),
                                self.graph.get_node("VarNode_001").marginal(),
                                atol=1e-3, rtol=1e-2)
        self.assertTrue(all(node_equal))

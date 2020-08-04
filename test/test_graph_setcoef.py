import unittest

from cbp.graph.coef_policy import avg_policy, bp_policy

from .utils import six_node_graph


class TestSetCNPCoef(unittest.TestCase):
    def setUp(self):
        self.graph = six_node_graph()

    def test_avg_policy(self):
        self.graph.coef_policy = avg_policy
        self.graph.bake()

        for node in self.graph.nodes:
            self.assertAlmostEqual(node.node_coef, 1.0 / 11)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_000").get_hat_c_ialpha("VarNode_000"), 2.0 / 11)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_000").get_hat_c_ialpha("VarNode_001"), 10.0 / 11)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_002").get_hat_c_ialpha("VarNode_001"), 6.0 / 11)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_004").get_hat_c_ialpha("VarNode_005"), 2.0 / 11)

    def test_bp_policy(self):
        self.graph.coef_policy = bp_policy
        self.graph.bake()

        self.assertAlmostEqual(
            self.graph.get_node("VarNode_001").node_coef, -2)
        self.assertAlmostEqual(
            self.graph.get_node("VarNode_003").node_coef, -2)

        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_000").get_hat_c_ialpha("VarNode_000"), 1)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_000").get_hat_c_ialpha("VarNode_001"), 1)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_002").get_hat_c_ialpha("VarNode_001"), 1)
        self.assertAlmostEqual(self.graph.get_node(
            "FactorNode_004").get_hat_c_ialpha("VarNode_005"), 1)

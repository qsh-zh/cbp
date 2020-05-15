import unittest
from pathlib import Path

import numpy as np
from numpy.random import RandomState

from cbp.graph import GraphModel
from cbp.graph.coef_policy import avg_policy, bp_policy
from cbp.node import FactorNode, VarNode


class TestLargeGraph(unittest.TestCase):
    def setUp(self):
        Path('data').mkdir(exist_ok=True)
        self.rng = RandomState(1)
        self.graph = GraphModel(coef_policy=avg_policy)
        self.rv_dim = 2

        # init varnode
        for _ in range(6):
            potential = np.ones([self.rv_dim])
            varnode = VarNode(self.rv_dim, potential)
            self.graph.add_varnode(varnode)

        # init factornode
        edges = [
            [0, 1],
            [2, 1],
            [1, 3],
            [3, 4],
            [3, 5]
        ]
        for item in edges:
            potential = self.rng.normal(size=[self.rv_dim, self.rv_dim])
            factorname = [f"VarNode_{data:03d}" for data in item]
            factornode = FactorNode(factorname, np.exp(potential))
            self.graph.add_factornode(factornode)
        self.graph.plot(f"data/test_graph_setcoef.png")

    def test_avg_policy(self):
        self.graph.init_node_recorder()
        self.graph.bake()
        for node in self.graph.nodes:
            node.cal_cnp_coef()

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
        self.graph.init_node_recorder()
        self.graph.bake()
        for node in self.graph.nodes:
            node.cal_cnp_coef()

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


if __name__ == '__main__':
    unittest.main()

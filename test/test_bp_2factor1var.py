import unittest

import numpy as np
from cbp.graph import GraphModel
from cbp.graph.coef_policy import bp_policy
from cbp.node import FactorNode, VarNode


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = GraphModel(coef_policy=bp_policy)

        # init varnode
        potential = np.ones(2)
        self.varnode_1 = VarNode(2, potential)
        self.graph.add_varnode(self.varnode_1)

        varnode_name = [node.name for node in [self.varnode_1]]
        factor_potential = np.array(
            [3, 2]
        )
        self.factornode_1 = FactorNode(varnode_name, factor_potential)
        self.graph.add_factornode(self.factornode_1)

        factor_potential_2 = np.array(
            [1, 4]
        )
        self.factornode_2 = FactorNode(varnode_name, factor_potential_2)
        self.graph.add_factornode(self.factornode_2)

    def test_marginal_brutal_force(self):
        self.graph.exact_marginal()
        node_equal = np.isclose(np.array([3.0 / 11, 8.0 / 11]),
                                self.graph.get_node("VarNode_000").bfmarginal)
        self.assertTrue(all(node_equal))

    def test_marginal_bp(self):
        self.graph.run_bp()
        node_equal = np.isclose(np.array([3.0 / 11, 8.0 / 11]),
                                self.graph.get_node("VarNode_000").marginal())
        self.assertTrue(all(node_equal))

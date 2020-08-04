import unittest

import numpy as np
from cbp.node import FactorNode, VarNode
from cbp.graph import GraphModel


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = GraphModel()

        # init varnode
        self.varnode_1 = VarNode(2)
        self.graph.add_varnode(self.varnode_1)

        varnode_name = [node.name for node in [self.varnode_1]]
        factor_potential = np.array(
            [3, 2]
        )
        self.factornode = FactorNode(varnode_name, factor_potential)
        self.graph.add_factornode(self.factornode)

    def test_marginal_brutal_force(self):
        self.graph.exact_marginal()
        node_equal = np.isclose(
            np.array([0.6, 0.4]),
            self.graph.get_node("VarNode_000").bfmarginal)
        self.assertTrue(all(node_equal))

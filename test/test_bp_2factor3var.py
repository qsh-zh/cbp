import unittest

import numpy as np

from .utils import three_node_tree


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = three_node_tree()

    def test_marginal_bp(self):
        self.graph.run_bp()
        self.graph.exact_marginal()

        isequal_list = []

        for item in self.graph.varnode_recorder.values():
            marginal_equal = np.isclose(item.bfmarginal, item.marginal())
            isequal_list.append(all(marginal_equal))

        self.assertTrue(all(isequal_list))

    def test_marginal_brutal_force(self):
        self.graph.exact_marginal()
        node_equal = np.isclose(np.array([70.0 / 215, 145.0 / 215]),
                                self.graph.get_node("VarNode_000").bfmarginal)
        self.assertTrue(all(node_equal))
        node_equal = np.isclose(np.array([120.0 / 215, 95.0 / 215]),
                                self.graph.get_node("VarNode_001").bfmarginal)
        self.assertTrue(all(node_equal))

        node_equal = np.isclose(np.array([55.0 / 215, 160.0 / 215]),
                                self.graph.get_node("VarNode_002").bfmarginal)
        self.assertTrue(all(node_equal))

import unittest
import numpy as np
from cbp.graph import DiscreteGraph
from cbp.node import VarNode, FactorNode


class TestDiscreteGraph(unittest.TestCase):
    def setUp(self):
        self.graph = DiscreteGraph()
        self.dim = 4
        self.rng = np.random.RandomState(1)

    def test_add_varnode(self):
        var1 = VarNode(self.dim)
        var2 = VarNode(self.dim)
        self.graph.add_varnode(var1)
        self.graph.add_varnode(var2)
        self.graph.init_node_list()

        self.assertEqual(self.graph.cnt_varnode, 2)
        self.assertEqual(id(self.graph.get_node('VarNode_000')), id(var1))

        c_var1 = VarNode(
            self.dim, constrained_marginal=np.ones(
                self.dim) / self.dim)

        self.graph.add_varnode(c_var1)
        self.graph.init_node_list()
        self.assertEqual(self.graph.constrained_names, ['VarNode_002'])

    def test_delete_var(self):
        self.test_add_varnode()
        self.graph.delete_node('VarNode_000')
        self.graph.init_node_list()
        self.assertNotIn('VarNode_000', self.graph.varnode_recorder)

    def test_add_factor_node(self):
        self.test_delete_var()

        factor = FactorNode(['VarNode_001', 'VarNode_002'],
                            potential=self.rng.uniform(size=(self.dim, self.dim)))
        self.graph.add_factornode(factor)
        self.graph.init_node_list()
        self.assertEqual(id(self.graph.get_node('FactorNode_000')), id(factor))
        self.graph.delete_node('FactorNode_000')
        self.assertEqual(len(self.graph.factornode_recorder), 0)

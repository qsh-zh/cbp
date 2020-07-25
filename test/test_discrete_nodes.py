import unittest
import numpy as np
from cbp.node import VarNode, FactorNode


class TestDiscreteNode(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)

    def test_var_equal(self):
        dim = 10
        _potential = self.rng.uniform(size=dim)
        node = VarNode(rv_dim=dim, potential=_potential)
        target_node = VarNode(rv_dim=dim, potential=_potential)
        self.assertEqual(node, target_node)

        _target_node = VarNode(dim, self.rng.uniform(size=dim))
        self.assertNotEqual(node, _target_node)

    def test_constrain_var_equal(self):
        dim = 10
        _potential = self.rng.uniform(size=dim)
        node = VarNode(dim, _potential, _potential / np.sum(_potential))
        target_node = VarNode(dim, _potential, _potential / np.sum(_potential))
        self.assertEqual(node, target_node)

        new_p = self.rng.uniform(size=dim)
        _target_node = VarNode(dim, _potential, new_p / np.sum(new_p))
        self.assertNotEqual(node, _target_node)

        _target_node = VarNode(dim, new_p, _potential / np.sum(_potential))
        self.assertNotEqual(node, _target_node)

    def test_factor_equal(self):
        dim = (4, 10)
        _potential = self.rng.uniform(size=dim)

        node = FactorNode(['VarNode_000', 'VarNode_001'], _potential)
        _target = FactorNode(['VarNode_000', 'VarNode_001'], _potential)
        self.assertEqual(node, _target)

        new_p = self.rng.uniform(size=dim)
        _target = FactorNode(['VarNode_000', 'VarNode_001'], new_p)
        self.assertNotEqual(node, _target)

        _target = FactorNode(
            ['VarNode_000', 'VarNode_001', 'VarNode_002'], _potential)
        self.assertNotEqual(node, _target)

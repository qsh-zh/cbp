import unittest

import numpy as np
from cbp.node import GOFactorNode, GOVarNode


class TestGOVarNode(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)

    def test_var_construct(self):
        node = GOVarNode(self.rng.normal(size=(10)))

    def test_factor_construct(self):
        connected_names = ['VarNode_000', 'VarNode_001']
        node = GOFactorNode(connected_names,
                            loc=self.rng.normal(size=10),
                            scale=self.rng.uniform(size=10))

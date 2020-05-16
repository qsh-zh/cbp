import unittest

import numpy as np
from cbp.graph.coef_policy import bp_policy

from .utils import six_node_graph


class TestBpSxiVar(unittest.TestCase):
    def setUp(self):
        self.graph = six_node_graph()

    def test_marginal_bp(self):
        self.graph.coef_policy = bp_policy
        self.graph.run_bp()
        self.graph.exact_marginal()

        isequal_list = []

        for item in self.graph.varnode_recorder.values():
            marginal_equal = np.isclose(item.bfmarginal, item.marginal())
            isequal_list.append(all(marginal_equal))

        self.assertTrue(all(isequal_list))

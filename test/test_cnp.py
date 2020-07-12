import unittest

import numpy as np
from cbp.builder import HMMBuilder, HMMZeroBuilder, LineBuilder
from cbp.graph.coef_policy import bp_policy, avg_policy, factor_policy

from .utils import sinkhorn_bp_equal


class TestGraph(unittest.TestCase):
    def test_hmm_policy(self):
        rng = np.random.RandomState(1)
        for i in range(5):
            for policy in [bp_policy, avg_policy, factor_policy]:
                num_node = int(rng.randint(2, 6))
                node_dim = int(rng.randint(4, 6))
                print(
                    f"Run {i}-th experiment, {num_node} nodes with {node_dim} status")
                self.graph = HMMBuilder(num_node, node_dim, policy)()
                self.graph.run_cnp()
                self.graph.sinkhorn()
                self.assertTrue(all(sinkhorn_bp_equal(self.graph, num_node)))

    def test_line_policy(self):
        rng = np.random.RandomState(1)
        for i in range(5):
            for policy in [bp_policy, avg_policy, factor_policy]:
                num_node = int(rng.randint(2, 4))
                node_dim = int(rng.randint(2, 5))
                print(
                    f"Run {i}-th experiment, {num_node} nodes with {node_dim} status")
                self.graph = LineBuilder(num_node, node_dim, avg_policy)()
                self.graph.run_cnp()
                self.graph.sinkhorn()
                self.assertTrue(all(sinkhorn_bp_equal(self.graph, num_node)))

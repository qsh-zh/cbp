import unittest

import numpy as np
from cbp.builder import HMMBuilder, HMMZeroBuilder, LineBuilder
from cbp.graph.coef_policy import bp_policy

from .utils import sinkhorn_bp_equal


class TestITSbp(unittest.TestCase):
    def test_itsbp_get_loop(self):
        graph = HMMBuilder(3, 2, bp_policy)()
        graph.bake()
        _, looplink = graph.its_next_looplink()
        nodenames = [node.name for node in looplink]
        expected_names = [
            f'VarNode_{1:03d}',
            f'FactorNode_{0:03d}',
            f'VarNode_{0:03d}',
            f'FactorNode_{1:03d}',
            f'VarNode_{2:03d}',
            f'FactorNode_{2:03d}',
            f'VarNode_{3:03d}']
        self.assertEqual(nodenames, expected_names)

        _, looplink = graph.its_next_looplink()
        nodenames = [node.name for node in looplink]
        expected_names = [
            f'VarNode_{3:03d}',
            f'FactorNode_{2:03d}',
            f'VarNode_{2:03d}',
            f'FactorNode_{3:03d}',
            f'VarNode_{4:03d}',
            f'FactorNode_{4:03d}',
            f'VarNode_{5:03d}']
        self.assertEqual(nodenames, expected_names)

        _, looplink = graph.its_next_looplink()
        nodenames = [node.name for node in looplink]
        expected_names = [
            f'VarNode_{5:03d}',
            f'FactorNode_{4:03d}',
            f'VarNode_{4:03d}',
            f'FactorNode_{3:03d}',
            f'VarNode_{2:03d}',
            f'FactorNode_{1:03d}',
            f'VarNode_{0:03d}',
            f'FactorNode_{0:03d}',
            f'VarNode_{1:03d}']
        self.assertEqual(nodenames, expected_names)

    def test_itsbp_line(self):
        for _ in range(10):
            num_node = int(np.random.randint(3, 10))
            node_dim = int(np.random.randint(2, 5))
            self.graph = LineBuilder(num_node, node_dim, bp_policy)()
            self.graph.run_bp(self.graph.iterative_scaling)
            self.graph.sinkhorn()
            self.assertTrue(
                all(sinkhorn_bp_equal(self.graph, len_node=num_node)))

    def test_itsbp_hmm(self):
        for i in range(10):
            num_node = int(np.random.randint(3, 6))
            node_dim = int(np.random.randint(2, 5))
            self.graph = HMMBuilder(num_node, node_dim, bp_policy)()
            self.graph.sinkhorn()
            print(f"{i}-th test, run {node_dim} status with {num_node} nodes")
            self.graph.run_bp(self.graph.iterative_scaling)
            self.assertTrue(
                all(sinkhorn_bp_equal(self.graph, len_node=num_node)))

    # @unittest.skip("Expensive test!")
    def test_zero_hmm(self):
        graph = HMMZeroBuilder(3, 3, bp_policy)()
        graph.sinkhorn()
        graph.run_bp(graph.iterative_scaling)
        self.assertTrue(
            all(sinkhorn_bp_equal(graph, len_node=3)))


if __name__ == '__main__':
    unittest.main()

import timeit
import unittest
from test.utils import sinkhorn_bp_equal, two_node_tree

import numpy as np
from cbp.builder import HMMBuilder, HMMZeroBuilder, LineBuilder
from cbp.configs import TestConfig
from cbp.graph.coef_policy import bp_policy


def next_links(graph):
    _, link = graph.its_next_looplink()
    nodenames = [node.name for node in link]
    return nodenames


class TestITSbp(unittest.TestCase):
    def test_itsbp_get_loop_single(self):
        graph = two_node_tree()
        graph.bake()
        _, link = graph.its_next_looplink()
        nodenames = [node.name for node in link]
        expected_names = [
            "VarNode_000",
            "FactorNode_000",
            "VarNode_001"
        ]
        self.assertEqual(nodenames, expected_names)

    zero2one = [
        f'VarNode_{1:03d}',
        f'FactorNode_{0:03d}',
        f'VarNode_{0:03d}',
        f'FactorNode_{1:03d}',
        f'VarNode_{2:03d}',
        f'FactorNode_{2:03d}',
        f'VarNode_{3:03d}']
    one2two = [
        f'VarNode_{3:03d}',
        f'FactorNode_{2:03d}',
        f'VarNode_{2:03d}',
        f'FactorNode_{3:03d}',
        f'VarNode_{4:03d}',
        f'FactorNode_{4:03d}',
        f'VarNode_{5:03d}']
    two2zero = [
        f'VarNode_{5:03d}',
        f'FactorNode_{4:03d}',
        f'VarNode_{4:03d}',
        f'FactorNode_{3:03d}',
        f'VarNode_{2:03d}',
        f'FactorNode_{1:03d}',
        f'VarNode_{0:03d}',
        f'FactorNode_{0:03d}',
        f'VarNode_{1:03d}']

    def test_get_loop_forward(self):
        graph = HMMBuilder(3, 2, bp_policy)()
        graph.bake()
        self.assertEqual(next_links(graph), TestITSbp.zero2one)
        self.assertEqual(next_links(graph), TestITSbp.one2two)
        self.assertEqual(next_links(graph), TestITSbp.two2zero)
        self.assertEqual(next_links(graph), TestITSbp.zero2one)

    def test_get_loop_backward(self):
        graph = HMMBuilder(3, 2, bp_policy)()
        graph.cfg = TestConfig()
        graph.bake()
        self.assertEqual(next_links(graph), TestITSbp.zero2one)
        self.assertEqual(next_links(graph), TestITSbp.one2two)
        self.assertEqual(list(reversed(next_links(graph))), TestITSbp.one2two)
        self.assertEqual(list(reversed(next_links(graph))), TestITSbp.zero2one)
        self.assertEqual(next_links(graph), TestITSbp.zero2one)

    def _profile_hmm_schedule(self, cfg=None):
        rng = np.random.RandomState(1)
        for _ in range(10):
            num_node = int(rng.randint(13, 15))
            node_dim = int(rng.randint(4, 8))
            graph = HMMBuilder(num_node, node_dim, bp_policy)()
            if cfg:
                graph.cfg = cfg
            graph.run_bp()

    @unittest.skip("Proved! Loopy schedule is fast!")
    def test_cmp_schedule(self):
        print(timeit.timeit(
            lambda: self._profile_hmm_schedule(
                TestConfig()),
            number=3))
        print(timeit.timeit(lambda: self._profile_hmm_schedule(), number=3))

    def test_itsbp_line(self):
        rng = np.random.RandomState(1)
        for _ in range(10):
            num_node = int(rng.randint(3, 10))
            node_dim = int(rng.randint(2, 5))
            self.graph = LineBuilder(num_node, node_dim, bp_policy)()
            self.graph.run_bp()
            self.graph.sinkhorn()
            self.assertTrue(
                all(sinkhorn_bp_equal(self.graph, len_node=num_node)))

    def test_itsbp_hmm(self):
        rng = np.random.RandomState(1)
        for i in range(10):
            num_node = int(rng.randint(3, 6))
            node_dim = int(rng.randint(2, 5))
            self.graph = HMMBuilder(num_node, node_dim, bp_policy)()
            self.graph.sinkhorn()
            self.graph.run_bp()
            self.assertTrue(
                all(sinkhorn_bp_equal(self.graph, len_node=num_node)))

    def test_zero_hmm(self):
        graph = HMMZeroBuilder(3, 3, bp_policy)()
        graph.sinkhorn()
        graph.run_bp()
        self.assertTrue(all(sinkhorn_bp_equal(graph, len_node=3)))


if __name__ == '__main__':
    unittest.main()

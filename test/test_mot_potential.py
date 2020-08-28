import unittest
import numpy as np
from cbp.node import MOTCluster
from .utils import three_node_tree
import cbp.utils.np_utils as npu


class TestPotential(unittest.TestCase):
    def test_2var1factor(self):
        graph = three_node_tree()
        var0 = graph.get_node('VarNode_000')
        var1 = graph.get_node('VarNode_001')
        factor0 = graph.get_node('FactorNode_000')
        cluster = MOTCluster(None, [var0, var1], [factor0])

        self.assertTrue((cluster.potential == factor0.potential).all())

    def test_3var2factor(self):
        graph = three_node_tree()
        factor0 = graph.get_node('FactorNode_000')
        factor1 = graph.get_node('FactorNode_001')
        cluster = MOTCluster(None,
                             list(graph.varnode_recorder.values()),
                             list(graph.factornode_recorder.values()))

        potential = npu.nd_multiexpand(factor0.potential, (2, 2, 2), [0, 1]) * \
            npu.nd_multiexpand(factor1.potential, (2, 2, 2), [1, 2])
        # print(cluster.potential)
        # print(potential)
        self.assertTrue((cluster.potential == potential).all())

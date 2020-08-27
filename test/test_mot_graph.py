from cbp.graph import MOTGraph
from cbp.node import VarNode, FactorNode, MOTCluster, MOTSeperator
import unittest
from .utils import two_node_tree


class TestMOTGraph(unittest.TestCase):
    def test_plot(self):
        graph = two_node_tree()
        var0 = graph.get_node('VarNode_000')
        var0.isconstrained = True
        var1 = graph.get_node('VarNode_001')
        factor = graph.get_node('FactorNode_000')
        mot_graph = MOTGraph()
        mot_graph.add_seperator(MOTSeperator([var0]))
        mot_graph.add_seperator(MOTSeperator([var1]))
        cluster = MOTCluster(['MOTSeperator_000'], [var0])
        # print(cluster.connections)
        mot_graph.add_cluster(cluster)
        mot_graph.add_cluster(MOTCluster(['MOTSeperator_001'], [var1]))
        mot_graph.add_cluster(
            MOTCluster(['MOTSeperator_000', 'MOTSeperator_001'],
                       [var0, var1], [factor]))
        mot_graph.bake()
        mot_graph.plot()

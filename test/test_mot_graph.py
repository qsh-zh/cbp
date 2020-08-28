from cbp.graph import MOTGraph
from cbp.node import VarNode, FactorNode, MOTCluster, MOTSeperator
import unittest
from .utils import two_node_tree, six_node_graph
import numpy as np
import cbp.utils.np_utils as npu
from cbp.builder import HMMBuilder
# TODO: remove it to single file
from cbp.graph.coef_policy import bp_policy


def cgmTree2mot(cgm):
    var2seperator = {}
    motgraph = MOTGraph()
    for varnode in cgm.varnode_recorder.values():
        if varnode not in cgm.leaf_nodes:
            sep_str = motgraph.add_seperator(MOTSeperator([varnode]))
            var2seperator[varnode.name] = sep_str

    for node_name in cgm.constrained_names:
        varnode = cgm.node_recorder[node_name]
        sep_str = motgraph.add_seperator(MOTSeperator([varnode]))
        var2seperator[varnode.name] = sep_str
        # TODO: discuss is it good to have additional cluster
        cluster = MOTCluster([sep_str], [varnode])
        motgraph.add_cluster(cluster)

    for factor in cgm.factornode_recorder.values():
        connect_sep = []
        for node_str in factor.connected_nodes.keys():
            if node_str in var2seperator:
                connect_sep.append(var2seperator[node_str])

        cluster = MOTCluster(
            connect_sep, list(
                factor.connected_nodes.values()), [factor])
        motgraph.add_cluster(cluster)

    return motgraph


class TestMOTGraph(unittest.TestCase):
    def construct_2var(self):
        graph = two_node_tree()
        var0 = graph.get_node('VarNode_000')
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
        return mot_graph

    def test_plot(self):
        mot_graph = self.construct_2var()
        seperator0 = mot_graph.get_node('MOTCluster_000')
        seperator0.isconstrained = True
        mot_graph.bake()
        mot_graph.plot()

    def test_jt_local_equality(self):
        mot_graph = self.construct_2var()
        mot_graph.tree_bp()
        sep0 = mot_graph.get_node('MOTSeperator_000')
        sep1 = mot_graph.get_node('MOTSeperator_001')
        cluster = mot_graph.get_node('MOTCluster_002')
        self.assertTrue(npu.isequal(sep0.marginal(), np.array([0.2, 0.8])))
        self.assertTrue(npu.isequal(sep1.marginal(), np.array([0.24, 0.76])))
        self.assertTrue(npu.isequal(sep0.marginal(),
                                    cluster.marginal_dims(["VarNode_000"])))
        self.assertTrue(npu.isequal(sep1.marginal(),
                                    cluster.marginal_dims(["VarNode_001"])))

    def test_cgm2mot(self):
        cgm = six_node_graph()
        cgm.bake()
        mot = cgmTree2mot(cgm)
        mot.bake()
        mot.plot('data/mot6.png')

        mot.tree_bp()
        mot_node_marginal = mot.export_node_marginal()
        cgm.tree_bp()
        cgm_node_marginal = cgm.export_marginals()

        for key, value in mot_node_marginal.items():
            self.assertTrue(npu.isequal(value, cgm_node_marginal[key]))

    def test_hmm(self):
        cgm = HMMBuilder(3, 3, bp_policy)()
        cgm.bake()
        mot = cgmTree2mot(cgm)
        mot.bake()
        mot.plot('data/mot_hmm_3.png')

        mot.itsbp()
        mot_node_marginal = mot.export_node_marginal()
        cgm.run_bp()
        cgm_node_marginal = cgm.export_marginals()

        for key, value in mot_node_marginal.items():
            self.assertTrue(npu.isequal(value, cgm_node_marginal[key]))

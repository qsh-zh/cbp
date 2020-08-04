import unittest

import numpy as np
from cbp.graph import GOGraph
from cbp.node import GOFactorNode, GOVarNode, VarNode, FactorNode
from scipy.stats import norm


class TestGOGraph(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)

    def test_factor_discrete(self):
        num_obser = 5
        dim = 10
        connected_names = ['VarNode_000', 'VarNode_001']
        observation = self.rng.normal(size=num_obser)
        mu = self.rng.normal(size=dim)
        sigma = self.rng.uniform(size=dim)

        var = VarNode(dim, node_coef=0)
        go_var = GOVarNode(observation)
        factor = GOFactorNode(connected_names, mu, sigma)

        go_graph = GOGraph()
        go_graph.add_varnode(var)
        go_graph.add_varnode(go_var)
        go_graph.add_factornode(factor)
        go_graph.bake()
        discrete_graph = go_graph.discrete_graph

        self.assertEqual(var, discrete_graph.get_node('VarNode_000'))

        target_var = VarNode(
            rv_dim=num_obser,
            potential=np.ones(num_obser),
            constrained_marginal=np.ones(num_obser) / num_obser, node_coef=0)
        target_var.name = 'VarNode_001'
        target_var.register_connection('FactorNode_000')
        self.assertEqual(target_var, discrete_graph.get_node('VarNode_001'))

        target_potential = []
        for i in range(dim):
            target_potential.append(norm.pdf(observation,
                                             loc=mu[i], scale=sigma[i]))
        target_factor = FactorNode(
            connected_names,
            np.array(target_potential))

        self.assertEqual(
            target_factor,
            discrete_graph.get_node('FactorNode_000'))

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import RandomState
from cbp.graph import GraphModel
from cbp.node import VarNode, FactorNode

from cbp.builder.potential_utils import diagonal_potential_different
from .potential_utils import diagonal_potential, diagonal_potential_conv


class BaseBuilder(ABC):
    def __init__(self, dim, policy, rand_seed=1):
        self.policy = policy
        self.graph = self._create_graph()
        self.node_dim = dim
        self.rng = RandomState(rand_seed)

    def _create_graph(self):
        return GraphModel(coef_policy=self.policy)

    def __call__(self):
        self.init_graph()
        return self.graph

    def add_constrained_node(self, probability=None):
        if probability is None:
            log_probability = self.rng.normal(size=self.node_dim)
            probability = np.exp(log_probability)
        else:
            probability = np.array(probability)

        dim = probability.shape[0]
        varnode = VarNode(dim,
                          constrained_marginal=probability /
                          np.sum(probability))
        self.graph.add_varnode(varnode)
        return varnode

    def add_trivial_node(self, dim=None):
        if dim is None:
            dim = self.node_dim
        varnode = VarNode(dim)
        self.graph.add_varnode(varnode)
        return varnode

    def add_factor(self, name_list, is_obser=False):
        if is_obser:
            factor_potential = diagonal_potential_conv(
                self.node_dim, self.node_dim, self.rng)
        else:
            factor_potential = diagonal_potential(
                self.node_dim, self.node_dim, self.rng)
        factornode = FactorNode(name_list, factor_potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_branch(self, head_node=None, is_constrained=False,
                   prob=None, is_obser=False):
        if head_node is None:
            head_node = f"VarNode_{self.graph.cnt_varnode-1:03d}"
        if is_constrained:
            node = self.add_constrained_node(prob)
        else:
            node = self.add_trivial_node()

        name_list = [head_node, node.name]
        self.add_factor(name_list, is_obser)

    @abstractmethod
    def init_graph(self):
        pass

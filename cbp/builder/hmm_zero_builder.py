import numpy as np
from cbp.node import FactorNode, VarNode

from .hmm_builder import HMMBuilder
from .potential_utils import identity_potential


class HMMZeroBuilder(HMMBuilder):
    def __init__(self, length, d, policy, rand_seed=1):
        super().__init__(length, d, policy, rand_seed=rand_seed)

    def add_factor(self, name_list, is_obser=False):
        factor_potential = identity_potential(
            self.node_dim, self.node_dim, self.rng)
        factornode = FactorNode(name_list, factor_potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_constrained_node(self, probability=None):
        probability = np.zeros(self.node_dim)
        probability[0] = 1
        probability[-1] = 1
        varnode = VarNode(
            self.node_dim,
            constrained_marginal=probability /
            np.sum(probability))
        self.graph.add_varnode(varnode)
        return varnode

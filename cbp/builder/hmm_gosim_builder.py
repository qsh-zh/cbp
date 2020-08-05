from cbp.node import GOFactorNode, GOVarNode
from .base_hmmsim_builder import BaseHMMSimBuilder


class HMMGOSimBuilder(BaseHMMSimBuilder):
    def __init__(self, length, simulator, policy, random_seed=1):
        super().__init__(length, simulator, policy, random_seed=random_seed)

    def add_constrained_node(self, probability=None):
        bins = self.simulator.get_fix_margin(
            self.cnt_constrained_node)
        self.cnt_constrained_node += 1
        node = GOVarNode(bins)
        self.graph.add_varnode(node)
        return node

    def add_emission(self, name_list):
        potential = self.simulator.get_emission_potential()
        factornode = GOFactorNode(name_list, **potential)
        self.graph.add_factornode(factornode)
        return factornode

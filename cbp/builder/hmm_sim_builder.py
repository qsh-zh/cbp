from cbp.node import FactorNode
from .base_hmmsim_builder import BaseHMMSimBuilder


class HMMSimBuilder(BaseHMMSimBuilder):
    def __init__(self, length, simulator, policy,
                 random_seed=1):
        super().__init__(length, simulator, policy, random_seed)

    def add_constrained_node(self, probability=None):
        prob = self.simulator.get_fix_margin(
            self.cnt_constrained_node)
        self.cnt_constrained_node += 1
        return super().add_constrained_node(prob)

    def add_emission(self, name_list):
        potential = self.simulator.get_emission_potential()
        factornode = FactorNode(name_list, potential)
        self.graph.add_factornode(factornode)
        return factornode

    def compare_acc(self):
        return super().compare_acc(self.graph)

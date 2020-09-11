from cbp.node import FactorNode
from .base_hmmsim_builder import BaseHMMSimBuilder


class HMMSimBuilder(BaseHMMSimBuilder):

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

    def compare_acc(self):  # pylint: disable=arguments-differ
        return super().compare_acc(self.graph)

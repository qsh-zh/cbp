from cbp.node import FactorNode

from .hmm_builder import HMMBuilder


class HMMSimBuilder(HMMBuilder):
    def __init__(self, length, simulator, policy,
                 random_seed=1):
        super().__init__(length, simulator.record.state_num, policy, random_seed)
        self.simulator = simulator
        self.cnt_constrained_node = 0

    def fix_initpotential(self, potential=None):
        first_node = self.graph.get_node(f"VarNode_{0:03d}")
        if potential is None:
            potential = self.simulator.get_hidden_margin(0)
        first_node.potential = potential

    def add_constrained_node(self, probability=None):
        prob = self.simulator.get_fix_margin(
            self.cnt_constrained_node)
        self.cnt_constrained_node += 1
        return super().add_constrained_node(prob)

    def add_factor(self, name_list, is_conv=False):
        """add factor to hmm graph

        :param name_list: connected node
        :type name_list: list
        :param is_conv: is emit or not , defaults to False-- transition
        :type is_conv: bool, optional
        :return: FactorNode
        :rtype: cbp.FactorNode
        """
        if is_conv:  # emit
            potential = self.simulator.get_emission_potential()
        else:
            potential = self.simulator.get_transition_potential()
        factornode = FactorNode(name_list, potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_branch(self, head_node=None, is_constrained=False,
                   prob=None, is_conv=False):
        if is_conv:  # emit
            return super().add_branch(head_node, True, prob, is_conv)
        return super().add_branch(head_node, False, prob, is_conv)

    def example(self, num_sample):
        self.simulator.reset()
        self.simulator.example(num_sample)

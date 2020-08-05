from abc import abstractmethod
import numpy as np

from cbp.node import FactorNode
from .hmm_builder import HMMBuilder


class BaseHMMSimBuilder(HMMBuilder):
    def __init__(self, length, simulator, policy,
                 random_seed=1):
        super().__init__(length, simulator.record.state_num,
                         policy, random_seed)
        self.simulator = simulator
        self.cnt_constrained_node = 0

    def fix_initpotential(self, potential=None):
        first_node = self.graph.get_node(f"VarNode_{0:03d}")
        if potential is None:
            potential = self.simulator.get_init_potential()
        first_node.potential = potential

    def add_factor(self, name_list, is_obser=False):
        """add factor to hmm graph

        :param name_list: connected node
        :type name_list: list
        :param is_obser: is emit or not , defaults to False-- transition
        :type is_obser: bool, optional
        :return: FactorNode
        :rtype: cbp.FactorNode
        """
        if is_obser:  # emit
            return self.add_emission(name_list)

        return self.add_transition(name_list)

    def add_transition(self, name_list):
        potential = self.simulator.get_transition_potential()
        factornode = FactorNode(name_list, potential)
        self.graph.add_factornode(factornode)
        return factornode

    @abstractmethod
    def add_emission(self, name_list):
        """interface to implement
        """

    def add_branch(self, head_node=None, is_constrained=False,
                   prob=None, is_obser=False):
        if is_obser:  # emit
            return super().add_branch(head_node, True, prob, is_obser)
        return super().add_branch(head_node, False, prob, is_obser)

    def example(self, num_sample):
        self.simulator.example(num_sample)

    def compare_acc(self, graph):
        infer_marginal = []
        gt_marginal = []
        for i in range(0, self.hmm_length, 2):
            infer_marginal.append(graph.get_node(f'VarNode_{i:03d}').marginal())
            gt_marginal.append(self.simulator.get_hidden_margin(i // 2))
        return np.sum(np.abs(np.array(infer_marginal) - np.array(gt_marginal)))

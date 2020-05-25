from cbp.node import FactorNode

from .hmm_builder import HMMBuilder


class HMMSimBuilder(HMMBuilder):
    """HMM builder according the data from simulator with time-homogenous potential

    * Simulator gives the marginal at each time step
    * Simulator gives the transition and emission potential
    * For potential, `is_conv` means whether or not it is an emission
    """

    def __init__(self, length, simulator, policy, random_seed=1):
        super().__init__(length, simulator.status_d, policy, random_seed)
        self.simulator = simulator
        self.cnt_constrained_node = 0

    def fix_initpotential(self, potential=None):
        """change the initial distribution, or first node potential

        :param potential: if None use the ground truth distribution as potential,
                        otherwise use this one, defaults to None
        :type potential: [type], optional
        """
        first_node = self.graph.get_node(f"VarNode_{0:03d}")
        if potential is None:
            potential = self.simulator.get_gt_marginal(0)
        first_node.potential = potential

    def add_constrained_node(self, probability=None):
        """overwrite add observation node, the marginal is given by the simulator

        :param probability: [description], defaults to None
        :type probability: [type], optional
        """
        prob = self.simulator.get_constrained_marginal(
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
            potential = self.simulator.get_tansition_potential()
        factornode = FactorNode(name_list, potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_branch(self, head_node=None, is_constrained=False,
                   prob=None, is_conv=False):
        """overwrite the add a pair of node behavior

        :param head_node: after which node to add a pair of factor and variable,
             defaults to None, means the last node
        :type head_node: [type], optional
        :param is_constrained: whether or not the added variable node is
            constrained or not, defaults to False
        :type is_constrained: bool, optional
        :param prob: [description], defaults to None
        :type prob: [type], optional
        :param is_conv: In HMM it means is_emission, defaults to False
        :type is_conv: bool, optional
        :return: [description]
        :rtype: [type]
        """
        if is_conv:  # emission
            return super().add_branch(head_node, True, prob, is_conv)
        return super().add_branch(head_node, False, prob, is_conv)

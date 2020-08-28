from cbp.utils import (compare_marginals, diff_max_marginals,
                       engine_loop)
from cbp.configs.base_config import baseconfig
from cbp.node import VarNode

from .coef_policy import bp_policy
from .graph_utils import itsbp_inner_loop, find_link
from .discrete_graph import DiscreteGraph


class GraphModel(DiscreteGraph):
    def __init__(self, coef_policy=bp_policy, config=baseconfig):
        super().__init__(config, coef_policy=coef_policy)

    def add_varnode(self, node):
        """ add check node type and call parent checker

        :param node: one VarNode
        :type node: VarNode
        :return: name of varnode
        :rtype: str
        """
        assert isinstance(node, VarNode)
        return super().add_varnode(node)

    def bake(self):
        super().bake()
        for node in self.nodes:
            node.cal_cnp_coef()

    def run_cnp(self):
        self.bake()
        return self.norm_product_bp()

    def run_bp(self):
        if self.coef_policy != bp_policy:  # pylint: disable=comparison-with-callable
            self.coef_policy = bp_policy
        self.bake()
        return self.itsbp()

    def norm_product_bp(self, error_fun=None):
        if error_fun is None:
            error_fun = diff_max_marginals
        self.first_belief_propagation()
        return self.engine_loop(
            max_iter=self.cfg.cnp_engine_iteration,
            engine_fun=self.parallel_message,
            tolerance=self.cfg.cnp_engine_tolerance,
            error_fun=error_fun,
            isoutput=self.cfg.verbose_engine_cnp)

    def cal_bethe(self, margin):
        """calculate bethe energy

        :param margin: node_name : margin
        :type margin: dict
        :return: KL divergence between expoert joint dist and p_graph
        :rtype: float
        """
        sum_item = []
        for node in self.nodes:
            sum_item.append(node.cal_bethe(margin[node.name]))

        return np.sum(sum_item)

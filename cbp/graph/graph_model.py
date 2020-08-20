from cbp.utils import (compare_marginals, diff_max_marginals,
                       engine_loop)
from cbp.configs.base_config import baseconfig
from cbp.node import VarNode

from .coef_policy import bp_policy
from .graph_utils import itsbp_inner_loop, find_link
from .discrete_graph import DiscreteGraph


class GraphModel(DiscreteGraph):
    def __init__(self, coef_policy=bp_policy, config=baseconfig):
        super().__init__(coef_policy=coef_policy)
        self.cfg = config
        self.itsbp_outer_cnt = 0

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

    def engine_loop(  # pylint: disable= too-many-arguments
            self,
            engine_fun,
            max_iter=5000000,
            tolerance=1e-2,
            error_fun=None,
            isoutput=False):
        if error_fun is None:
            error_fun = compare_marginals

        epsilons, step, timer = engine_loop(
            engine_fun=engine_fun,
            max_iter=max_iter,
            tolerance=tolerance,
            error_fun=error_fun,
            meassure_fun=self.export_convergence_marginals,
            isoutput=isoutput)

        return epsilons, step, timer

    def itsbp(self):
        """run sinkhorn or iterative scaling inference

        :return: [description]
        :rtype: [type]
        """
        self.first_belief_propagation()
        return self.engine_loop(self.itsbp_outer_loop,
                                tolerance=self.cfg.itsbp_outer_tolerance,
                                error_fun=diff_max_marginals,
                                isoutput=self.cfg.verbose_itsbp_outer)

    def its_next_looplink(self):
        target_node = self.leaf_nodes[self.itsbp_outer_cnt]

        next_node = self.leaf_nodes[(
            self.itsbp_outer_cnt + 1) % len(self.leaf_nodes)]

        self.itsbp_outer_cnt = self.cfg.itsbp_schedule(
            self.itsbp_outer_cnt, self.leaf_nodes)
        return target_node, find_link(target_node, next_node)

    def itsbp_outer_loop(self):
        for _ in range(len(self.leaf_nodes)):
            _, loop_link = self.its_next_looplink()
            itsbp_inner_loop(loop_link, self.cfg.verbose_node_send_msg)

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            # sendind in messages from factors
            target_var.sendin_message(self.cfg.verbose_node_send_msg)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.cfg.verbose_node_send_msg)

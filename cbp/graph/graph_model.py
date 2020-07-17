from cbp.utils import (compare_marginals, diff_max_marginals,
                       engine_loop)
from cbp.configs.base_config import baseconfig

from .base_graph import BaseGraph
from .coef_policy import bp_policy
from .graph_utils import itsbp_inner_loop, find_link


class GraphModel(BaseGraph):
    def __init__(self, silent=True, epsilon=1,
                 coef_policy=bp_policy, config=baseconfig):
        super().__init__(config=config, silent=silent,
                         epsilon=epsilon, coef_policy=coef_policy)
        self.itsbp_outer_cnt = 0

    def init_cnp_coef(self):
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

    def norm_product_bp(self, max_iter=5000000, tolerance=1e-5, error_fun=None):
        if error_fun is None:
            error_fun = diff_max_marginals
        self.init_cnp_coef()
        self.first_belief_propagation()
        return self.engine_loop(
            max_iter=max_iter,
            engine_fun=self.parallel_message,
            tolerance=tolerance,
            error_fun=error_fun,
            isoutput=False)

    def engine_loop(  # pylint: disable= too-many-arguments
            self,
            engine_fun,
            max_iter=5000000,
            tolerance=1e-2,
            error_fun=None,
            isoutput=False):
        if error_fun is None:
            error_fun = compare_marginals

        epsilons, step, _ = engine_loop(
            engine_fun=engine_fun,
            max_iter=max_iter,
            tolerance=tolerance,
            error_fun=error_fun,
            meassure_fun=self.export_convergence_marginals,
            isoutput=isoutput,
            silent=self.silent
        )

        return epsilons, step

    def itsbp(self):
        """run sinkhorn or iterative scaling inference

        :return: [description]
        :rtype: [type]
        """
        self.init_cnp_coef()
        self.first_belief_propagation()
        return self.engine_loop(self.itsbp_outer_loop,
                                tolerance=1e-4,
                                error_fun=diff_max_marginals,
                                isoutput=False)

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
            itsbp_inner_loop(loop_link, self.silent)

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            # sendind in messages from factors
            target_var.sendin_message(self.silent)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.silent)

    def two_pass(self):
        # TODO: remove!
        self.init_cnp_coef()
        self.first_belief_propagation()
        for node in self.nodes:
            node.marked = False

        for node in self.nodes:
            if len(node.connections) == 1:
                root_node = node

        self.send_from(root_node)
        self.send_out(root_node)

    def send_from(self, node):
        node.marked = True
        for cur_node in node.connected_nodes.values():
            if not cur_node.marked:
                self.send_from(cur_node)
                cur_node.send_message(node)

    def send_out(self, node):
        node.marked = False
        for cur_node in node.connected_nodes.values():
            if cur_node.marked:
                node.send_message(cur_node)
                self.send_out(cur_node)

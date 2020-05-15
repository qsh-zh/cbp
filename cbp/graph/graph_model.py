from functools import partial

from cbp.utils import (compare_marginals, diff_max_marginals,
                       engine_loop)

from .base_graph import BaseGraph
from .coef_policy import bp_policy
from .graph_utils import iterative_scaling_inner_loop, find_link


class GraphModel(BaseGraph):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        super().__init__(silent=silent, epsilon=epsilon, coef_policy=coef_policy)
        self.iterative_scaling_outer_cnt = 0

    def init_cnp_coef(self):
        for node in self.nodes:
            node.reset()
            node.cal_cnp_coef()

    # decay interface
    def belif_p(self, algo=None):
        if algo is None:
            algo = self.norm_product_bp
        elif algo == self.iterative_scaling:
            if self.coef_policy != bp_policy:
                self.coef_policy = bp_policy

        self.bake()
        return algo()

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

    def iterative_scaling(self):
        self.init_cnp_coef()
        self.first_belief_propagation()

        inner_bind = partial(self.parallel_message, False)
        self.engine_loop(inner_bind,
                         tolerance=1e-2,
                         error_fun=diff_max_marginals)

        return self.engine_loop(self.iterative_scaling_outer_loop,
                                tolerance=1e-3,
                                error_fun=diff_max_marginals,
                                isoutput=False)

    def iterative_scaling_outer_counting(self):
        self.iterative_scaling_outer_cnt += 1

        self.iterative_scaling_outer_cnt %= len(self.constrained_nodes)

    def its_next_looplink(self):
        target_node = self.constrained_nodes[self.iterative_scaling_outer_cnt]
        # target_node.sendout_message()

        next_node = self.constrained_nodes[(
            self.iterative_scaling_outer_cnt + 1) % len(self.constrained_nodes)]

        self.iterative_scaling_outer_counting()
        return target_node, find_link(target_node, next_node)

    def iterative_scaling_outer_loop(self):
        for _ in range(len(self.constrained_nodes)):
            _, loop_link = self.its_next_looplink()
            inner_fun = partial(iterative_scaling_inner_loop, loop_link)

            self.engine_loop(inner_fun,
                             tolerance=1e-2,
                             error_fun=diff_max_marginals,
                             isoutput=False)

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            # sendind in messages from factors
            target_var.sendin_message(self.silent)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.silent)

    def two_pass(self):
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

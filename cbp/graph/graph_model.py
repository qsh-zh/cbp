from functools import partial

import numpy as np
from cbp.node import FactorNode, VarNode
from cbp.utils import (Message, compare_marginals, diff_max_marginals,
                       engine_loop)
from cbp.utils.np_utils import expand_ndarray, reduction_ndarray

from .base_graph import BaseGraph
from .coef_policy import *


class GraphModel(BaseGraph):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        super().__init__(silent=silent, epsilon=epsilon, coef_policy=coef_policy)

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
        self.init_cnp_coef()
        self.first_belief_propagation()
        return self.engine_loop(
            engine_fun=self.parallel_message,
            tolerance=1e-4,
            error_fun=diff_max_marginals,
            isoutput=False)

    def engine_loop(
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
        try:
            self.iterative_scaling_outer_cnt += 1
        except AttributeError:
            self.iterative_scaling_outer_cnt = 0

        self.iterative_scaling_outer_cnt %= len(self.constrained_nodes)

    def its_next_looplink(self):
        self.iterative_scaling_outer_counting()
        target_node = self.constrained_nodes[self.iterative_scaling_outer_cnt]
        # target_node.sendout_message()

        next_node = self.constrained_nodes[(
            self.iterative_scaling_outer_cnt + 1) % len(self.constrained_nodes)]

        return target_node, self.find_link(target_node, next_node)

    def iterative_scaling_outer_loop(self):
        for _ in range(len(self.constrained_nodes)):
            cur_node, loop_link = self.its_next_looplink()
            inner_fun = partial(self.iterative_scaling_inner_loop, loop_link)

            self.engine_loop(inner_fun,
                             tolerance=1e-2,
                             error_fun=diff_max_marginals,
                             isoutput=False)

    def iterative_scaling_inner_loop(self, loop_link):
        if len(loop_link) == 2:
            return

        for sender, reciever in zip(loop_link[0:-1], loop_link[1:]):
            sender.send_message(reciever)

        loop_link.reverse()
        for sender, reciever in zip(loop_link[0:-1], loop_link[1:]):
            sender.send_message(reciever)

    def find_link(self, node_a, node_b):
        a_2root = self.get_node2root(node_a)
        b_2root = self.get_node2root(node_b)
        while (len(b_2root) > 1 and len(a_2root) > 1):
            if b_2root[-1] == a_2root[-1] and b_2root[-2] == a_2root[-2]:
                b_2root.pop()
                a_2root.pop()
            else:
                break
        b_2root.reverse()
        if len(b_2root) >= 2:
            return a_2root + b_2root[1:]
        else:
            return a_2root[:-1] + b_2root[:]

    def get_node2root(self, node):
        rtn = []
        tmp = node
        while True:
            rtn.append(tmp)
            tmp = tmp.parent
            if not tmp:
                break

        return rtn

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            connected_factor_names = target_var.connections
            connected_factor = [self.node_recorder[name]
                                for name in connected_factor_names]
            # sendind in messages from factors
            target_var.sendin_message(self.silent)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.silent)

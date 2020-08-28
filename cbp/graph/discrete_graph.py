from functools import partial

import numpy as np
from cbp.utils import (Message, diff_max_marginals,
                       engine_loop)
from cbp.utils.np_utils import (nd_expand, nd_multiexpand,
                                nd_reduce)
from .coef_policy import bp_policy
from .graph_utils import cal_marginal_from_tensor

from .cnp_graph import CnpGraph
from cbp.configs.base_config import baseconfig


class DiscreteGraph(CnpGraph):
    def __init__(self, config=baseconfig, coef_policy=bp_policy):
        super().__init__(config, coef_policy)

    def __pmf(self):
        """output the probability mass matrix through brutal-force methods

        :return: joint probability mass matrix
        :rtype: ndarray
        """
        varnode_names = list(self.varnode_recorder.keys())
        varnodes = list(self.varnode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        assert len(var_dim) < 32, "max number of vars for brute_force is 32 \
            (numpy matrix dim limit)"
        joint_acc = np.ones(var_dim)
        for factor in self.factornode_recorder.values():
            which_dims = [varnode_names.index(v)
                          for v in factor.connections]
            factor_acc = np.ones(var_dim)

            factor_acc = nd_multiexpand(factor.potential, var_dim, which_dims)

            joint_acc *= factor_acc

        joint_prob = joint_acc / np.sum(joint_acc)
        return joint_prob

    def exact_marginal(self):
        self.bake()
        varnodes = list(self.varnode_recorder.values())
        prob_tensor = self.__pmf()

        marginal_list = cal_marginal_from_tensor(prob_tensor, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.bfmarginal = marginal

    def __init_sinkhorn_node(self):
        varnode_names = list(self.varnode_recorder.keys())
        self.sinkhorn_node_coef = {}  # pylint: disable=attribute-defined-outside-init
        for node_name in self.constrained_names:
            node_instance = self.varnode_recorder[node_name]
            self.sinkhorn_node_coef[node_name] = {
                'index': varnode_names.index(node_name),
                'mu': node_instance.constrained_marginal,
                'u': np.ones(node_instance.rv_dim)
            }
        for node in self.varnode_recorder.values():
            node.sinkhorn = np.ones(node.rv_dim) / node.rv_dim

    def __build_big_u(self):
        varnodes = list(self.varnode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        joint_acc = np.ones(var_dim)

        for _, recoder in self.sinkhorn_node_coef.items():
            constrained_acc = nd_expand(
                recoder['u'], tuple(var_dim), recoder['index'])
            joint_acc *= constrained_acc

        return joint_acc / np.sum(joint_acc)

    # TODO this is a bug!!!!
    def sinkhorn_update(self, tilde_c):
        for _, recorder in self.sinkhorn_node_coef.items():
            big_u = self.__build_big_u()
            normalized_denominator = (big_u * tilde_c) / \
                np.sum(big_u * tilde_c)

            copy_denominator = nd_reduce(
                normalized_denominator, recorder['index'])
            copy_denominator = np.clip(copy_denominator, 1e-12, None)
            recorder['u'] = recorder['u'] * recorder['mu'] / copy_denominator

        varnodes = list(self.varnode_recorder.values())
        marginal_list = cal_marginal_from_tensor(
            normalized_denominator, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.sinkhorn = marginal

    def __check_sinkhorn(self):
        if len(self.constrained_names) == 0:
            raise RuntimeError(
                "There is no constrained nodes, use brutal force")

    def sinkhorn(self, max_iter=5000000, tolerance=1e-5):
        self.__check_sinkhorn()
        tilde_c = self.__pmf()
        self.__init_sinkhorn_node()

        sinkhorn_func = partial(self.sinkhorn_update, tilde_c)
        return engine_loop(engine_fun=sinkhorn_func,
                           max_iter=max_iter,
                           tolerance=tolerance,
                           error_fun=diff_max_marginals,
                           meassure_fun=self.export_sinkhorn,
                           isoutput=False)

    def export_sinkhorn(self):
        return {node_name: node.sinkhorn
                for node_name, node in self.varnode_recorder.items()}

from functools import partial

import numpy as np
from cbp.utils import (Message, diff_max_marginals,
                       engine_loop)
from cbp.utils.np_utils import (nd_expand, nd_multiexpand,
                                reduction_ndarray)
from .coef_policy import bp_policy
from .graph_utils import cal_marginal_from_tensor

from .base_graph import BaseGraph


class DiscreteGraph(BaseGraph):
    def __init__(self, coef_policy=bp_policy):
        super().__init__()
        self.constrained_names = []
        self.coef_policy = coef_policy

    def add_varnode(self, node):
        node_name = super().add_varnode(node)
        if node.isconstrained:
            self.constrained_names.append(node_name)
        return node_name

    def _delete_node_recorder(self, node):
        if node.name in self.constrained_names:
            self.constrained_names.remove(node.name)
        return super()._delete_node_recorder(node)

    def set_node(self, node_name, potential=None, isconstrained=None):
        """change node property
        1. check whether or not in recorder
        2. change potential easily[this may a duplicate function]
        3. change isconstrained if possible, delete from recorder

        :param node_name: [description]
        :type node_name: [type]
        :param potential: [description], defaults to None
        :type potential: [type], optional
        :param isconstrained: [description], defaults to None
        :type isconstrained: [type], optional
        :raises RuntimeError: [description]
        """
        if node_name not in self.node_recorder:
            raise RuntimeError
        node = self.node_recorder[node_name]
        if potential is not None:
            node.potential = potential
        if isconstrained is not None:
            if node.isconstrained != isconstrained:
                node.isconstrained = isconstrained
                if isconstrained:
                    self.constrained_names.append(node_name)
                else:
                    self.constrained_names.remove(node_name)

    def first_belief_propagation(self):
        for node in self.nodes:
            for recipient_name in node.connections:
                recipient = self.node_recorder[recipient_name]
                if recipient.name not in node.message_inbox:
                    val = node.make_init_message(recipient_name)
                    message = Message(node, val)
                    self.node_recorder[recipient_name].store_message(message)

    def bake(self):
        super().bake()
        self.__cal_node_coef(self.get_root())

    def __cal_node_coef(self, node):
        for _node in self.nodes:
            setattr(_node, 'is_traversed', False)
        self.__traverse_node_coef(node)

    def __traverse_node_coef(self, node):
        node.is_traversed = True
        for item in node.connections:
            if not self.node_recorder[item].is_traversed:
                self.__traverse_node_coef(self.node_recorder[item])

        node.auto_coef(self.node_recorder, self.coef_policy)
        node.is_traversed = False

    def copy_bp_initialization(self, another_graph):
        """copy message setup from the another graph has same structure

        :param another_graph: another graph which close to the optimal point
        :type another_graph: BaseGraph
        """
        # TODO: copy safety!!!
        for node in self.nodes:
            if node.name in another_graph.node_recorder:
                another_node = another_graph.node_recorder[node.name]
                node.message_inbox = another_node.message_inbox
                node.latest_message = another_node.latest_message
            else:
                raise f"{node.name} not in this graph"

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

            copy_denominator = reduction_ndarray(
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

    def tree_bp(self):
        """run classical belief propagation on a tree graph, only need forward
        and backward

            * add attr: is_send_forward: begin send forward false, after forward
             before backward true, after backward false
        :raises RuntimeError: Only works for the tree graph, loopy graph does
        not work, root node not decided
        """
        assert len(self.constrained_names) == 0
        self.bake()
        self.first_belief_propagation()
        for node in self.nodes:
            setattr(node, 'is_send_forward', False)

        tree_root = self.get_root()

        self._send_forward(tree_root)
        self._send_backward(tree_root)

    def _send_forward(self, node):
        node.is_send_forward = True
        for cur_node in node.connected_nodes.values():
            if not cur_node.is_send_forward:
                self._send_forward(cur_node)
                cur_node.send_message(node)

    def _send_backward(self, node):
        node.is_send_forward = False
        for cur_node in node.connected_nodes.values():
            if cur_node.is_send_forward:
                node.send_message(cur_node)
                self._send_backward(cur_node)

    def export_sinkhorn(self):
        return {node_name: node.sinkhorn
                for node_name, node in self.varnode_recorder.items()}

    def export_marginals(self):
        """export the marginal for variable nodes

        :return: {node.key:node.marginal}
        :rtype: dict
        """
        return {
            n.name: n.marginal() for n in self.varnode_recorder.values()
        }

    def export_convergence_marginals(self):
        """export the marginal for variable nodes and factor nodes

        :return: {node.key:node.marginal}
        :rtype: dict
        """
        return {n.name: n.marginal() for n in self.nodes}

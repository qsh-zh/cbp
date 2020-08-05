import numpy as np
from cbp.utils.np_utils import nd_expand

from .discrete_node import DiscreteNode


class VarNode(DiscreteNode):
    """Variable Node in Factor graph

        Add new attr:
        * ``isconstrained`` Fixed marginal or not
        * ``hat_c_i`` See Norm-Product paper
    """

    def __init__(self, rv_dim, potential=None,
                 constrained_marginal=None, node_coef=0):
        self.rv_dim = rv_dim
        self.hat_c_i = None
        super().__init__(node_coef, potential)

        if constrained_marginal is None:
            self.isconstrained = False
        else:
            assert constrained_marginal.shape[0] == rv_dim
            assert abs(np.sum(constrained_marginal) - 1) < 1e-6
            self.isconstrained = True
            constrained_marginal = np.clip(constrained_marginal, 1e-12, None)
        self.constrained_marginal = constrained_marginal

    def _check_potential(self, potential):
        """deal None case
        """
        if potential is None:
            return np.ones([self.rv_dim])
        assert potential.shape[0] == self.rv_dim
        return np.clip(potential, 1e-12, None)

    def auto_coef(self, node_map, assign_policy=None):
        super().auto_coef(node_map, assign_policy)

        sum_i_alpha, unset_edge = self.__sum_neighbor_coef()
        if unset_edge:
            new_i_alpha = self.node_coef - \
                (1 - len(self.connections)) - sum_i_alpha
            self.connected_nodes[unset_edge].set_i_alpha(self.name, new_i_alpha)

    def __sum_neighbor_coef(self):
        sum_i_alpha = 0
        uninit_neighbor = None
        for node in self.connected_nodes.values():
            i_alpha = node.get_i_alpha(self.name)
            if i_alpha is not None:
                sum_i_alpha += i_alpha
            else:
                uninit_neighbor = node.name
        return sum_i_alpha, uninit_neighbor

    def cal_cnp_coef(self):
        self.coef_ready = True

        self.hat_c_i = self.node_coef
        for item in self.connections:
            self.hat_c_i += self.connected_nodes[item].node_coef

    def _make_message_first_term(self, recipient_node):
        recipient_index_in_var = self.search_msg_index(self.latest_message,
                                                       recipient_node.name)
        hat_c_ialpha = recipient_node.get_hat_c_ialpha(self.name)
        c_alpha = recipient_node.node_coef
        vals = [message.val for message in self.latest_message]

        with np.errstate(divide='raise'):
            if self.isconstrained:
                log_numerator = self.epsilon * np.log(self.constrained_marginal)
            else:
                potential_part = 1.0 / self.hat_c_i * np.log(self.potential)
                message_part = 1.0 / self.hat_c_i * \
                    np.log(np.clip(np.prod(vals, axis=0), 1e-12, None))
                log_numerator = potential_part + message_part
            clip_base = np.clip(vals[recipient_index_in_var], 1e-12, None)
            log_denominator = 1.0 / hat_c_ialpha * np.log(clip_base)

            log_base = c_alpha * (log_numerator - log_denominator)
            return np.exp(log_base)

    def make_message_bp(self, recipient_node):
        assert self.coef_ready,\
            f"{self.name} need to cal_cnp_coef by graph firstly"
        # first_term.shape equals (self.rv_dim,)
        first_term = self._make_message_first_term(recipient_node)
        assert first_term.shape[0] == self.rv_dim
        # second_term shape equals shape of recipient_node
        second_term = recipient_node.get_varnode_extra_term(self.name)
        assert second_term.shape == self.connected_nodes[recipient_node.name].potential.shape

        var_index_in_recipient = recipient_node.search_node_index(self.name)

        expanded_first_term = nd_expand(
            first_term, second_term.shape, var_index_in_recipient)

        return np.multiply(expanded_first_term, second_term)

    def make_message(self, recipient_node):
        return self.make_message_bp(recipient_node)

    def cal_bethe(self, margin):
        clip_margin = np.clip(margin, 1e-12, None)
        log_margin = np.log(clip_margin)
        entropy_term = -(self.node_degree - 1) * np.sum(margin * log_margin)
        clip_potential = np.clip(self.potential, 1e-12, None)
        potential_term = -np.sum(margin * np.log(clip_potential))
        return potential_term + entropy_term

    def marginal(self):
        if self.isconstrained:
            return self.constrained_marginal
        if self.message_inbox:
            vals = [message.val for message in self.latest_message]
            vals_prod = np.prod(vals, axis=0)
            prod = self.potential * vals_prod
            belief = np.power(prod, 1.0 / self.hat_c_i)
            return belief / np.sum(belief)

        return np.ones(self.rv_dim) / self.rv_dim

    def __eq__(self, value):
        if isinstance(value, type(self)):
            flag = []
            flag.append(self.isconstrained == value.isconstrained)
            flag.append(np.array_equal(self.constrained_marginal,
                                       value.constrained_marginal))
            if np.sum(flag) == len(flag):
                return super().__eq__(value)

        return False

    def plot(self, graph):
        if self.isconstrained:
            graph.add_node(self.name, color='red', style='filled')
        else:
            super().plot(graph)

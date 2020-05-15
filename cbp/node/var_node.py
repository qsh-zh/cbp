import json

import numpy as np
from cbp.utils.np_utils import expand_ndarray

from .base_node import BaseNode


class VarNode(BaseNode):
    """Variable Node in Factor graph

        Add new attr:
        * ``isconstrained`` Fixed marginal or not
        * ``hat_c_i`` See Norm-Product paper
    """

    def __init__(self, rv_dim, potential=None,
                 constrained_marginal=None, node_coef=1):
        super().__init__(node_coef, potential)

        self.rv_dim = rv_dim
        self.hat_c_i = None

        if self.potential is None:
            self.potential = np.ones([rv_dim])
        assert self.potential.shape[0] == rv_dim

        if constrained_marginal is None:
            self.isconstrained = False
        else:
            assert constrained_marginal.shape[0] == rv_dim
            assert abs(np.sum(constrained_marginal) - 1) < 1e-6
            self.isconstrained = True
        self.constrained_marginal = constrained_marginal

    def auto_coef(self, node_map, assign_policy=None):
        super().auto_coef(node_map, assign_policy)

        sum_i_alpha = 0
        unset_edge = None
        for item in self.connected_nodes.values():
            i_alpha = item.get_i_alpha(self.name)
            if i_alpha is not None:
                sum_i_alpha += i_alpha
            else:
                unset_edge = item.name
        if unset_edge:
            new_i_alpha = self.node_coef - \
                (1 - len(self.connections)) - sum_i_alpha
            self.connected_nodes[unset_edge].set_i_alpha(self.name, new_i_alpha)

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
        if self.isconstrained:
            log_numerator = self.epsilon * np.log(self.constrained_marginal)
        else:
            potential_part = 1.0 / self.hat_c_i * np.log(self.potential)
            message_part = 1.0 / self.hat_c_i * np.log(np.prod(vals, axis=0))
            log_numerator = potential_part + message_part
        clip_base = vals[recipient_index_in_var]
        log_denominator = 1.0 / hat_c_ialpha * np.log(clip_base)

        log_base = c_alpha * (log_numerator - log_denominator)
        log_base = log_base - np.max(np.nan_to_num(log_base))
        return np.exp(log_base)

    def make_message_bp(self, recipient_node):
        assert self.coef_ready, f"{self.name} need to cal_cnp_coef by graph firstly"
        # first_term.shape equals (self.rv_dim,)
        first_term = self._make_message_first_term(recipient_node)
        assert first_term.shape[0] == self.rv_dim
        # second_term shape equals shape of recipient_node
        second_term = recipient_node.get_varnode_extra_term(self.name)
        assert second_term.shape == self.connected_nodes[recipient_node.name].potential.shape

        var_index_in_recipient = recipient_node.search_node_index(self.name)

        expanded_first_term = expand_ndarray(
            first_term, second_term.shape, var_index_in_recipient)

        return np.multiply(expanded_first_term, second_term)

    def make_message(self, recipient_node):
        return self.make_message_bp(recipient_node)

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

    def to_json(self, separators=(',', ':'), indent=4):
        return json.dumps({
            'class': 'VarNode',
            'name': self.name,
            'potential': self.potential.tolist(),
            'node_coef': self.node_coef,
            'constraine_marginal': self.constrained_marginal.tolist() if self.isconstrained else None,
            'connections': self.connections
        }, separators=separators, indent=indent)

    @classmethod
    def from_json(cls, json_file):
        d_context = json.loads(json_file)

        if d_context['class'] != 'VarNode':
            raise IOError(
                f"Need a VarNode class json to construct VarNode instead of {d_context['class']}")

        potential = d_context['potential']
        coef = d_context['node_coef']
        constraine_marginal = d_context['constraine_marginal']
        if isinstance(constraine_marginal, list):
            constraine_marginal = np.asarray(constraine_marginal)
        node = cls(
            len(potential),
            np.asarray(potential),
            constraine_marginal,
            node_coef=coef)
        node.format_name(d_context['name'])
        for factor_node in d_context['connections']:
            node.register_connection(factor_node)
        return node

    def __eq__(self, value):
        if isinstance(value, VarNode):
            flag = []
            flag.append(self.name == value.name)
            flag.append(np.array_equal(self.potential, value.potential))
            flag.append(np.array_equal(self.node_coef, value.node_coef))
            flag.append(self.isconstrained == value.isconstrained)
            flag.append(
                np.array_equal(
                    self.constrained_marginal,
                    value.constrained_marginal))
            flag.append(self.node_degree == value.node_degree)
            flag.append(self.connections == value.connections)
            if np.sum(flag) == len(flag):
                return True

        return False

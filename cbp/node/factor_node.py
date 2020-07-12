import numpy as np
from cbp.utils.np_utils import nd_expand

from .base_node import BaseNode


class FactorNode(BaseNode):
    """Factor Node in factor graph

      Add new attr:
        * ``isconstrained`` Fixed marginal or not
        * ``hat_c_ialpha`` See Norm-Product paper
        * ``last_innerparenthese_msg`` See Norm-Product paper
    """

    def __init__(self, connections, potential, coef=1):
        super().__init__(coef, potential)
        self.connections = connections
        self.last_innerparenthese_msg = {}
        self.hat_c_ialpha = {}

        self.i_alpha = {}

        num_connectednode = []
        for item in self.connections:
            self.i_alpha[item] = None
            num_connectednode.append(int(item[8:]))

        if any(i > j for i, j in zip(num_connectednode, num_connectednode[1:])):
            raise RuntimeError('Set the connection of factor in order')

    def check_before_run(self, node_map):
        super().check_before_run(node_map)
        self.check_potential(node_map)

    def check_potential(self, node_map):
        for i, varnode_name in enumerate(self.connections):
            varnode = node_map[varnode_name]
            assert self.potential.shape[i] == varnode.rv_dim, \
                f"Dimention mismatch! At {i:02d} axis in Factor:{self.name} \
                    rv_dim:{varnode.rv_dim:02d}, \
                    potential: {self.potential.shape[i]}"
            self.last_innerparenthese_msg[varnode_name] = np.ones(
                self.potential.shape)

    def _check_potential(self, potential):
        return potential / np.sum(potential)

    def auto_coef(self, node_map, assign_policy=None):
        super().auto_coef(node_map, assign_policy)

        sum_i_alpha = 0
        unset_edge = None
        for connected_var in self.connections:
            i_alpha = self.get_i_alpha(connected_var)
            if i_alpha is not None:
                sum_i_alpha += i_alpha
            else:
                unset_edge = connected_var
        if unset_edge:
            new_i_alpha = 1 - self.node_coef - sum_i_alpha
            self.set_i_alpha(unset_edge, new_i_alpha)

    def get_i_alpha(self, connection_name):
        return self.i_alpha[connection_name]

    def set_i_alpha(self, connection_name, value):
        self.i_alpha[connection_name] = value

    def cal_cnp_coef(self):
        self.coef_ready = True

        for item in self.connections:
            hat_c_ialpha = self.node_coef + self.i_alpha[item]
            assert hat_c_ialpha != 0
            self.hat_c_ialpha[item] = hat_c_ialpha

    def get_hat_c_ialpha(self, node_name):
        if self.coef_ready:
            return self.hat_c_ialpha[node_name]
        return None

    def get_varnode_extra_term(self, node_name):
        """
        Norm-Product Belief Propagation, n_{i -> alpha} second term
        This term is always 1 in stardard bp
        """
        if node_name not in self.last_innerparenthese_msg:
            raise RuntimeError(
                f"{node_name} do not have previous msg sent by {self.name}")

        if abs(self.i_alpha[node_name]) < 1e-5 and self.node_coef == 1:
            return np.ones_like(
                self.last_innerparenthese_msg[node_name])

        # TODO when the a^x, a = 0, it has some problem
        coef_exp = -1.0 * \
            self.i_alpha[node_name] / self.hat_c_ialpha[node_name]
        base = self.last_innerparenthese_msg[node_name]
        value = np.power(base, coef_exp)
        return value

    def make_message(self, recipient_node):
        assert recipient_node.name in self.connections
        if len(self.connections) == 1:
            self.last_innerparenthese_msg[recipient_node.name] = self.potential
            return self.summation(self.potential, recipient_node)

        product_out = self.cal_inner_parentheses(recipient_node)

        with np.errstate(divide='raise'):
            hat_c_ialpha = self.hat_c_ialpha[recipient_node.name]
            log_media = 1.0 / hat_c_ialpha * \
                np.log(np.clip(product_out, 1e-12, 10))
            product_out_power = np.exp(
                log_media - np.max(np.nan_to_num(log_media)))
            return np.power(
                self.summation(
                    product_out_power,
                    recipient_node),
                hat_c_ialpha)

    def cal_bethe(self, margin):
        clip_potential = np.clip(self.potential, 1e-12, np.inf)
        return np.sum(margin * np.log(margin / clip_potential))

    def marginal(self):
        message_val = np.array([message.val for message in self.latest_message])
        prod_messages = np.prod(message_val, axis=0)
        product_out = np.multiply(self.potential, prod_messages)
        unormalized = np.power(
            product_out / np.sum(product_out),
            1.0 / self.node_coef)
        return unormalized

    def cal_inner_parentheses(self, recipient_node):
        latest_message = self.latest_message
        filtered_message = [message for message in latest_message
                            if not message.sender.name == recipient_node.name]

        message_val = np.array([message.val for message in filtered_message])

        prod_messages = np.prod(message_val, axis=0)

        prod_messages /= np.mean(prod_messages)

        product_out = np.multiply(self.potential, prod_messages)
        self.last_innerparenthese_msg[recipient_node.name] = product_out
        return product_out

    def store_message(self, message):
        assert message.val.shape == self.potential.shape, \
            f"From {message.sender.name} to {self.name} shape mismatch, \
                expected {self.potential.shape}, received {message.val.shape}"
        super().store_message(message)

    def reformat_message(self, message):
        potential_dims = self.potential.shape
        states = message.val
        which_dim = self.connections.index(message.sender.name)

        return nd_expand(states, potential_dims, which_dim)

    def summation(self, potential, node):
        potential_dim = potential.shape
        node_index = self.connections.index(node.name)
        assert potential_dim[node_index] == node.rv_dim
        return potential.sum(
            tuple(j for j in range(potential.ndim) if j != node_index))

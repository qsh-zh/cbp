from .msg_node import MsgNode
import cbp.utils.np_utils as npu
import numpy as np


def construct_potential(list_var, list_factor):
    dims = tuple(node.rv_dim for node in list_var)
    names = [node.name for node in list_var]
    potential = np.ones(dims)
    for i, var in enumerate(list_var):
        potential *= npu.nd_expand(var.potential, dims, i)
    for factor in list_factor:
        first = names.index(factor.connections[0])
        second = names.index(factor.connections[1])
        potential *= npu.nd_multiexpand(factor.potential, dims, [first, second])
    return potential


class MOTNode(MsgNode):
    def __init__(self, list_var, list_factor=[]):
        self.list_var = list_var
        self.list_name = [node.name for node in list_var]
        self.rv_dim = len(list_var)
        self.list_factor = list_factor
        self.mot_name = self.extract_name()
        super().__init__(construct_potential(list_var, list_factor))

    def extract_name(self):
        connected_vars = []
        for node_name in self.list_name:
            connected_vars.append(int(node_name[-3:]))
        return ','.join(map(str, connected_vars))

    def make_init_message(self, recipient_node_name):
        recipient_node = self.connected_nodes[recipient_node_name]
        message_dim = recipient_node.potential.shape
        return np.ones(message_dim)

    def _check_potential(self, potential):
        return potential

    def marginal(self):
        unnormalized = self.prodmsg()
        marginal = unnormalized / np.sum(unnormalized)
        setattr(self, 'cached_marginal', marginal)
        return marginal

    def marginal_dims(self, str_dims):
        dims = [self.list_name.index(name) for name in str_dims]
        if not hasattr(self, 'cached_marginal'):
            marginal = self.marginal()
        else:
            marginal = self.cached_marginal

        return npu.nd_multireduce(marginal, dims)

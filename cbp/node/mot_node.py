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
    def __init__(self, list_var, list_factor):
        self.list_var = list_var
        self.list_name = [node.name for node in list_var]
        self.list_factor = list_factor
        super().__init__(construct_potential(list_var, list_factor))

    def idx_dims(self, node):
        return [self.list_name.index(name) for name in node.list_name]

    def make_init_message(self, recipient_node_name):
        recipient_node = self.connected_nodes[recipient_node_name]
        message_dim = recipient_node.potential.shape
        return np.ones(message_dim)

    def _check_potential(self, potential):
        return potential

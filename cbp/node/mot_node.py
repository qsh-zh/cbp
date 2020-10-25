import numpy as np

import cbp.utils.np_utils as npu

from .msg_node import MsgNode

from typing import List


def construct_potential(list_var, list_factor):
    """construct the potential of mot node

    :param list_var: container of varnode
    :param list_factor: container of factor node
    """
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


class MOTNode(MsgNode):  # pylint: disable= abstract-method
    """node in mot graph, junction graph
    """

    def __init__(self, list_var, list_factor=None, potential=None):
        """
        * ``rv_dim`` int: number of varnodes in the mot_node

        """
        self.list_var = list_var
        self.list_varname = [node.name for node in list_var]
        self.rv_dim = len(list_var)
        self.list_factor = list_factor if list_factor is not None else []
        # TODO: check whether or not do we need the list_factor?
        self.connected_varname = self.__parse_varname()
        node_potential = construct_potential(
            list_var, self.list_factor) if potential is None else potential
        super().__init__(node_potential)
        self.cached_marginal = None

    def __parse_varname(self):
        connected_vars = []
        for node_name in self.list_varname:
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

    def margin_vars(self, name_vars: List[str]):
        """calculate from joint vars to marginal of requested vars

        :param name_vars: [description]
        :type name_vars: List
        :return: [description]
        :rtype: [type]
        """
        dims = [self.list_varname.index(name) for name in name_vars]
        if self.cached_marginal is None:
            marginal = self.marginal()
        else:
            marginal = self.cached_marginal

        return npu.nd_multireduce(marginal, dims)

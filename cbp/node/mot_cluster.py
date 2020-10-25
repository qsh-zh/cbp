import numpy as np

import cbp.utils.np_utils as npu

from .mot_node import MOTNode


class MOTCluster(MOTNode):
    """Cluster node in the mot graph or junction graph.

    - connection -> which seperators to connect
    - isconstrained -> whether or not the cluster is from a constrained VarNode

    We treat a contrained varnode as a cluster in the graph.
    TODO: Do we need to consider the fixed factor node?
    """

    def __init__(self, connections, list_var, list_factor=None, potential=None):
        super().__init__(list_var, list_factor, potential)
        if len(list_var) == 1 and list_var[0].isconstrained:
            self.isconstrained = True
            self.constrained_marginal = list_var[0].constrained_marginal
        else:
            self.isconstrained = False
        self.connections = connections

    def idx_dims(self, node):
        return [self.list_varname.index(name) for name in node.list_varname]

    def make_message(self, recipient_node):
        if self.isconstrained:
            return self.constrained_marginal / self.latest_message[0].val

        product_out = self.prod2node(recipient_node)
        multi_idx = self.idx_dims(recipient_node)
        return npu.nd_multireduce(product_out, multi_idx)

    def marginal(self):
        if self.isconstrained:
            setattr(self, 'cached_marginal', self.constrained_marginal)
            return self.constrained_marginal

        return super().marginal()

    def plot(self, graph):
        node_label = '\n'.join([self.name[3:], self.connected_varname])
        if self.isconstrained:
            graph.add_node(
                self.name,
                color='green',
                style='filled',
                label=node_label)
        else:
            graph.add_node(
                self.name,
                color='green',
                style='bold',
                label=node_label)

        for node in self.connected_nodes.values():
            graph.add_edge(self.name, node.name, color='black')

    def minimization(self, marginal=None):
        if self.isconstrained:
            return 0
        marginal = self.marginal() if marginal is None else marginal
        clip_potential = np.clip(self.potential, 1e-12, None)
        return np.sum(marginal * np.log(clip_potential))
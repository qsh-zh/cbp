from .mot_node import MOTNode
import cbp.utils.np_utils as npu


class MOTCluster(MOTNode):
    def __init__(self, connections, list_var, list_factor=[]):
        super().__init__(list_var, list_factor)
        if len(list_var) == 1 and list_var[0].isconstrained:
            self.isconstrained = True
            self.constrained_marginal = list_var[0].constrained_marginal
        else:
            self.isconstrained = False
        self.connections = connections

    def idx_dims(self, node):
        return [self.list_name.index(name) for name in node.list_name]

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
        node_label = '\n'.join([self.name[3:], self.mot_name])
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

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

    def make_message(self, recipient_node):
        if self.isconstrained:
            return self.constrained_marginal / self.latest_message[0]

        product_out = self.prod2node(recipient_node)
        multi_idx = self.idx_dims(recipient_node)
        return npu.nd_multireduce(product_out, multi_idx)

    def plot(self, graph):
        if self.isconstrained:
            graph.add_node(
                self.name,
                color='red',
                style='filled',
                label=self.mot_name)
        else:
            graph.add_node(self.name, color='green', label=self.mot_name)

        for node in self.connected_nodes.values():
            graph.add_edge(self.name, node.name, color='black')

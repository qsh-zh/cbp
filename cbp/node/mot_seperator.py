from .mot_node import MOTNode
import cbp.utils.np_utils as npu


class MOTSeperator(MOTNode):

    def idx_dims(self, node):
        return [node.list_name.index(name) for name in self.list_name]

    def make_message(self, recipient_node):
        product_out = self.prod2node(recipient_node)
        multi_idx = self.idx_dims(recipient_node)
        return npu.nd_multiexpand(
            product_out,
            recipient_node.potential.shape,
            multi_idx)

    def plot(self, graph):
        node_label = '\n'.join([self.name[3:], self.mot_name])
        graph.add_node(
            self.name,
            color='blue',
            style='bold',
            shape='box',
            label=node_label)

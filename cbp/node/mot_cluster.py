from .mot_node import MOTNode
import cbp.utils.np_utils as npu


class MOTCluster(MOTNode):
    def make_message(self, recipient_node):
        product_out = self.prod2node(recipient_node)
        multi_idx = self.idx_dims(recipient_node)
        return npu.nd_multireduce(product_out, multi_idx)

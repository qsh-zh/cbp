from .msg_graph import MsgGraph


class CnpGraph(MsgGraph):
    def __init__(self, config, coef_policy):
        super().__init__(config)
        self.coef_policy = coef_policy

    def bake(self):
        super().bake()
        self.__cal_node_coef(self.get_root())

    def __cal_node_coef(self, node):
        for _node in self.nodes:
            setattr(_node, 'is_traversed', False)
        self.__traverse_node_coef(node)

    def __traverse_node_coef(self, node):
        node.is_traversed = True
        for item in node.connections:
            if not self.node_recorder[item].is_traversed:
                self.__traverse_node_coef(self.node_recorder[item])

        node.auto_coef(self.node_recorder, self.coef_policy)
        node.is_traversed = False

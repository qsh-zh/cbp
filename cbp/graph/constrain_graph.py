from .base_graph import BaseGraph


class ConstrainGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.constrained_names = []

    def add_varnode(self, node):
        node_name = super().add_varnode(node)
        if node.isconstrained:
            self.constrained_names.append(node_name)
        return node_name

    def _delete_node_recorder(self, node):
        if node.name in self.constrained_names:
            self.constrained_names.remove(node.name)
        return super()._delete_node_recorder(node)

    def set_node(self, node_name, potential=None, isconstrained=None):
        """change node property
        1. check whether or not in recorder
        2. change potential easily[this may a duplicate function]
        3. change isconstrained if possible, delete from recorder

        :param node_name: [description]
        :type node_name: [type]
        :param potential: [description], defaults to None
        :type potential: [type], optional
        :param isconstrained: [description], defaults to None
        :type isconstrained: [type], optional
        :raises RuntimeError: [description]
        """
        if node_name not in self.node_recorder:
            raise RuntimeError
        node = self.node_recorder[node_name]
        if potential is not None:
            node.potential = potential
        if isconstrained is not None:
            if node.isconstrained != isconstrained:
                node.isconstrained = isconstrained
                if isconstrained:
                    self.constrained_names.append(node_name)
                else:
                    self.constrained_names.remove(node_name)

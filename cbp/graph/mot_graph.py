from .base_graph import BaseGraph


class MOTGraph(BaseGraph):
    def ___init__(self):
        super().__init__()

    def add_cluster(self, node):
        self.add_factornode(node)

    def add_seperator(self, node):
        self.add_varnode(node)

    def delete_node(self):
        pass

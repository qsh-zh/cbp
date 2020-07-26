from cbp.node import VarNode

from .base_graph import BaseGraph
from .graph_model import GraphModel


class GOGraph(BaseGraph):
    def __init__(self):
        self.discrete_graph = None
        super().__init__()

    def bake(self):
        self.discrete_graph = GraphModel()
        for node in self.varnode_recorder:
            if isinstance(node, VarNode):
                self.discrete_graph.add_varnode(
                    VarNode(node.rv_dim, node.potential,
                            node.constrained_marginal))
            else:
                self.discrete_graph.add_varnode(node.discrete())

    def run_bp(self):
        return self.discrete_graph.run_bp()

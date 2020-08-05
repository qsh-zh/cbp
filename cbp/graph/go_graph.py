from cbp.node import VarNode, FactorNode

from .base_graph import BaseGraph
from .graph_model import GraphModel


class GOGraph(BaseGraph):
    """graph with continuous observation model.
    The observations node, which only exists as leaf nodes.
    Only task: construct a discrete graph with same probability
    """

    def __init__(self):
        self.discrete_graph = None
        super().__init__()

    def bake(self):
        super().bake()
        for node in self.nodes:
            node.register_nodes(self.node_recorder)

        self.construct_discrete_graph()

    def construct_discrete_graph(self):
        self.discrete_graph = GraphModel()
        self._discrete_var()
        self._discrete_factor()
        self.discrete_graph.bake()

    def _discrete_var(self):
        for node in self.varnode_recorder.values():
            if isinstance(node, VarNode):
                self.discrete_graph.add_varnode(
                    VarNode(node.rv_dim, node.potential,
                            node.constrained_marginal))
            else:
                self.discrete_graph.add_varnode(node.discrete())

    def _discrete_factor(self):
        for node in self.factornode_recorder.values():
            if isinstance(node, FactorNode):
                self.discrete_graph.add_factornode(
                    FactorNode(node.connections, node.potential)
                )
            else:
                self.discrete_graph.add_factornode(node.discrete())

    def run_bp(self):
        self.bake()
        return self.discrete_graph.run_bp()

import numpy as np
from scipy.stats import norm

from .base_node import BaseNode
from .factor_node import FactorNode
from .go_var_node import GOVarNode


class GOFactorNode(BaseNode):
    def __init__(self, connections, loc, scale):
        assert loc.ndim == 1
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale
        super().__init__()
        self.connections = connections

    def discrete_var(self, varnode):
        assert isinstance(varnode, GOVarNode)
        potenital = []
        for loc, scale in zip(self.loc, self.scale):
            potenital.append(norm.pdf(varnode.bins, loc=loc, scale=scale))
        return FactorNode(self.connections, np.array(potenital))

    def discrete(self):
        for node in self.connected_nodes.values():
            if isinstance(node, GOVarNode):
                return self.discrete_var(node)

        raise RuntimeError("go graph has no GOVarNode")

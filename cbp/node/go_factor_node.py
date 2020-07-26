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

    def discrete(self, varnode):
        assert isinstance(varnode, GOVarNode)
        potenital = []
        for loc, scale in zip(self.loc, self.scale):
            potenital.append(norm.pdf(varnode.bins, loc=loc, scale=scale))
        return FactorNode(self.connections, np.array(potenital))

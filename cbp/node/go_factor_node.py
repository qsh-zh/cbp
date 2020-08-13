import numpy as np
from scipy.stats import norm, multivariate_normal

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

    def _discrete_var(self, varnode):
        assert isinstance(varnode, GOVarNode)
        potenital = []
        pdf_func = multivariate_normal if varnode.is_multi else norm
        for loc, scale in zip(self.loc, self.scale):
            potenital.append(pdf_func(loc, scale).pdf(varnode.bins))
            # potenital.append(norm.pdf(varnode.bins, loc=loc, scale=scale))
        return FactorNode(self.connections, np.array(potenital))

    def discrete(self):
        for node in self.connected_nodes.values():
            if isinstance(node, GOVarNode):
                return self._discrete_var(node)

        raise RuntimeError("go factor has no GOVarNode")

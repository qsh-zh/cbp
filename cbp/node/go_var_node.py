import numpy as np

from .base_node import BaseNode
from .var_node import VarNode


class GOVarNode(BaseNode):
    def __init__(self, bins):
        self.bins = bins
        super().__init__()

    def discrete(self):
        num_sample = len(self.bins)
        return VarNode(num_sample, np.ones(num_sample) / num_sample)

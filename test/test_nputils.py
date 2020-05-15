import unittest

import numpy as np
from cbp.utils.np_utils import expand_ndarray


class TestNpUtils(unittest.TestCase):
    def test_expand_ndarray(self):
        inputdata = [1, 2]
        target_shape = (1, 2, 3)
        expand_dim = 1
        output = expand_ndarray(inputdata, target_shape, expand_dim)
        target = np.array([
            [1, 1, 1],
            [2, 2, 2]
        ])
        equal = np.isclose(output, target)
        self.assertTrue(equal.all())

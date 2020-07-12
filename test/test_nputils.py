import unittest

import numpy as np
from cbp.utils.np_utils import nd_expand, nd_multiexpand


class TestNpUtils(unittest.TestCase):
    def test_nd_expand(self):
        inputdata = np.array([1, 2])
        target_shape = (1, 2, 3)
        expand_dim = 1
        output = nd_expand(inputdata, target_shape, expand_dim)
        target = np.array([
            [1, 1, 1],
            [2, 2, 2]
        ])
        equal = np.isclose(output, target)
        self.assertTrue(equal.all())

    def test_nd_multiexpand(self):
        in_ = np.array([[1, 2]])
        out_shape = (1, 2, 3)
        dims = (0, 1)
        out_ = nd_multiexpand(in_, out_shape, dims)
        target = np.array([
            [1, 1, 1],
            [2, 2, 2]
        ])
        equal = np.isclose(out_, target)
        self.assertTrue(equal.all())

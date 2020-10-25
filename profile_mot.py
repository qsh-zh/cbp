import numpy as np
from numpy.random import RandomState
import ot
from cbp.node import MOTCluster, MOTSeperator, VarNode
from cbp.graph import MOTGraph
from cbp.configs.base_config import BaseConfig

"""
graph structure
0    1
x----x
|    |
x----x
  \  |
x----x

1, 5 are constrained
"""


class SetUpGraph:
    def __init__(self) -> None:
        self.node_dim = 100
        self.num_node = 8
        self.varnode = {}
        cfg = BaseConfig(verbose_itsbp_outer=True,
                         itsbp_outer_tolerance=1e-50,
                         verbose_itsbp_link=False,
                         itsbp_outer_iteration=10
                         )
        self.mot = MOTGraph(cfg)
        self.rng = RandomState(1)
        self.constrained = [1, 5]
        self.cluster2var = [
            [0, 1, 2, 3],
            [2, 3, 5],
            [4, 5]
        ] + [[item] for item in self.constrained]
        self.sep2var = [
            [2, 3],
            [5]
        ] + [[item] for item in self.constrained]
        self.cluster2sep = [
            [0, 2],
            [0, 1, 3],
            [1],
            [2],
            [3]
        ]

    def setup(self):
        self.setup_var()
        self.setup_sep()
        self.setup_cluster()

    def list2seq(self, cur_list):
        return [f"MOTSeperator_{i:03d}" for i in cur_list]

    def list2var(self, cur_list):
        return [self.varnode[f"VarNode_{i:03d}"] for i in cur_list]

    def list2pot(self, cur_list):
        shape = tuple([self.node_dim] * len(cur_list))
        noise = np.exp(self.rng.randint(0, 1, size=shape))
        stair = np.arange(np.prod(shape)).reshape(shape) * 20
        return stair + noise

    def setup_var(self):
        for i in range(0, self.num_node):
            node = VarNode(rv_dim=self.node_dim)
            node.format_name(f"VarNode_{i:03d}")
            self.varnode[node.name] = node
        for i in self.constrained:
            node = VarNode(
                rv_dim=self.node_dim,
                constrained_marginal=ot.datasets.make_1D_gauss(
                    self.node_dim,
                    m=i * self.node_dim / self.num_node,
                    s=self.node_dim / self.num_node)
            )
            node.format_name(f"VarNode_{i:03d}")
            self.varnode[node.name] = node

    def setup_cluster(self):
        for vars, seps in zip(self.cluster2var, self.cluster2sep):
            node = MOTCluster(self.list2seq(seps),
                              self.list2var(vars),
                              potential=self.list2pot(vars))
            self.mot.add_cluster(node)

    def setup_sep(self):
        for vars in self.sep2var:
            node = MOTSeperator(self.list2var(vars))
            self.mot.add_seperator(node)


if __name__ == "__main__":
    mot_helper = SetUpGraph()
    mot_helper.setup()
    mot = mot_helper.mot
    mot.bake()
    mot.itsbp()

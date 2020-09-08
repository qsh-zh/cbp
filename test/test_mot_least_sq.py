from cbp.node import VarNode, MOTCluster, MOTSeperator
from cbp.graph import MOTGraph
from scipy.ndimage import gaussian_filter1d
import unittest
import numpy as np


def visualize_dist(x_list, dist_list, bin_locs, strength):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    num_line, num_bins = dist_list.shape
    for i, x in enumerate(x_list):
        if i == 0 or i == len(x_list) - 1:
            ax.plot(np.ones(num_bins) * x, bin_locs, dist_list[i], 'r')
        else:
            ax.plot(np.ones(num_bins) * x, bin_locs, dist_list[i], 'b')

    # return ax
    plt.savefig(f'data/viz_{strength}.png')


class TestMotLeastSq(unittest.TestCase):
    def construct_dist(self):
        self.strength = 50000
        self.rng = np.random.RandomState(1)
        self.dim = 100 + 1
        self.num_sample = 20 + 1
        self.locs = np.linspace(0, 1.0, self.dim)
        self.varnode = {}
        for i in range(0, self.num_sample):
            zero = np.zeros(self.dim)
            zero[int((self.dim + 1) / 2) - 2 * i] = i / 1
            zero[int((self.dim + 1) / 2) + 2 *
                 i] = (self.num_sample - 1 - i) / 1
            zero = gaussian_filter1d(zero, 1)
            node = VarNode(rv_dim=self.dim,
                           constrained_marginal=zero / np.sum(zero))
            node.format_name(f"VarNode_{i:03d}")
            self.varnode[node.name] = node

    def cal_potential(self, i):
        potential = np.zeros((self.dim, self.dim, self.dim))
        it = np.nditer(potential, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            coef_start = 1 - i / (self.num_sample - 1)
            potential[idx] = np.power(self.locs[idx[1]]
                                      - coef_start * self.locs[idx[0]]
                                      - (1 - coef_start) * self.locs[idx[2]], 2)
        return np.exp(-self.strength * potential)

    def constuct_graph(self):
        motgraph = MOTGraph()
        node_start, node_end = VarNode(self.dim), VarNode(self.dim)
        node_start.format_name("VarNode_000")
        node_end.format_name(f"VarNode_{self.dim-1:03d}")
        center_sep = MOTSeperator([node_start, node_end])
        motgraph.add_seperator(center_sep)
        random_leaf = self.rng.permutation(
            np.arange(1, self.num_sample - 1))[:10]
        choosed_dist = []
        for i in random_leaf:
            cur_var = self.varnode[f"VarNode_{i:03d}"]
            sep = MOTSeperator([cur_var])
            motgraph.add_seperator(sep)
            cluster = MOTCluster([center_sep.name, sep.name],
                                 [node_start, cur_var, node_end],
                                 potential=self.cal_potential(i))
            motgraph.add_cluster(cluster)
            leaf = MOTCluster([sep.name], [cur_var])
            motgraph.add_cluster(leaf)
            choosed_dist.append(cur_var.constrained_marginal)

        motgraph.bake()
        # motgraph.plot("data/least_sq.png")
        print("Finish build graph")
        return motgraph, random_leaf / (self.num_sample - 1), choosed_dist

    def test_run(self):
        self.construct_dist()
        mot, x_array, dist = self.constuct_graph()
        x_list = x_array.tolist()
        x_list.insert(0, 0)
        x_list.insert(0, 0)
        x_list.append(1.0)
        x_list.append(1.0)
        result = mot.itsbp()
        mot_node_marginal = mot.export_node_marginal()
        dist.insert(0, mot_node_marginal["VarNode_000"])
        dist.insert(0, self.varnode[f"VarNode_{0:03d}"].constrained_marginal)
        dist.append(
            self.varnode[f"VarNode_{self.num_sample-1:03d}"].constrained_marginal)
        dist.append(mot_node_marginal[f"VarNode_{self.dim-1:03d}"])

        visualize_dist(x_list, np.array(dist), self.locs, self.strength)
        # print(mot_node_marginal["VarNode_000"])
        # print(mot_node_marginal[f"VarNode_{self.dim-1:03d}"])

    # def test_onepoint(self):
    #     node_start = VarNode(3)
    #     node_end = VarNode(3)
    #     node_mid = VarNode(3, constrained_marginal=np.array(
    #         [0.0 / 3, 3.0 / 3, 0.0 / 3]))
    #     node_start.format_name("VarNode_000")
    #     node_mid.format_name("VarNode_001")
    #     node_end.format_name("VarNode_002")
    #     mot = MOTGraph()
    #     center_sep = MOTSeperator([node_start, node_end
    #                                ])
    #     mid_sep = MOTSeperator([node_mid])
    #     mot.add_seperator(center_sep)
    #     mot.add_seperator(mid_sep)

    #     potential = np.zeros((3, 3, 3))
    #     coef_start = 1.0 / 3
    #     it = np.nditer(potential, flags=['multi_index'])
    #     for _ in it:
    #         idx = it.multi_index
    #         potential[idx] = np.power(
    #             idx[1] - coef_start * idx[0] - (1 - coef_start) * idx[2], 2)

    #     potential = np.exp(-1 * potential)

    #     center_cluster = MOTCluster([mid_sep.name, center_sep.name],
    #                                 [node_start, node_mid, node_end],
    #                                 potential=potential)
    #     mot.add_cluster(center_cluster)
    #     mot.add_cluster(MOTCluster([mid_sep.name], [node_mid]))

    #     mot.bake()
    #     mot.itsbp()
    #     mot_node_marginal = mot.export_node_marginal()
    #     print(mot_node_marginal["VarNode_000"])
    #     print(mot_node_marginal["VarNode_002"])

    #     for i in range(3):
    #         potential[:, i, :] *= 1 / np.sum(potential[:, i, :])

    #     print(potential.sum(axis=(1, 2)) / potential.sum())
    #     print(potential.sum(axis=(0, 1)) / potential.sum())

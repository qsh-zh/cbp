from matplotlib.lines import Line2D
import cbp.utils.np_utils as npu
from cbp.node import VarNode, MOTCluster, MOTSeperator
from cbp.graph import MOTGraph
from scipy.ndimage import gaussian_filter1d
import unittest
import numpy as np
from numba import jit
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
PLTLine = namedtuple('PLTLine', ['x', 'density', 'color'])


class VizDist:
    def __init__(self, num_sample, locs):
        self.num_sample = num_sample
        self.bin_locs = locs
        self.lines = []

    def add_line(self, x, density, color='blue'):
        self.lines.append(PLTLine(x / (self.num_sample - 1), density, color))

    def plot(self, name='a.png'):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        len_x = len(self.bin_locs)
        for line in self.lines:
            if line.color == 'blue':
                alpha = 0.4
            else:
                alpha = 0.8
            ax.plot(np.ones(len_x) * line.x, self.bin_locs, line.density,
                    line.color, alpha=alpha)

        ax.legend([Line2D([0], [0], color='blue'), Line2D(
            [0], [0], color='red')], ['Ground Truth', 'Estimated Interpolation'])
        ax.view_init(60, 300)
        plt.show()
        plt.savefig(name)


class BuildMOTLsPotential:
    def __init__(self, num_dim, locs):
        tensor_shape = (num_dim, num_dim, num_dim)
        self.inter_point = npu.nd_expand(locs, tensor_shape, 1)
        self.start_point = npu.nd_expand(locs, tensor_shape, 0)
        self.end_point = npu.nd_expand(locs, tensor_shape, 2)

    def tri_potential(self, per_start):
        diff = self.inter_point - per_start * self.start_point - \
            (1 - per_start) * self.end_point
        return np.power(diff, 2)

    def bi_potential(self):
        start_point = self.start_point[:, :, 0]
        end_point = self.end_point[0, :, :]
        return np.power(start_point - end_point, 2)


# @jit(nopython=True, cache=True)
# def construct_dist(num_dim, num_sample):
#     zero = np.zeros((num_sample, num_dim))
#     for i in range(0, num_sample):
#         zero[i, int((num_dim + 1) / 2) - 2 * i] = i / 1
#         zero[i, int((num_dim + 1) / 2) + 2 *
#              i] = (num_sample - 1 - i) / 1
#     return gaussian_filter1d(zero, 1)


def construct_dist(num_dim, num_sample):
    start = np.zeros(num_dim)
    end = np.zeros(num_dim)
    for i in range(int(num_dim / 2), num_dim, 5):
        start[i] = num_dim - i
    start[int(num_dim / 3)] = (num_dim) / 3
    for i in range(0, int(num_dim / 2), 5):
        end[i] = i
    end[int(2 * num_dim / 3)] = (num_dim) / 3
    start_dist = gaussian_filter1d(start, 5)
    end_dist = gaussian_filter1d(end, 5)

    interpolater = DistInterpolate(
        start_dist, end_dist, num_dim, np.linspace(
            0, 1, num_dim))

    rtn_list = [start_dist]
    for i in range(1, num_sample - 1):
        rtn_list.append(interpolater.interpolate(1 - i / (num_sample - 1)))

    rtn_list.append(end_dist)
    return np.array(rtn_list)


class DistInterpolate:
    def __init__(self, start_dist, end_dist, num_dim, locs):
        ex_shape = (num_dim, num_dim)
        self.start_dist = npu.nd_expand(start_dist, ex_shape, 0)
        self.end_dist = npu.nd_expand(end_dist, ex_shape, 1)
        self.dist = (self.start_dist * self.end_dist).flatten()
        self.start_val = npu.nd_expand(locs, ex_shape, 0)
        self.end_val = npu.nd_expand(locs, ex_shape, 1)
        self.locs = np.append(locs, [1.01])

    def interpolate(self, per_start):
        val = (self.start_val * per_start +
               self.end_val * (1 - per_start)).flatten()
        rtn_dist, _ = np.histogram(val, self.locs, weights=self.dist)
        return rtn_dist


class TestMotLeastSq(unittest.TestCase):
    def init(self):
        self.strength = 50000
        self.rng = np.random.RandomState(1)
        self.dim = 100 + 1
        self.num_sample = 20 + 1
        self.locs = np.linspace(0, 1.0, self.dim)
        self.viz = VizDist(self.num_sample, self.locs)
        self.varnode = {}
        self.construct_var()
        self.potential_maker = BuildMOTLsPotential(self.dim, self.locs)
        self.is_penalty = True

    def construct_var(self):
        dists = construct_dist(self.dim, self.num_sample)
        for i in range(0, self.num_sample):
            node = VarNode(rv_dim=self.dim,
                           constrained_marginal=dists[i] / np.sum(dists[i]))
            node.format_name(f"VarNode_{i:03d}")
            self.varnode[node.name] = node

        for i in [0, self.num_sample - 1]:
            node = self.varnode[f"VarNode_{i:03d}"]
            self.viz.add_line(i, node.constrained_marginal)

    def cal_potential(self, i):
        potential = self.potential_maker.tri_potential(
            1 - i / (self.num_sample - 1))
        return np.exp(-self.strength * potential)

    def constuct_graph(self):
        motgraph = MOTGraph()
        node_start, node_end = VarNode(self.dim), VarNode(self.dim)
        node_start.format_name("VarNode_000")
        node_end.format_name(f"VarNode_{self.num_sample-1:03d}")
        center_sep = MOTSeperator([node_start, node_end])
        motgraph.add_seperator(center_sep)
        random_leaf = self.rng.permutation(
            np.arange(1, self.num_sample - 1))[:10]

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

            self.viz.add_line(i, cur_var.constrained_marginal)

        if self.is_penalty:
            potential = np.exp(-self.strength *
                               self.potential_maker.bi_potential())
            cluster = MOTCluster([center_sep.name], [
                                 node_start, node_end], potential=potential)
            motgraph.add_cluster(cluster)

        motgraph.bake()
        # motgraph.plot("data/least_sq.png")
        print("Finish build graph")
        return motgraph, random_leaf

    @unittest.skip("Move it to example")
    def test_run(self):
        self.init()
        mot, x_array = self.constuct_graph()
        result = mot.itsbp()
        mot_node_marginal = mot.export_node_marginal()
        infer_start_dist = mot_node_marginal["VarNode_000"]
        infer_end_dist = mot_node_marginal[f"VarNode_{self.num_sample-1:03d}"]
        interpolater = DistInterpolate(
            infer_start_dist, infer_end_dist, self.dim, self.locs)
        for t in x_array:
            dist = interpolater.interpolate(
                (self.num_sample - 1 - t) / (self.num_sample - 1))
            self.viz.add_line(t, dist, 'red')
        self.viz.add_line(0, infer_start_dist, 'red')
        self.viz.add_line(self.num_sample - 1, infer_end_dist, 'red')
        self.viz.plot()

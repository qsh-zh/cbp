import unittest
import numpy as np
from cbp.builder import GOHMMSimulator, HMMGOSimBuilder, PotentialType
from cbp.graph import bp_policy


def construct_init(dim):
    init = np.array([(dim - i) for i in range(dim)])
    return init / np.sum(init)


def construct_shift_trans(dim):
    ones = np.eye(dim)
    shift = np.hstack([ones[:, 1:], ones[:, 0:1]])
    return shift


def construct_gaussian_emission(dim):
    loc = 5 * np.arange(dim)
    scale = [1] * dim
    return {"loc": np.array(loc), "scale": np.array(scale)}


class TestITSBPHMM(unittest.TestCase):

    def test_acc(self):
        dim = 5
        length = 4
        sim = GOHMMSimulator(length, dim, 1)
        sim.register_potential(PotentialType.INIT, construct_init(dim))
        sim.register_potential(PotentialType.TRANSITION,
                               construct_shift_trans(dim))
        sim.register_potential(PotentialType.EMISSION,
                               construct_gaussian_emission(dim))
        sim.sample(2000)
        builder = HMMGOSimBuilder(length, sim, bp_policy)

        graph = builder()
        builder.fix_initpotential()
        graph.bake()

        discrete_graph = graph.discrete_graph
        discrete_graph.first_belief_propagation()
        for i in range(20):
            print(i, discrete_graph.get_node("VarNode_000").marginal(),
                  builder.compare_acc())
            discrete_graph.itsbp_outer_loop()

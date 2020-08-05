import unittest
from cbp.builder import MigrSimulator, HMMSimBuilder
from cbp.graph import bp_policy


class TestITSBPHMM(unittest.TestCase):
    def test_acc(self):
        sim = MigrSimulator(15, 15, 15, 1)
        sim.compile()
        sim.sample(10)
        builder = HMMSimBuilder(15, sim, bp_policy)
        graph = builder()
        graph.bake()
        builder.fix_initpotential()
        graph.first_belief_propagation()
        print(0, builder.compare_acc())
        for i in range(20):
            graph.itsbp_outer_loop()
            print(i, builder.compare_acc())

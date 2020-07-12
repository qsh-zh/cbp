import unittest
import numpy as np
from cbp.builder import HMMSimulator, PotentialType


class TestSim(unittest.TestCase):
    def test_sim_sample(self):
        rng = np.random.RandomState(1)
        for i in range(10):
            state_num = 10
            obser_num = 5
            sim = HMMSimulator(10, state_num, obser_num, i, is_theorem=False)
            sim.register_potential(
                PotentialType.INIT, rng.dirichlet(
                    [1] * state_num))
            sim.register_potential(
                PotentialType.TRANSITION, rng.dirichlet(
                    [1] * state_num, size=state_num))
            sim.register_potential(
                PotentialType.EMISSION, rng.dirichlet(
                    [1] * obser_num, size=state_num))
            sim.sample(5000)
            sim.record.cal_theory(False)
            self.assertGreater(0.5, sim.record["sim_state_error"])
            self.assertGreater(0.5, sim.record["sim_obser_error"])

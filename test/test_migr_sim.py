import unittest

from cbp.builder import MigrSimulator


class TestMigrSim(unittest.TestCase):
    def test_process(self):
        sim = MigrSimulator(3, 3, 3, 1)
        sim.compile()
        sim.sample(10)
        sim.viz_gt()
        sim.viz_sensor()
        sim.viz_emission_potential()
        sim.viz_trans_potential()

from cbp.builder import WifiSimulator, WifiHMMBuilder
from cbp.graph.coef_policy import bp_policy
import unittest


class TestWifiSim(unittest.TestCase):
    def test_wifi_sim(self):
        sim = WifiSimulator(3, 5, 5, 3)
        sim.random_sensor(10)
        sim.compile()
        sim.sample(1000)
        sim.viz_gt()
        sim.viz_sensor()
        sim.viz_trans_potential()

    @unittest.skip("test build")
    def test_wifi_builder(self):
        builder = WifiHMMBuilder(
            length=20,
            grid_d=10,
            policy=bp_policy,
            rand_seed=5,
            num_sensor=64,
            time_step=20)

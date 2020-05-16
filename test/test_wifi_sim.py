from cbp.builder import WifiSimulator
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

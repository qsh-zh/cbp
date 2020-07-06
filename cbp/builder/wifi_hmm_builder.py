from pathlib import Path

from .wifi_simulator import WifiSimulator
from .hmm_sim_builder import HMMSimBuilder


class WifiHMMBuilder(HMMSimBuilder):
    """builder with wifi-type simulator. Sparse observation. pass the
    wifi-simulator to the `cbp.builder.HMMSimBuilder`
    """

    def __init__(self, length, grid_d, policy,  # pylint: disable=too-many-arguments
                 rand_seed=1, num_sensor=16, time_step=60):
        sim_name = f'wifi-timestep_{time_step}-d_{grid_d}-rs_{rand_seed}-ns_{num_sensor}'
        sim_path = Path(f"data/sim/{sim_name}/sim.pkl")
        if sim_path.is_file():
            print(f"loading simulator from {sim_path}")
            simulator = WifiSimulator.load(sim_name)
        else:
            simulator = WifiSimulator(time_step, grid_d, grid_d, rand_seed)#Changed 60 to time_step
            simulator.name = sim_name
            simulator.random_sensor(num_sensor)
            simulator.compile()
            simulator.sample(10000)
            simulator.viz_sensor()
            simulator.viz_gt()
            print(f"saving simulator in {simulator.path}")
            simulator.save()

        super().__init__(length, simulator, policy, rand_seed)

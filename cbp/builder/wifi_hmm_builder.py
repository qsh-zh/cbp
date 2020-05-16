import pickle
from pathlib import Path

from cbp.node import FactorNode

from .hmm_builder import HMMBuilder
from .wifi_simulator import WifiSimulator


class WifiHMMBuilder(HMMBuilder):
    def __init__(self, length, grid_d, policy,
                 rand_seed=1, num_sensor=16, time_step=60):
        super().__init__(length, grid_d * grid_d, policy, rand_seed)
        simulator_path = Path(
            f'data/wifi-timestep_{time_step}-d_{grid_d}-rs_{rand_seed}-ns_{num_sensor}.pkl')
        if simulator_path.is_file():
            print(f"loading simulator from {simulator_path}")
            with open(simulator_path, 'rb') as handle:
                self.simulator = pickle.load(handle)
        else:
            Path('data').mkdir(exist_ok=True)
            self.simulator = WifiSimulator(60, grid_d, grid_d, rand_seed)
            self.simulator.random_sensor(num_sensor)
            self.simulator.compile()
            self.simulator.sample(10000)
            self.simulator.viz_sensor()
            print(f"saving simulator in {simulator_path}")
            with open(simulator_path, 'wb') as handle:
                pickle.dump(self.simulator, handle)
        self.num_sensor = num_sensor
        self.cnt_constrained_node = 0

    def fix_initpotential(self, potential=None):
        first_node = self.graph.get_node(f"VarNode_{0:03d}")
        if potential is None:
            potential = self.simulator.get_gt_marginal(0)
        first_node.potential = potential

    def add_constrained_node(self, probability=None):
        prob = self.simulator.get_constrained_marginal(
            self.cnt_constrained_node)
        self.cnt_constrained_node += 1
        return super().add_constrained_node(prob)

    def add_factor(self, name_list, is_conv=False):
        """add factor to hmm graph

        :param name_list: connected node
        :type name_list: list
        :param is_conv: is emit or not , defaults to False-- transition
        :type is_conv: bool, optional
        :return: FactorNode
        :rtype: cbp.FactorNode
        """
        if is_conv:  # emit
            potential = self.simulator.get_observation_potential()
        else:
            potential = self.simulator.get_tansition_potential()
        factornode = FactorNode(name_list, potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_branch(self, head_node=None, is_constrained=False,
                   prob=None, is_conv=False):
        if is_conv:  # emit
            return super().add_branch(head_node, True, prob, is_conv)
        return super().add_branch(head_node, False, prob, is_conv)

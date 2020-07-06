import numpy as np
from numpy.random import RandomState
from scipy.ndimage import gaussian_filter
from cbp.utils.np_utils import batch_normal_angle

from .migr_visualizer import MigrVisualizer
from .hmm_simulator import HMMSimulator, PotentialType


class MigrSimulator(
        HMMSimulator):  # pylint: disable=too-many-instance-attributes
    def __init__(self, time_step, d_col, d_row, random_seed):  # pylint: disable=too-many-function-args
        super().__init__(time_step, d_col * d_row, d_col * d_row, random_seed)
        self.d_row = d_row
        self.d_col = d_col
        self.rng = RandomState(random_seed)

        self._sim = {
            "angle_wind": np.pi / 2,
            "sensor_sigma": 4,
            "num_sensor": self.record.state_num,
            "destination": (self.d_row, self.d_col)
        }
        self.visualizer = MigrVisualizer(self.d_row, self.d_col)

    def compile(self):
        """register various potential for simulation
        """
        self._produce_transition_potential()
        self._produce_sensor_potential()

        init_potential = np.zeros(self.record.state_num)
        init_potential[0] = 1
        self.register_potential(PotentialType.INIT, init_potential)

    def _transition_factor_goal(self, cur_col, cur_row):
        col, row = np.meshgrid(np.arange(self.d_col), np.arange(self.d_row))
        diff_row = row - cur_row
        diff_col = col - cur_col

        angle_goal = np.arctan2(
            self._sim["destination"][0] - 1 - cur_row,
            self._sim["destination"][1] - 1 - cur_col)

        angle_matrix = np.arctan2(diff_row, diff_col)

        goal_matrix = np.abs(
            batch_normal_angle(
                angle_matrix - angle_goal))
        return 5 * goal_matrix

    def _transition_factor_wind(self, cur_col, cur_row):
        col, row = np.meshgrid(np.arange(self.d_col), np.arange(self.d_row))
        diff_row = row - cur_row
        diff_col = col - cur_col

        angle_matrix = np.arctan2(diff_row, diff_col)

        wind_matrix = np.abs(
            batch_normal_angle(
                angle_matrix -
                self._sim["angle_wind"]))

        return 3 * wind_matrix

    def _transition_factor_dist(self, cur_col, cur_row):
        col, row = np.meshgrid(np.arange(self.d_col), np.arange(self.d_row))
        diff_row = row - cur_row
        diff_col = col - cur_col
        dist_matrix = np.sqrt(np.power(diff_row, 2)
                              + np.power(diff_col, 2))

        return 1.6 * dist_matrix

    def _transition_logistic_regression(self, cur_col, cur_row):
        dist_matrix = self._transition_factor_dist(cur_col, cur_row)
        wind_matrix = self._transition_factor_wind(cur_col, cur_row)
        goal_matrix = self._transition_factor_goal(cur_col, cur_row)

        exponent = -dist_matrix - wind_matrix - goal_matrix
        exponent[cur_row, cur_col] += 1

        cur_prob = np.exp(exponent)

        return cur_prob.flatten() / np.sum(cur_prob)

    def _produce_transition_potential(self):
        potential = []

        for cur_row in range(self.d_row):
            for cur_col in range(self.d_col):
                potential.append(
                    self._transition_logistic_regression(
                        cur_col, cur_row))
        self.register_potential(PotentialType.TRANSITION, np.array(
            potential).reshape(self.record.state_num, self.record.state_num))

    def _produce_sensor_potential(self):
        potential = []
        for i in range(self.d_row):
            for j in range(self.d_col):
                empty_matrix = np.zeros((self.d_row, self.d_col))
                empty_matrix[i, j] = 100.0
                result = gaussian_filter(
                    empty_matrix, sigma=self._sim["sensor_sigma"])
                potential.append(result.flatten() / np.sum(result))

        self.register_potential(PotentialType.EMISSION, np.array(
            potential).reshape(self.record.state_num, self.record.state_num))

    def viz_emission_potential(self):
        self.visualizer.potential_heatmap(
            self.record[PotentialType.EMISSION],
            title="sensor_potential",
            path=f"{self.path}/sensor_potential")

    def viz_trans_potential(self):
        self.visualizer.potential_heatmap(
            self.record[PotentialType.TRANSITION],
            title="transition_potential",
            path=f"{self.path}/transition_potential")

    def viz_gt(self):
        self.visualizer.migration(self.record["traj"],
                                  **{"title": "bird traj",
                                     "path": f"{self.path}/gt",
                                     "ylabel": True,
                                     "xlabel": 'Ground Truth'})

    def viz_sensor(self):
        self.visualizer.migration(self.record["sensor"],
                                  **{"title": "bird traj",
                                     "path": f"{self.path}/sensor"})

    def viz_estm(self, estimated_marginal):
        for i in range(self.record.time_step):
            bins = self.record["num_sample"] * \
                estimated_marginal[i, :]
            png_name = f"{self.path}/estimated_{i}.png"
            self.visualizer.visualize_map_bins(
                bins, fig_name=png_name, xlabel='Estimated')

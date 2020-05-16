from pathlib import Path

import numpy as np
from numpy.random import RandomState
from scipy.ndimage import gaussian_filter
from cbp.utils.np_utils import batch_normal_angle, empirical_marginal

from .migr_visualizer import MigrVisualizer


class MigrSimulator:  # pylint: disable=too-many-instance-attributes
    def __init__(self, time_step, d_col, d_row, random_seed):  # pylint: disable=too-many-function-args
        Path('data/migr').mkdir(parents=True, exist_ok=True)
        self.time_step = time_step
        self.d_row = d_row
        self.d_col = d_col
        self.status_d = self.d_row * self.d_col
        self.rng = RandomState(random_seed)

        self._sim = {
            "angle_wind": np.pi / 2,
            "sensor_sigma": 4,
            "num_sensor": self.status_d,
            "destination": (self.d_row, self.d_col)
        }
        self.visualizer = MigrVisualizer(self.d_row, self.d_col)

        self._prcs = {}

    def compile(self):
        self._produce_transition_potential()
        self._produce_sensor_potential()

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

        self._prcs["transition_potential"] = np.array(
            potential).reshape(self.status_d, self.status_d)

    def _produce_sensor_potential(self):
        potential = []
        for i in range(self.d_row):
            for j in range(self.d_col):
                empty_matrix = np.zeros((self.d_row, self.d_col))
                empty_matrix[i, j] = 100.0
                result = gaussian_filter(
                    empty_matrix, sigma=self._sim["sensor_sigma"])
                potential.append(result.flatten() / np.sum(result))

        self._prcs["sensor_potential"] = np.array(
            potential).reshape(self.status_d, self.status_d)

    def sample_engine(self, state, dim, potential):
        """sample according to conditional prob table

        :param state: cur_state
        :type state: int
        :param dim: range of next state
        :type dim: int
        :param potential: conditional prob table
        :type potential: ndarray
        :return: next state
        :rtype: int
        """
        assert state < self.status_d
        conditional_prob = potential[state, :]
        next_state = self.rng.choice(
            dim, p=conditional_prob / np.sum(conditional_prob))
        return next_state

    def step(self, state):
        return self.sample_engine(state, self.status_d,
                                  self._prcs["transition_potential"])

    def observe(self, state):
        return self.sample_engine(state, self._sim["num_sensor"],
                                  self._prcs["sensor_potential"])

    def init_stats_sampler(self):
        return 0

    def sample(self, num_sample):
        self._sim["num_sample"] = num_sample
        traj_recorder = []
        sensor_recorder = []
        for _ in range(num_sample):
            states = []
            sensors = []
            single_state = self.init_stats_sampler()
            for _ in range(self.time_step):
                sensors.append(self.observe(single_state))
                states.append(single_state)
                single_state = self.step(single_state)
            traj_recorder.append(states)
            sensor_recorder.append(sensors)

        self._prcs["traj"] = np.array(
            traj_recorder).reshape(num_sample, self.time_step)
        self._prcs["sensor"] = np.array(sensor_recorder).reshape(
            num_sample, self.time_step)

    def viz_sensor_potential(self):
        self.visualizer.potential_heatmap(
            self._prcs["sensor_potential"],
            title="sensor_potential",
            path="data/migr/sensor_potential")

    def viz_trans_potential(self):
        self.visualizer.potential_heatmap(
            self._prcs["transition_potential"],
            title="transition_potential",
            path="data/migr/transition_potential")

    def ind2rowcol(self, index):
        index = np.array(index).astype(np.int64)
        row = (index / self.d_col).astype(np.int64)
        col = index % self.d_col
        return row, col

    def viz_gt(self):
        self.visualizer.migration(self._prcs["traj"],
                                  **{"title": "bird traj",
                                     "path": "data/migr/gt",
                                     "ylabel": True,
                                     "xlabel": 'Ground Truth'})

    def viz_sensor(self):
        self.visualizer.migration(self._prcs["sensor"],
                                  **{"title": "bird traj",
                                     "path": "data/migr/sensor"})

    def viz_estm(self, estimated_marginal):
        for i in range(self.time_step):
            bins = self._sim["num_sample"] * \
                estimated_marginal[i, :]
            png_name = f"data/migr/estimated_{i}.png"
            self.visualizer.visualize_map_bins(
                bins, fig_name=png_name, xlabel='Estimated')

    def get_constrained_marginal(self, time_step=None):
        if "constrained_marginal" not in self._prcs:
            self._prcs["constrained_marginal"] = empirical_marginal(
                self._prcs["sensor"], self._sim["num_sensor"])

        if isinstance(time_step, int):
            return self._prcs["constrained_marginal"][time_step, :]
        else:
            return self._prcs["constrained_marginal"]

    def get_gt_marginal(self, time_step=None):
        if "gt_marginal" not in self._prcs:
            self._prcs["gt_marginal"] = empirical_marginal(
                self._prcs["traj"], self.status_d)

        if isinstance(time_step, int):
            return self._prcs["gt_marginal"][time_step, :]
        else:
            return self._prcs["gt_marginal"]

    def get_tansition_potential(self):
        return self._prcs["transition_potential"]

    def get_sensor_potential(self):
        return self._prcs["sensor_potential"]

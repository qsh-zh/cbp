import pickle
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import seaborn as sns
from cbp.utils.np_utils import empirical_marginal


class PotentialType(str, Enum):
    """Potential Type Enum
    """
    INIT = 'Init'
    TRANSITION = 'Transition'
    EMISSION = 'Emission'


class HMMSimulator:  # pylint: disable=too-many-instance-attributes
    """simulator from a time-homogenous process.\
        The setup need:

        * potential: type in the `cbp.builder.PotentialType`, register various potential
        * change to a distinguishable name
        * call sample methods
    """

    def __init__(self, time_step, dim_status, dim_observations, random_seed):  # pylint: disable=too-many-function-args
        self.__name = None
        self.path = Path('data/sim')
        self.path.mkdir(parents=True, exist_ok=True)
        self.time_step = time_step
        self.status_d = dim_status
        self.obser_d = dim_observations
        self.rng = RandomState(random_seed)

        self._prcs = {}

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        self.__name = new_name
        self.path = Path(f"data/sim/{new_name}")
        self.path.mkdir(parents=True, exist_ok=True)

    def register_potential(self, ptype, potential):
        """register potential for simulation

        :param ptype: potential type
        :type ptype: PotentialType
        :param potential: [description]
        :type potential: ndarray
        """
        assert isinstance(ptype, PotentialType)

        if ptype == PotentialType.TRANSITION:
            assert np.isclose(potential.sum(axis=1), 1).all(),\
                "potential should be a conditional distribution"
            assert potential.shape == (self.status_d, self.status_d)
        elif ptype == PotentialType.EMISSION:
            assert np.isclose(potential.sum(axis=1), 1).all(),\
                "potential should be a conditional distribution"
            assert potential.shape == (self.status_d, self.obser_d)
        elif ptype == PotentialType.INIT:
            assert potential.shape == (self.status_d,)

        self._prcs[ptype] = potential

    def sample_engine(self, state, dim, potential):
        """sample according to conditional prob table

        :param state: cur_state
        :type state: int
        :param dim: range of next state or observation
        :type dim: int
        :param potential: conditional prob table
        :type potential: ndarray
        :return: next state or observation
        :rtype: int
        """
        assert state < self.status_d
        conditional_prob = potential[state, :]
        next_state = self.rng.choice(
            dim, p=conditional_prob)
        return next_state

    def step(self, state):
        """do a status transition sample

        :param state: current status
        :type state: int
        :return: the next status
        :rtype: int
        """
        return self.sample_engine(state, self.status_d,
                                  self._prcs[PotentialType.TRANSITION])

    def observe(self, state):
        """do a emission sample

        :param state: current status
        :type state: int
        :return: the observation of cur status
        :rtype: int
        """
        return self.sample_engine(state, self.obser_d,
                                  self._prcs[PotentialType.EMISSION])

    def __init_stats_sampler(self):  # pylint: disable=no-self-use
        return self.rng.choice(self.status_d, p=self._prcs[PotentialType.INIT])

    def sample(self, num_sample):
        """start simulation process with many particles

        :param num_sample: num of particles
        :type num_sample: int
        """
        self._prcs["num_sample"] = num_sample
        traj_recorder = []
        sensor_recorder = []
        for _ in range(num_sample):
            states = []
            sensors = []
            single_state = self.__init_stats_sampler()
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

    def viz_emission_potential(self):
        axes = sns.heatmap(self._prcs[PotentialType.EMISSION])
        axes.set_title("Emission Potential")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/hmm_emission.png")
        plt.close(fig)

    def viz_trans_potential(self):
        axes = sns.heatmap(self._prcs[PotentialType.TRANSITION])
        axes.set_title("Transition Potential")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/hmm_transition.png")
        plt.close(fig)

    def get_constrained_marginal(self, time_step=None):
        """return observation marginal

        :param time_step: if int then return a specific time distribution
        otherwise all distributions as matrxi, defaults to None
        :type time_step: int, optional
        :return: array for single time_step or a matrix
        :rtype: ndarray
        """
        if "constrained_marginal" not in self._prcs:
            self._prcs["constrained_marginal"] = empirical_marginal(
                self._prcs["sensor"], self.obser_d)

        if isinstance(time_step, int):
            return self._prcs["constrained_marginal"][time_step, :]

        return self._prcs["constrained_marginal"]

    def get_gt_marginal(self, time_step=None):
        """return ground truth marginal

        :param time_step: if int then return a specific time distribution
        otherwise all distributions as matrix, defaults to None
        :type time_step: int, optional
        :return: array for single time_step or a matrix
        :rtype: ndarray
        """
        if "gt_marginal" not in self._prcs:
            self._prcs["gt_marginal"] = empirical_marginal(
                self._prcs["traj"], self.status_d)

        if isinstance(time_step, int):
            return self._prcs["gt_marginal"][time_step, :]

        return self._prcs["gt_marginal"]

    def get_tansition_potential(self):
        return self._prcs[PotentialType.TRANSITION]

    def get_emission_potential(self):
        return self._prcs[PotentialType.EMISSION]

    def save(self):
        """save the instance to a pkl
        """
        sim_path = f"{self.path}/sim.pkl"
        with open(sim_path, 'wb') as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, sim_name: str):
        """load a simulator from pkl file

        :param sim_name: name for the simulator
        :type sim_name: str
        :return: simulator
        :rtype: simulator
        """
        sim_path = f"data/sim/{sim_name}/sim.pkl"
        with open(sim_path, 'rb') as handle:
            simulator = pickle.load(handle)
        return simulator

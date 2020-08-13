import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import jit
from numpy.random import RandomState

from cbp.utils.np_utils import empirical_marginal


class PotentialType(str, Enum):
    """Potential Type Enum
    """
    INIT = 'Init'
    TRANSITION = 'Transition'
    EMISSION = 'Emission'


@jit(cache=True, nopython=True)
def count_joint(traj, col_num, state_num):
    idx = traj[:, col_num:col_num + 2]
    empty = np.zeros((state_num, state_num))
    for i, j in zip(idx[:, 0], idx[:, 1]):
        empty[i, j] += 1
    return empty / traj.shape[0]


class TrajData(dict):
    def __init__(self, time_step, state_num):  # pylint: disable=super-init-not-called
        self.time_step = time_step
        self.state_num = state_num

    @property
    def traj(self):
        return dict.__getitem__(self, "traj")

    @traj.setter
    def traj(self, traj_data):
        assert traj_data.shape[1] == self.time_step
        dict.__setitem__(self, "traj", traj_data)
        dict.__setitem__(self, "gt_margin", empirical_marginal(
            traj_data, self.state_num))

    def _traj_joint(self):
        record = []
        for i in range(self.time_step - 1):
            record.append(count_joint(self["traj"], i, self.state_num))
        self["gt_joint"] = record

    def cal_traj_theory(self):
        transition = self[PotentialType.TRANSITION].T
        init = self[PotentialType.INIT].reshape(-1, 1)
        state_record = []
        for _ in range(self.time_step):
            state_record.append(init.flatten())
            init = transition @ init
        self["th_margin"] = np.array(state_record)
        state_err = np.linalg.norm(self["th_margin"] -
                                   self["gt_margin"])
        self["sim_state_error"] = state_err


class TrajSimulator(ABC):
    """discrete traj simulator, abstract inference for observation

    :param ABC: [description]
    :type ABC: [type]
    """

    def __init__(self, record, random_seed, is_theorem=False):
        self.__name = None
        self.path = Path('data/sim')
        self.path.mkdir(parents=True, exist_ok=True)
        self.is_theorem = is_theorem
        self.rng = RandomState(random_seed)
        self.record = record

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
            assert potential.shape == (
                self.record.state_num, self.record.state_num)
        elif ptype == PotentialType.INIT:
            assert potential.shape == (self.record.state_num,)

        self.record[ptype] = potential

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
        assert state < self.record.state_num
        conditional_prob = potential[state, :]
        next_state = self.rng.choice(dim, p=conditional_prob)
        return next_state

    def step(self, state):
        """do a states transition sample

        :param state: current states
        :type state: int
        :return: the next states
        :rtype: int
        """
        return self.sample_engine(state, self.record.state_num,
                                  self.record[PotentialType.TRANSITION])

    def __init_stats_sampler(self):
        return self.rng.choice(self.record.state_num,
                               p=self.record[PotentialType.INIT])

    def sample_traj(self, num_sample):
        self.record["num_sample"] = num_sample
        traj_recorder = []
        for _ in range(num_sample):
            states = []
            single_state = self.__init_stats_sampler()
            for _ in range(self.record.time_step):
                states.append(single_state)
                single_state = self.step(single_state)
            traj_recorder.append(states)

        self.record.traj = np.array(traj_recorder).reshape(num_sample, -1)

    def viz_trans_potential(self):
        axes = sns.heatmap(self.record[PotentialType.TRANSITION])
        axes.set_title("Transition Potential")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/hmm_transition.png")
        plt.close(fig)

    def viz_gt(self):
        gt_margin = self.get_hidden_margin()
        axes = sns.heatmap(gt_margin.T)
        axes.set_title("Evolution of distribution")
        axes.set_xlabel("time")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/gt.png")
        plt.close(fig)

    def get_init_potential(self):
        return self.record[PotentialType.INIT]

    def get_transition_potential(self):
        return self.record[PotentialType.TRANSITION]

    def get_hidden_margin(self, time_step=None):
        """return ground truth marginal

        :param time_step: if int then return a specific time distribution
        otherwise all distributions as matrix, defaults to None
        :type time_step: int, optional
        :return: array for single time_step or a matrix
        :rtype: ndarray
        """
        key_word = "th_margin" if self.is_theorem else "gt_margin"
        if isinstance(time_step, int):
            return self.record[key_word][time_step, :]

        return self.record[key_word]

    @abstractmethod
    def observe(self, state):
        """do a emission sample

        :param state: current states
        :type state: int
        :return: the observation of cur states
        :rtype: int
        """

    def observe_traj(self, traj):
        rtn = np.zeros_like(traj,dtype=float)
        loop_iter = np.nditer(traj, flags=['multi_index'])
        for i in loop_iter:
            rtn[loop_iter.multi_index] = self.observe(i)
        return rtn

    def sample(self, num_sample):
        self.sample_traj(num_sample)
        self.record.sensor = self.observe_traj(self.record.traj)

    def get_fix_margin(self, time_step=None):
        """return observation marginal

        :param time_step: if int then return a specific time distribution
        otherwise all distributions as matrxi, defaults to None
        :type time_step: int, optional
        :return: array for single time_step or a matrix
        :rtype: ndarray
        """
        key_word = "th_observation" if self.is_theorem else "fix_margin"
        if isinstance(time_step, int):
            return self.record[key_word][time_step, :]

        return self.record[key_word]

    def get_emission_potential(self):
        return self.record[PotentialType.EMISSION]

    def save(self):
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

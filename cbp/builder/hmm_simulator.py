import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cbp.utils.np_utils import empirical_marginal

from .base_simulator import PotentialType, TrajData, TrajSimulator


class DiscreteObserData(TrajData):
    def __init__(self, time_step, state_num, obser_num):
        super().__init__(time_step, state_num)
        self.obser_num = obser_num

    @property
    def sensor(self):
        return dict.__getitem__(self, "sensor")

    @sensor.setter
    def sensor(self, sensor_data):
        assert sensor_data.shape[1] == self.time_step
        dict.__setitem__(self, "sensor", sensor_data)
        dict.__setitem__(self, "fix_margin", empirical_marginal(
            self["sensor"], self.obser_num))

    def cal_theory(self, verbose):
        self.cal_traj_theory()
        observation_record = []
        emis = self[PotentialType.EMISSION].T
        traj = self["th_margin"]
        for i in range(self.time_step):
            obs = emis @ traj[i].reshape(-1, 1)
            observation_record.append(obs.flatten())
        self["th_observation"] = np.array(observation_record)
        obser_err = np.linalg.norm(self["th_observation"] -
                                   self["fix_margin"])
        self["sim_obser_error"] = obser_err
        state_err = self["sim_state_error"]
        if verbose:
            print(f"Sim state err: {state_err}")
            print(f"Sim obser err: {obser_err}")


class HMMSimulator(TrajSimulator):
    def __init__(self, time_step, dim_states, dim_observations, random_seed, is_theorem=False):  # pylint: disable=too-many-arguments
        record = DiscreteObserData(time_step, dim_states, dim_observations)
        super().__init__(record, random_seed, is_theorem)

    def register_potential(self, ptype, potential):
        if ptype == PotentialType.EMISSION:
            assert np.isclose(potential.sum(axis=1), 1).all(),\
                "potential should be a conditional distribution"
            assert potential.shape == (
                self.record.state_num, self.record.obser_num)
        return super().register_potential(ptype, potential)

    def observe(self, state):
        return self.sample_engine(state, self.record.obser_num,
                                  self.record[PotentialType.EMISSION])

    def viz_emission_potential(self):
        axes = sns.heatmap(self.record[PotentialType.EMISSION])
        axes.set_title("Emission Potential")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/hmm_emission.png")
        plt.close(fig)

    def viz_sensor(self):
        sensor = self.get_fix_margin()
        axes = sns.heatmap(sensor.T)
        axes.set_title("Evolution of distribution")
        axes.set_xlabel("time")
        fig = axes.get_figure()
        fig.savefig(f"{self.path}/sensor.png")
        plt.close(fig)

    def get_precious(self, time_step=None, verbose=False):
        """return precious marginal

        :param time_step: if int then return a specific time distribution
        otherwise all distributions as matrix, defaults to None
        :type time_step: int, optional
        :param verbose: whether or not ouput difference between simulation and theorical margin
        :return: array for single time_step or a matrix
        :rtype: ndarray
        """
        if "th_margin" not in self.record:
            self.record.cal_theory(verbose)

        if isinstance(time_step, int):
            return self.record["th_margin"][time_step, :]

        return self.record["th_margin"]

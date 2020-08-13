import numpy as np

from .base_simulator import PotentialType, TrajData, TrajSimulator


class ContinuousObserData(TrajData):
    @property
    def sensor(self):
        return dict.__getitem__(self, "fix_margin")

    @sensor.setter
    def sensor(self, sensor_data):
        assert sensor_data.shape[1] == self.time_step
        dict.__setitem__(self, "fix_margin", np.swapaxes(sensor_data, 0, 1))


class GOHMMSimulator(TrajSimulator):
    def __init__(self, time_step, dim_states, random_seed):
        record = ContinuousObserData(time_step, dim_states)
        super().__init__(record, random_seed, is_theorem=False)

    def register_potential(self, ptype, potential):
        if ptype == PotentialType.EMISSION:
            assert potential["loc"].shape == (self.record.state_num,)
            assert potential["scale"].shape == (self.record.state_num,)

        super().register_potential(ptype, potential)

    def observe(self, state):
        return self.rng.normal(loc=self.record[PotentialType.EMISSION]["loc"][state],
                               scale=self.record[PotentialType.EMISSION]["scale"][state])


class GMOHMMSimulator(TrajSimulator):
    def __init__(self, record, random_seed, is_theorem=False):
        super().__init__(record, random_seed, is_theorem=is_theorem)

    def register_potential(self, ptype, potential):
        if ptype == PotentialType.EMISSION:
            if potential["loc"].ndim > 1:
                setattr(self, "obser_dim", potential["loc"].shape[1])
            else:
                setattr(self, "obser_dim", 1)
            assert potential["loc"].shape[0] == self.record.state_num
            assert potential["scale"].shape[0] == self.record.state_num

        super().register_potential(ptype, potential)

    def observe_traj(self, traj):
        rtn = np.zeros_like((traj.shape[0], traj.shape[1], self.obser_dim),
                            traj, dtype=float)
        loop_iter = np.nditer(traj, flags=['multi_index'])
        for i in loop_iter:
            rtn[loop_iter.multi_index] = self.observe(i)
        return rtn

    def observe(self, state):
        return self.rng.multivariate_normal(mean=self.record[PotentialType.EMISSION]["loc"][state],
                                            cov=self.record[PotentialType.EMISSION]["scale"][state])

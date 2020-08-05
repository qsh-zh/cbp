from .base_simulator import TrajSimulator, TrajData, PotentialType


class ContinuousObserData(TrajData):
    @property
    def sensor(self):
        return dict.__getitem__(self, "fix_margin")

    @sensor.setter
    def sensor(self, sensor_data):
        assert sensor_data.shape[1] == self.time_step
        dict.__setitem__(self, "fix_margin", sensor_data.T)


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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .migr_simulator import MigrSimulator

from .hmm_simulator import PotentialType


class WifiSimulator(MigrSimulator):
    """simulate ensemble move, sensor is sparse
    - place sensors in some place
    - draw various data
    """

    def __init__(self, time_step, d_col, d_row, random_seed):
        # use random_seed 3, random_sensor 16, reproduce the experiment result
        super().__init__(time_step, d_col, d_row, random_seed)
        self.record.obser_num = 0
        self.hotspot = []

    def compile(self):
        super().compile()

        potential = np.zeros(self.record.state_num)
        potential[0] = 0.7
        potential[int(self.d_col / 2)] = 0.3
        self.register_potential(PotentialType.INIT, potential)

    def register_hotspot(self, row, col):
        self.record.obser_num += 1
        self.hotspot.append((row, col))

    def _produce_sensor_potential(self):
        potential = []
        for cur_row in range(self.d_row):
            for cur_col in range(self.d_col):
                cur_potential = np.zeros(self.record.obser_num)

                for i, sensor in enumerate(self.hotspot):
                    distance = np.linalg.norm(
                        [cur_row - sensor[0], cur_col - sensor[1]])
                    cur_potential[i] = np.min([0.99, 1.0 * np.exp(-distance)])

                potential.append(cur_potential / np.sum(cur_potential))

        self.register_potential(
            PotentialType.EMISSION,
            np.array(potential).reshape(
                self.record.state_num,
                self.record.obser_num))

        return int(self.d_col / 2)

    def fixed_sensors(self, space_d=1):
        for cur_row in range(0, self.d_row, space_d):
            for cur_col in range(0, self.d_col, space_d):
                self.register_hotspot(cur_row, cur_col)

    def random_sensor(self, num):
        for _ in range(num):
            row = self.rng.uniform(0, self.d_row - 1)
            col = self.rng.uniform(0, self.d_col - 1)
            self.register_hotspot(row, col)

    def draw_sensor(self):
        """draw wifi sensor position in grid
        """
        sensor_row = []
        sensor_col = []
        for sensor in self.hotspot:
            sensor_row.append(sensor[0] + 0.2)
            sensor_col.append(sensor[1] + 0.2)

        marker_col, marker_row = np.meshgrid(
            np.arange(self.d_col),
            np.arange(self.d_row))

        pos_x = np.concatenate((marker_col.flatten(), sensor_col), axis=0)
        pos_y = np.concatenate((marker_row.flatten(), sensor_row), axis=0)
        x_y_size = np.concatenate(
            (np.ones(int(marker_col.size)) * 1.0, np.ones(len(sensor_col)) * 5.0))
        style = ['cell'] * marker_col.size + ['sensor'] * len(sensor_row)
        ax = sns.scatterplot(x=pos_y,
                             y=pos_x,
                             size=x_y_size,
                             hue=style,
                             style=style,
                             markers={'cell': 's', 'sensor': 'X'})

        ax.set_xlim([-1, self.d_col])
        ax.set_xticks(np.arange(0, self.d_col, 2))
        ax.set_xticklabels(np.arange(0, self.d_col, 2).tolist())
        ax.set_ylim([-1, self.d_row])
        ax.set_yticks(np.arange(0, self.d_row, 2))
        ax.set_yticklabels(np.arange(0, self.d_row, 2).tolist())
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["cell", "sensor"])
        plt.savefig(f"{self.path}/sensor.png", bbox_inches='tight')
        plt.close()

    def viz_sensor(self):
        """draw raw sensor data
        """
        for i in range(self.record.time_step):
            locations = self.record["sensor"][:, i]
            bins, _ = np.histogram(
                locations, np.arange(
                    self.record.obser_num + 1))
            self.__draw_sensordata(bins, f"{self.path}/observer_{i}_step.png")

    def __draw_sensordata(self, bins, fig_name):
        xx = []
        yy = []
        xy_size = []
        for xy, cnt in enumerate(bins):
            if cnt > 0:
                xx.append(self.hotspot[xy][0])
                yy.append(self.hotspot[xy][1])
                xy_size.append(int(cnt))
        self.visualizer.visualize_location(xx, yy, xy_size, **{
            "fig_name": fig_name,
            "xlabel": 'Observed'})

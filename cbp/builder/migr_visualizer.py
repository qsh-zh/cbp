import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MigrVisualizer():
    def __init__(self, d_row, d_col):
        """initial vis. TopLeft is origin point.

        :param d_row: grid height
        :type d_row: int
        :param d_col: grid width
        :type d_col: grid width
        """
        self.d_row = d_row
        self.d_col = d_col
        self.states_num = d_row * d_col  # total states

    def potential_heatmap(self, data, **kwargs):
        """plot potential heatmap and save png.
        2 reserved keys:

        * ``title`` figure title
        * ``path`` figure path prefix

        :param data: i-th row represents the potential of d state
        :type data: ndarray
        """
        for i in range(self.states_num):
            distribution = data[i, :].reshape(self.d_row, self.d_col)
            axes = sns.heatmap(distribution)
            axes.set_title(f"{self.ind2rowcol(i)}_{kwargs['title']}")
            fig = axes.get_figure()
            fig.savefig(f"{kwargs['path']}_{i}_step.png")
            plt.close(fig)

    # FIXME: CHANGE API
    def visualize_location(self, xx, yy, xy_size, **kwargs):
        """plot grid location distribution. Origin point is in DownLeft.
        3 reservered key in kwargs:
        * ``fig_name``: savefig
        * ``xlabel``: xlabel
        * ``ylabel``

        :param xx: scatter plot x
        :type xx: ndarray
        :param yy: scatter plot y
        :type yy: ndarray
        :param xy_size: at position (x,y) the number of particles
        :type xy_size: ndarray
        """
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        # FIXME: remove the comments. use the quickplot to refactor code here
        with sns.axes_style("whitegrid"):
            axes = sns.scatterplot(x=xx, y=yy, size=xy_size,
                                   sizes=(0, np.sqrt(np.max(xy_size)) /
                                          np.sqrt(np.sum(xy_size)) * 5000),
                                   palette=cmap)

            axes.set_xlim([0 - 1, self.d_col + 1])
            axes.set_ylim([0 - 1, self.d_row + 1])

            axes.set_xticks(np.arange(1, self.d_col + 1, 3))
            axes.set_xticks(np.arange(1, self.d_col + 1), minor=True)
            axes.set_yticks(np.arange(1, self.d_row + 1, 3))
            axes.set_yticks(np.arange(1, self.d_col + 1), minor=True)
            axes.set_yticklabels([])
            axes.set_xticklabels([])

            if "ylabel" in kwargs:
                axes.set_ylabel(kwargs['ylabel'])
            if "xlabel" in kwargs:
                axes.set_xlabel(kwargs['xlabel'])

            axes.legend_.remove()
            axes.grid(which='both')
            axes.grid(which='minor', alpha=0.2)
            axes.grid(which='major', alpha=0.5)
            axes.yaxis.label.set_size(32)
            axes.xaxis.label.set_size(32)
            plt.savefig(kwargs["fig_name"], pad_inches=-4)
            plt.close()

    def migration(self, data, **kwargs):
        """draw migration figure, heat map distribution
        3 reservered key in kwargs:
        * ``fig_name``: savefig
        * ``xlabel``: xlabel
        * ``ylabel``

        :param data: [i,j] record i-th particle position at j timestamp
        """
        _, time_length = data.shape
        for i in range(time_length):
            locations = data[:, i]  # i timestamp distribution
            bins, _ = np.histogram(locations, np.arange(self.states_num + 1))

            if 'ylabel' in kwargs:
                kwargs['ylabel'] = f"t={i}"
            kwargs['fig_name'] = f"{kwargs['path']}_{i}_step.png"

            self.visualize_map_bins(bins, **kwargs)

    def visualize_map_bins(self, bins, **kwargs):
        """converts the statistics bins data to the map figure.
        3 reservered key in kwargs:
        * ``fig_name``: savefig
        * ``xlabel``: xlabel
        * ``ylabel``

        :param bins: every element represents how many particles in the cell
        :type bins: list or ndarray
        """
        xx = []
        yy = []
        xy_cnt = []
        for xy, cnt in enumerate(bins):
            if cnt > 0:
                row, col = self.ind2rowcol(xy)
                xx.append(col)
                yy.append(row)
                xy_cnt.append(int(cnt))
        self.visualize_location(xx, yy, xy_cnt, **kwargs)

    def ind2rowcol(self, index):
        index = np.array(index).astype(np.int64)
        row = (index / self.d_col).astype(np.int64)
        col = index % self.d_col
        return row, col

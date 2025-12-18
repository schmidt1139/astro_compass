import os
import pickle

import matplotlib.pyplot as plt

from astro_compass.utils.path_utils import DATA_ROOT, PLOT_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotter


class EphemPlotterExtend(EphemPlotter):
    def __init__(self, directory, num_ephems=1):
        files = os.listdir(directory)
        ephems = []
        for file in files[:num_ephems]:
            path_to_ephemeris = os.path.join(directory, file)
            with open(path_to_ephemeris, "rb") as f:
                eph = pickle.load(f)
            ephems.append(eph)
        self.ephems = ephems

    def plot(self):
        for i, ephem in enumerate(self.ephems):
            new_fig = True if i == 0 else False
            self.ephem = ephem
            fig_xy = self.plot_xy(new_fig=new_fig)
            fig_xy = self.adjust_plot_limits()

        # limit the legend to the first 4 entries
        legend = fig_xy.axes[0].get_legend()
        handles = legend.legend_handles[:4]
        labels = [handle.get_label() for handle in handles]
        fig_xy.axes[0].legend(handles, labels)

        plot_title = fig_xy.axes[0].get_title().replace(" ", "_").lower()
        file_path = os.path.join(PLOT_ROOT, f"{plot_title}.png")
        fig_xy.savefig(file_path, dpi=300)


def main():
    directory = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")

    vis = EphemPlotterExtend(directory, num_ephems=5)

    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()

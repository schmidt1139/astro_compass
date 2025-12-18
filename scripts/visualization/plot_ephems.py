import os
import pickle

import matplotlib.pyplot as plt

from astro_compass.utils.path_utils import DATA_ROOT, PLOT_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotter, plot_overlay_ballistic_orbit


class EphemPlotterExtend(EphemPlotter):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def plot(self):
        fig_xy = self.plot_xy()

        fig_xy = plot_overlay_ballistic_orbit(
            self.ephem.arr_x[0],
            self.ephem.arr_y[0],
            self.ephem.arr_vx[0],
            self.ephem.arr_vy[0],
            "Initial Orbit",
            color_in="lime",
        )

        fig_xy = plot_overlay_ballistic_orbit(
            self.ephem.arr_x[-1],
            self.ephem.arr_y[-1],
            self.ephem.arr_vx[-1],
            self.ephem.arr_vy[-1],
            "Final Orbit",
            color_in="red",
        )
        x, y = self.ephem.arr_x_target[-1], self.ephem.arr_y_target[-1]
        fig_xy = self.add_target_icon(x, y)
        fig_xy = self.adjust_plot_limits()

        plot_title = fig_xy.axes[0].get_title().replace(" ", "_").lower()

        file_path = os.path.join(PLOT_ROOT, f"{plot_title}.png")
        fig_xy.savefig(file_path, dpi=300)


def main(path_to_ephemeris):
    with open(path_to_ephemeris, "rb") as f:
        eph = pickle.load(f)

    # Plot the ephemeris
    vis = EphemPlotterExtend(eph)

    vis.plot()

    # Save the current XY plot to a file in the specified directory
    figs = vis.plot_all_ephemeris_data(flag_show=False)
    for i, fig in enumerate(figs):
        plot_title = fig.axes[0].get_title().replace(" ", "_").lower()
        file_path = os.path.join(PLOT_ROOT, f"{plot_title}.png")
        fig.savefig(file_path, dpi=300)

    plt.show()

    # report the final vector
    index = eph.num_vectors - 1
    state_vector = eph.get_vector_at_index(index)
    print("Final vector: ", state_vector)


if __name__ == "__main__":
    directory = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")
    path_to_ephemeris = os.path.join(directory, "8505947.pkl")

    main(path_to_ephemeris)

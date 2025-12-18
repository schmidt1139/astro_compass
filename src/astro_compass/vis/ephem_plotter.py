import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from astro_compass.constants.constants import Constants
from astro_compass.utils.path_utils import RUNS_ROOT


def plot_overlay_ballistic_orbit(
    x,
    y,
    vx,
    vy,
    label,
    color_in="lime",
):
    def dyn(t, x):
        mu = Constants.MU_SUN_M
        rx = x[0]
        ry = x[1]
        vx = x[2]
        vy = x[3]
        r = np.sqrt(rx**2 + ry**2)
        ax = -mu * rx / r**3
        ay = -mu * ry / r**3
        return [vx, vy, ax, ay]

    x_vec = [x, y, vx, vy]
    sol = solve_ivp(
        dyn,
        [0, Constants.YEARS_TO_SEC * 10.0],
        x_vec,
        rtol=1e-9,
        atol=1e-9,
        method="DOP853",
        max_step=Constants.DAYS_TO_SEC,
    )

    # Convert to AU

    # Convert arrays to AU before plotting
    plt.gca().plot(
        sol.y[0, :] / Constants.SMA_EARTH,
        sol.y[1, :] / Constants.SMA_EARTH,
        label=label,
        color=color_in,
    )
    plt.gca().legend()  # Update legend to include the new plot

    return


class EphemPlotter:
    def __init__(self, ephem):
        self.ephem = ephem

    def plot_xy(
        self,
        radius_central_body=Constants.RADIUS_SUN_M,
        plot_label="Trajectory",
        color_in="#5c01a6",
        new_fig=True,
    ):
        # plt.style.use("data/support_files/dark_scientific.mplstyle");

        # Convert all data to AU at the start
        scale = Constants.SMA_EARTH
        x_au = self.ephem.arr_x / scale
        y_au = self.ephem.arr_y / scale
        radius_cb_au = radius_central_body / scale

        arr_x_cb = np.array([])
        arr_y_cb = np.array([])

        max_x = max(abs(x_au))
        max_y = max(abs(y_au))

        max_lim = 1.1 * max([max_x, max_y])

        pts = 1000

        # plot central body
        for i in range(0, pts):
            theta = 2 * np.pi * i / pts
            x_cb = radius_cb_au * np.cos(theta)
            y_cb = radius_cb_au * np.sin(theta)

            arr_x_cb = np.append(arr_x_cb, x_cb)
            arr_y_cb = np.append(arr_y_cb, y_cb)

        if new_fig:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.gcf(), plt.gca()

        ax.set_aspect("equal")

        # Get initial and final states in AU
        x0 = x_au[0]
        y0 = y_au[0]
        xf = x_au[-1]
        yf = y_au[-1]

        if plt.rcParams["figure.facecolor"] == "black":
            markerfacecolor_in = color_in
            markeredgecolor_in = "white"
            background_color = "black"
        else:
            markerfacecolor_in = color_in
            markeredgecolor_in = "black"
            background_color = "white"

        ax.plot(
            x0,
            y0,
            label="Initial State",
            marker="o",
            color=background_color,
            linestyle=None,
            markerfacecolor="none",
            markeredgecolor=markeredgecolor_in,
            markersize=8,
        )
        ax.plot(
            xf,
            yf,
            label="Final State",
            marker="x",
            linestyle=None,
            markerfacecolor=markerfacecolor_in,
            markeredgecolor=markeredgecolor_in,
            color=background_color,
            markersize=8,
        )
        ax.plot(x_au, y_au, label="Trajectory", color=color_in)

        if radius_cb_au > 0.1 * max_lim:
            ax.plot(
                arr_x_cb, arr_y_cb, label="Central Body", linewidth=4, color="#f0f921"
            )
        else:
            ax.plot(
                arr_x_cb,
                arr_y_cb,
                label="Central Body",
                color=background_color,
                markerfacecolor="#f0f921",
                linestyle=None,
                marker="o",
                markersize=8,
            )

        ax.set_title("Trajectory")
        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_xlim([-max_lim, max_lim])
        ax.set_ylim([-max_lim, max_lim])
        ax.legend()
        ax.grid(False)

        self.ephem.fig_xy = fig
        self.ephem.ax_xy = ax

        return fig

    def plot_xy_ref_orbit(self, orbit_sma, label, color_in="lime"):
        fig = self.ephem.fig_xy
        ax = self.ephem.ax_xy

        arr_x_ref = np.array([])
        arr_y_ref = np.array([])

        pts = 1000

        # plot central body
        for i in range(0, pts):
            theta = 2 * np.pi * i / pts
            x_ref = orbit_sma * np.cos(theta)
            y_ref = orbit_sma * np.sin(theta)

            arr_x_ref = np.append(arr_x_ref, x_ref)
            arr_y_ref = np.append(arr_y_ref, y_ref)

        max_x_ref = max(arr_x_ref)
        max_y_ref = max(arr_y_ref)
        max_ref_val = max([max_x_ref, max_y_ref])
        plot_lim_ref = 1.1 * max_ref_val

        ax.plot(arr_x_ref, arr_y_ref, label=label, linestyle="dashed", color=color_in)
        ax.legend(loc="upper left")

        self.ephem.fig_xy = fig
        self.ephem.ax_xy = ax

        # adjust limits if necessary
        if max(ax.get_xlim()) < plot_lim_ref:
            ax.set_xlim([-plot_lim_ref, plot_lim_ref])
            ax.set_ylim([-plot_lim_ref, plot_lim_ref])

        return self.ephem.fig_xy

    def plot_all_ephemeris_data(self, flag_show=True):
        plt.style.use("data/support_files/dark_scientific.mplstyle")

        figs = []

        fig, ax = plt.subplots()
        ax.plot(self.ephem.arr_et, self.ephem.arr_m, label="Spacecraft Mass")
        ax.set_title("Spacecraft Mass over Time")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Mass [kg]")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plt.show()
        figs.append(fig)

        fig, ax = plt.subplots()
        ax.plot(self.ephem.arr_et, self.ephem.arr_alpha_x, label=r"$\alpha_x$")
        ax.plot(self.ephem.arr_et, self.ephem.arr_alpha_y, label=r"$\alpha_y$")
        ax.set_title("Spacecraft Thrust Direction Unit Vector")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Unit Vector Component Magnitude")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plt.show()
        figs.append(fig)

        fig, ax = plt.subplots()
        ax.plot(self.ephem.arr_et, self.ephem.arr_u, label="Spacecraft Throttle")
        ax.set_title("Spacecraft Throttle")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Throttle")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plt.show()
        figs.append(fig)

        return figs

    def adjust_plot_limits(self):
        # Adjust the plot limits of the existing XY plot based on current data
        fig = self.ephem.fig_xy
        ax = self.ephem.ax_xy

        # Convert to AU
        scale = Constants.SMA_EARTH
        max_x = max(abs(self.ephem.arr_x / scale))
        max_y = max(abs(self.ephem.arr_y / scale))

        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            max_x_line = max(abs(x_data))
            max_y_line = max(abs(y_data))
            if max_x_line > max_x:
                max_x = max_x_line
            if max_y_line > max_y:
                max_y = max_y_line

        max_lim = 1.1 * max([max_x, max_y])

        ax.set_xlim([-max_lim, max_lim])
        ax.set_ylim([-max_lim, max_lim])

        self.ephem.fig_xy = fig
        self.ephem.ax_xy = ax

        return self.ephem.fig_xy

    def add_target_icon(
        self, x_target, y_target, marker_in="+", color_in="red", size_in=12
    ):
        # Add a target icon to the existing XY plot
        fig = self.ephem.fig_xy
        ax = self.ephem.ax_xy

        # Convert to AU
        scale = Constants.SMA_EARTH

        ax.plot(
            x_target / scale,
            y_target / scale,
            label="Target",
            marker=marker_in,
            linestyle=None,
            color=color_in,
            markersize=size_in,
        )
        ax.legend()  # Update legend to include the new plot

        self.ephem.fig_xy = fig
        self.ephem.ax_xy = ax

        return self.ephem.fig_xy


def import_ephem(model_path, idx=0):
    rollouts_dir = os.path.join(model_path, "rollouts", f"rollout_data_{idx}.pkl")
    with open(rollouts_dir, "rb") as f:
        rollout_data, eph = pickle.load(f)
    return eph


def main():
    directory = os.path.join(RUNS_ROOT, "20251214_175440")
    ephem = import_ephem(directory, idx=0)

    vis = EphemPlotter(ephem)
    figs = vis.plot_all_ephemeris_data()

    plt.show()


if __name__ == "__main__":
    main()

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from astro_compass.constants.constants import Constants
from astro_compass.utils.path_utils import RUNS_ROOT


def plot_overlay_ballistic_orbit(
    x, y, vx, vy, env, fig, params, vis, label_in, color_in="lime"
):
    # check env type
    if params.get("env_type", "TwoBodyRendezvous_Env") == "TwoBodyRendezvous_Polar_Env":
        flag_use_obs = False
        state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]
    elif (
        params.get("env_type", "TwoBodyRendezvous_Env")
        == "TwoBodyRendezvous_Polar_Env2"
    ):
        flag_use_obs = False
        state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]
    elif (
        params.get("env_type", "TwoBodyRendezvous_Env")
        == "TwoBody_Orb2Orb_Transfer_Env_target"
    ):
        flag_use_obs = False
        state_in = [x, y, vx, vy, 1000.0, Constants.SMA_EARTH, 0.001, 0.0]
    else:
        flag_use_obs = True
        state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]

    obs, info = env.reset()

    state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]

    unwrapped_env = env.unwrapped
    obs, info = unwrapped_env.set_state(state_in)

    T = info["orbital_period_years"] * Constants.YEARS_TO_SEC

    time = 0.0
    flag_done = False
    arr_x = []
    arr_y = []
    max_steps = 10000  # Safety limit to prevent infinite loop
    step_count = 0

    while not flag_done and step_count < max_steps:
        obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.0])

        if flag_use_obs:
            obs = obs
            # dim state
            x_i = obs[0] * params["l_star"]
            y_i = obs[1] * params["l_star"]
        else:
            unwrapped_env = env.unwrapped
            obs = unwrapped_env.get_cartesian_state()
            x_i = obs[0]
            y_i = obs[1]

        if info["Elapsed time"] >= T or done or truncated:
            flag_done = True

        arr_x.append(x_i)
        arr_y.append(y_i)
        step_count += 1

    fig = vis.overlay_ref_orbit(
        ephem=None,
        label=label_in,
        color_in=color_in,
        arr_x=arr_x,
        arr_y=arr_y,
    )

    return fig


class EphemPlotter:
    def __init__(self, ephem):
        self.ephem = ephem

    def plot_xy(
        self,
        radius_central_body=Constants.RADIUS_SUN_M,
        plot_label="Trajectory",
        color_in="#5c01a6",
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

        fig, ax = plt.subplots(figsize=(6, 6))

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

    def overlay_ref_orbit(self, ephem, label, color_in="lime", arr_x=None, arr_y=None):
        # Overlay a reference Keplerian orbit on the existing XY plot
        fig = self.ephem.fig_xy
        ax = self.ephem.ax_xy
        flag_xy_exists = False

        # Convert to AU
        scale = Constants.SMA_EARTH

        if arr_x is None or arr_y is None:
            arr_x = np.array([])
            arr_y = np.array([])
            num_vecs = ephem.num_vectors
        else:
            arr_x = arr_x
            arr_y = arr_y
            num_vecs = len(arr_x)
            flag_xy_exists = True

        for i in range(0, num_vecs):
            if not flag_xy_exists:
                x = ephem.arr_x[i]
                y = ephem.arr_y[i]
            else:
                x = arr_x[i]
                y = arr_y[i]

            arr_x = np.append(arr_x, x)
            arr_y = np.append(arr_y, y)

        # Convert arrays to AU before plotting
        ax.plot(arr_x / scale, arr_y / scale, label=label, color=color_in)
        ax.legend()  # Update legend to include the new plot

        self.ephem.fig_xy = fig
        self.ephem.ax_xy = ax

        return self.ephem.fig_xy

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

    def compare_trajectories(
        self, other_ephem, position_tol=1e-12, velocity_tol=1e-6, verbose=False
    ):
        # Compare this ephemeris trajectory to another ephemeris trajectory
        # Returns True if all corresponding states are within the specified tolerances

        if self.ephem.num_vectors != other_ephem.num_vectors:
            if verbose == True:
                print(
                    f"Different number of vectors: {self.ephem.num_vectors} vs {other_ephem.num_vectors}"
                )
            return False  # Different number of vectors

        for i in range(self.ephem.num_vectors):
            dx = abs(self.ephem.arr_x[i] - other_ephem.arr_x[i])
            dy = abs(self.ephem.arr_y[i] - other_ephem.arr_y[i])
            dvx = abs(self.ephem.arr_vx[i] - other_ephem.arr_vx[i])
            dvy = abs(self.ephem.arr_vy[i] - other_ephem.arr_vy[i])

            if (
                dx > position_tol
                or dy > position_tol
                or dvx > velocity_tol
                or dvy > velocity_tol
            ):
                if verbose == True:
                    print(
                        f"Difference at index {i}: x={self.ephem.arr_x[i]}, y={self.ephem.arr_y[i]}, vx={self.ephem.arr_vx[i]}, vy={self.ephem.arr_vy[i]}"
                    )
                    print(
                        f"                 vs x={other_ephem.arr_x[i]}, y={other_ephem.arr_y[i]}, vx={other_ephem.arr_vx[i]}, vy={other_ephem.arr_vy[i]}"
                    )
                return False  # States differ beyond tolerances

        return True  # All states are within tolerances

    def save_plots(self, directory_path, file_tag, params, env):
        # Save the current XY plot to a file in the specified directory
        figs = self.plot_all_ephemeris_data(flag_show=False)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for i, fig in enumerate(figs):
            plot_title = fig.axes[0].get_title().replace(" ", "_").lower()
            # Normalize path to handle platform differences and invalid characters
            file_path = os.path.normpath(
                os.path.join(directory_path, f"{file_tag}_{plot_title}.png")
            )
            fig.savefig(file_path, dpi=300)

        fig_xy = self.plot_xy()
        x, y, vx, vy = (
            self.ephem.arr_x[0],
            self.ephem.arr_y[0],
            self.ephem.arr_vx[0],
            self.ephem.arr_vy[0],
        )
        fig_xy = plot_overlay_ballistic_orbit(
            x, y, vx, vy, env, fig_xy, params, self, "Initial Orbit", color_in="lime"
        )
        x, y, vx, vy = (
            self.ephem.arr_x[-1],
            self.ephem.arr_y[-1],
            self.ephem.arr_vx[-1],
            self.ephem.arr_vy[-1],
        )
        fig_xy = plot_overlay_ballistic_orbit(
            x, y, vx, vy, env, fig_xy, params, self, "Final Orbit", color_in="red"
        )
        x, y = self.ephem.arr_x_target[-1], self.ephem.arr_y_target[-1]
        fig_xy = self.add_target_icon(x, y)
        fig_xy = self.adjust_plot_limits()

        plot_title = fig_xy.axes[0].get_title().replace(" ", "_").lower()
        # Normalize path to handle platform differences and invalid characters
        file_path = os.path.normpath(
            os.path.join(directory_path, f"{file_tag}_{plot_title}.png")
        )
        fig_xy.savefig(file_path, dpi=300)


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

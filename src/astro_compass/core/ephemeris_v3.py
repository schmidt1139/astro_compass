import os
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plot
import numpy as np

from astro_compass.constants.constants import Constants
from astro_compass.utils.plotting_utils import plot_overlay_ballistic_orbit


class Ephemeris_v3:
    def reset(self):
        self.arr_et = np.array([])
        self.arr_x = np.array([])
        self.arr_y = np.array([])
        self.arr_vx = np.array([])
        self.arr_vy = np.array([])
        self.arr_m = np.array([])
        self.arr_x_target = np.array([])
        self.arr_y_target = np.array([])
        self.arr_vx_target = np.array([])
        self.arr_vy_target = np.array([])
        self.arr_TTG = np.array([])
        self.arr_alpha_x = np.array([])
        self.arr_alpha_y = np.array([])
        self.arr_u = np.array([])
        self.num_vectors = 0

    def __init__(self):
        # initialize an empty ephemeris object
        self.reset()
        self.version = 3.0

    def add_data(
        self,
        et,
        x,
        y,
        vx,
        vy,
        m,
        target_x,
        target_y,
        target_vx,
        target_vy,
        TTG,
        alpha_x=0.0,
        alpha_y=0.0,
        u=0.0,
    ):
        self.arr_et = np.append(self.arr_et, et)
        self.arr_x = np.append(self.arr_x, x)
        self.arr_y = np.append(self.arr_y, y)
        self.arr_vx = np.append(self.arr_vx, vx)
        self.arr_vy = np.append(self.arr_vy, vy)
        self.arr_m = np.append(self.arr_m, m)
        self.arr_x_target = np.append(self.arr_x_target, target_x)
        self.arr_y_target = np.append(self.arr_y_target, target_y)
        self.arr_vx_target = np.append(self.arr_vx_target, target_vx)
        self.arr_vy_target = np.append(self.arr_vy_target, target_vy)
        self.arr_TTG = np.append(self.arr_TTG, TTG)
        self.arr_alpha_x = np.append(self.arr_alpha_x, alpha_x)
        self.arr_alpha_y = np.append(self.arr_alpha_y, alpha_y)
        self.arr_u = np.append(self.arr_u, u)
        self.num_vectors = self.num_vectors + 1

    def plot_xy(
        self,
        radius_central_body=Constants.RADIUS_SUN_M,
        plot_label="Trajectory",
        color_in="#5c01a6",
    ):
        # plot.style.use("data/support_files/dark_scientific.mplstyle");

        # Convert all data to AU at the start
        scale = Constants.SMA_EARTH
        x_au = self.arr_x / scale
        y_au = self.arr_y / scale
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

        fig, ax = plot.subplots(figsize=(6, 6))

        ax.set_aspect("equal")

        # Get initial and final states in AU
        x0 = x_au[0]
        y0 = y_au[0]
        xf = x_au[-1]
        yf = y_au[-1]

        if plot.rcParams["figure.facecolor"] == "black":
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

        self.fig_xy = fig
        self.ax_xy = ax

        return fig

    def plot_xy_ref_orbit(self, orbit_sma, label, color_in="lime"):
        fig = self.fig_xy
        ax = self.ax_xy

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

        self.fig_xy = fig
        self.ax_xy = ax

        # adjust limits if necessary
        if max(ax.get_xlim()) < plot_lim_ref:
            ax.set_xlim([-plot_lim_ref, plot_lim_ref])
            ax.set_ylim([-plot_lim_ref, plot_lim_ref])

        return self.fig_xy

    def plot_all_ephemeris_data(self, flag_show=True):
        plot.style.use("data/support_files/dark_scientific.mplstyle")

        figs = []

        fig, ax = plot.subplots()
        ax.plot(self.arr_et, self.arr_m, label="Spacecraft Mass")
        ax.set_title("Spacecraft Mass over Time")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Mass [kg]")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plot.show()
        figs.append(fig)

        fig, ax = plot.subplots()
        ax.plot(self.arr_et, self.arr_alpha_x, label=r"$\alpha_x$")
        ax.plot(self.arr_et, self.arr_alpha_y, label=r"$\alpha_y$")
        ax.set_title("Spacecraft Thrust Direction Unit Vector")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Unit Vector Component Magnitude")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plot.show()
        figs.append(fig)

        fig, ax = plot.subplots()
        ax.plot(self.arr_et, self.arr_u, label="Spacecraft Throttle")
        ax.set_title("Spacecraft Throttle")
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Throttle")
        fig.tight_layout()
        ax.legend(loc="lower right")
        if flag_show:
            plot.show()
        figs.append(fig)

        return figs

    def write_to_file(self, file_path, mod_vector_write_frequency=1):
        file_name_base = os.path.basename(file_path)

        # Get generation time as UTC string
        time_generation = time.time()
        string_time_generation_utc = datetime.fromtimestamp(
            time_generation, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S.%f")

        # Modified number of vectors
        mod_num_vec = self.num_vectors // mod_vector_write_frequency

        with open(file_path, "w") as f:
            header = (
                "Astro Compass Ephemeris v 1.0\n"
                f"Version: 3.0\n"
                f"File name: {file_name_base}\n"
                f"Generation time (UTC): {string_time_generation_utc}\n"
                f"Number of vectors: {mod_num_vec}\n"
                "\n"
                "Columns\n"
                "1: Elapsed time [units: seconds]\n"
                "2: X position [units: meters]\n"
                "3: Y position [units: meters]\n"
                "4: VX position [units: meters/second]\n"
                "5: VY position [units: meters/second]\n"
                "6: Mass [units: kg]\n"
                "7: Target X position [units: meters]\n"
                "8: Target Y position [units: meters]\n"
                "9: Target VX position [units: meters/second]\n"
                "10: Target VY position [units: meters/second]\n"
                "11: Time to Go [units: seconds]\n"
                "12: Thrust Direction - X-hat [units: none]\n"
                "13: Thrust Direction - Y-hat [units: none]\n"
                "14: Thrust Throttle (ranges from 0-1) [units: none]\n"
                "\n"
                "<Ephemeris Start>\n"
            )

            f.write(header)

            for i in range(0, self.num_vectors):
                modulo = i % mod_vector_write_frequency

                if modulo == 0:
                    str_ephem_out = (
                        f"{self.arr_et[i]: .16e},"
                        f"{self.arr_x[i]: .16e},"
                        f"{self.arr_y[i]: .16e},"
                        f"{self.arr_vx[i]: .16e},"
                        f"{self.arr_vy[i]: .16e},"
                        f"{self.arr_m[i]: .16e},"
                        f"{self.arr_x_target[i]: .16e},"
                        f"{self.arr_y_target[i]: .16e},"
                        f"{self.arr_vx_target[i]: .16e},"
                        f"{self.arr_vy_target[i]: .16e},"
                        f"{self.arr_TTG[i]: .16e},"
                        f"{self.arr_alpha_x[i]: .16e},"
                        f"{self.arr_alpha_y[i]: .16e},"
                        f"{self.arr_u[i]: .16e}"
                    )

                    f.write(str_ephem_out + "\n")

            f.write("<Ephemeris End>\n")

        f.close()

        return f.closed

    def read_from_file(self, file_path):
        # clear the ephemeris states
        self.reset()

        # flag if data section has been reached
        flag_ephem_start = False

        # read in the lines from the file
        with open(file_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()

            # if the ephemeris has started, split the file contents
            # add the data to the ephemeris object
            if flag_ephem_start and line != "<Ephemeris End>":
                line_contents = line.split(",")
                ephem_data = [float(x) for x in line_contents]

                # unpack the data
                et = ephem_data[0]  # elapsed seconds
                x = ephem_data[1]  # x position [km]
                y = ephem_data[2]  # y position [km]
                vx = ephem_data[3]  # x velocity [km/s]
                vy = ephem_data[4]  # y velocity [km/s]
                m = ephem_data[5]  # mass kg
                x_target = ephem_data[6]  # x target [km]
                y_target = ephem_data[7]  # y target [km]
                vx_target = ephem_data[8]  # vx target [km/s]
                vy_target = ephem_data[9]  # vy target [km/s]
                TTG = ephem_data[10]  # time to go [s]
                alpha_x = ephem_data[11]  # thrust unit vec - x
                alpha_y = ephem_data[12]  # thrust unit vec - y
                u = ephem_data[13]  # throttle

                self.add_data(
                    et,
                    x,
                    y,
                    vx,
                    vy,
                    m,
                    x_target,
                    y_target,
                    vx_target,
                    vy_target,
                    TTG,
                    alpha_x,
                    alpha_y,
                    u,
                )

            elif line == "<Ephemeris End>":
                break

            # check if the ephemeris data section has been reached
            if line == "<Ephemeris Start>":
                flag_ephem_start = True

    def get_vector_at_index(self, index):
        # extract the vector elements at index
        et = self.arr_et[index]
        x = self.arr_x[index]
        y = self.arr_y[index]
        vx = self.arr_vx[index]
        vy = self.arr_vy[index]
        m = self.arr_m[index]
        x_target = self.arr_x_target[index]
        y_target = self.arr_y_target[index]
        vx_target = self.arr_vx_target[index]
        vy_target = self.arr_vy_target[index]
        TTG = self.arr_TTG[index]
        alpha_x = self.arr_alpha_x[index]
        alpha_y = self.arr_alpha_y[index]
        u = self.arr_u[index]

        # construct output vector
        vector = np.array(
            [
                et,
                x,
                y,
                vx,
                vy,
                m,
                x_target,
                y_target,
                vx_target,
                vy_target,
                TTG,
                alpha_x,
                alpha_y,
                u,
            ]
        )

        return vector

    def overlay_ref_orbit(self, ephem, label, color_in="lime", arr_x=None, arr_y=None):
        # Overlay a reference Keplerian orbit on the existing XY plot
        fig = self.fig_xy
        ax = self.ax_xy
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

        self.fig_xy = fig
        self.ax_xy = ax

        return self.fig_xy

    def adjust_plot_limits(self):
        # Adjust the plot limits of the existing XY plot based on current data
        fig = self.fig_xy
        ax = self.ax_xy

        # Convert to AU
        scale = Constants.SMA_EARTH
        max_x = max(abs(self.arr_x / scale))
        max_y = max(abs(self.arr_y / scale))

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

        self.fig_xy = fig
        self.ax_xy = ax

        return self.fig_xy

    def add_target_icon(
        self, x_target, y_target, marker_in="+", color_in="red", size_in=12
    ):
        # Add a target icon to the existing XY plot
        fig = self.fig_xy
        ax = self.ax_xy

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

        self.fig_xy = fig
        self.ax_xy = ax

        return self.fig_xy

    def compare_trajectories(
        self, other_ephem, position_tol=1e-12, velocity_tol=1e-6, verbose=False
    ):
        # Compare this ephemeris trajectory to another ephemeris trajectory
        # Returns True if all corresponding states are within the specified tolerances

        if self.num_vectors != other_ephem.num_vectors:
            if verbose == True:
                print(
                    f"Different number of vectors: {self.num_vectors} vs {other_ephem.num_vectors}"
                )
            return False  # Different number of vectors

        for i in range(self.num_vectors):
            dx = abs(self.arr_x[i] - other_ephem.arr_x[i])
            dy = abs(self.arr_y[i] - other_ephem.arr_y[i])
            dvx = abs(self.arr_vx[i] - other_ephem.arr_vx[i])
            dvy = abs(self.arr_vy[i] - other_ephem.arr_vy[i])

            if (
                dx > position_tol
                or dy > position_tol
                or dvx > velocity_tol
                or dvy > velocity_tol
            ):
                if verbose == True:
                    print(
                        f"Difference at index {i}: x={self.arr_x[i]}, y={self.arr_y[i]}, vx={self.arr_vx[i]}, vy={self.arr_vy[i]}"
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
        x, y, vx, vy = self.arr_x[0], self.arr_y[0], self.arr_vx[0], self.arr_vy[0]
        fig_xy = plot_overlay_ballistic_orbit(
            x, y, vx, vy, env, fig_xy, params, self, "Initial Orbit", color_in="lime"
        )
        x, y, vx, vy = self.arr_x[-1], self.arr_y[-1], self.arr_vx[-1], self.arr_vy[-1]
        fig_xy = plot_overlay_ballistic_orbit(
            x, y, vx, vy, env, fig_xy, params, self, "Final Orbit", color_in="red"
        )
        x, y = self.arr_x_target[-1], self.arr_y_target[-1]
        fig_xy = self.add_target_icon(x, y)
        fig_xy = self.adjust_plot_limits()

        plot_title = fig_xy.axes[0].get_title().replace(" ", "_").lower()
        # Normalize path to handle platform differences and invalid characters
        file_path = os.path.normpath(
            os.path.join(directory_path, f"{file_tag}_{plot_title}.png")
        )
        fig_xy.savefig(file_path, dpi=300)

    def convert_from_v2(self, ephem_v2):
        # Convert from an Ephemeris_v2 object to this Ephemeris_v3 object
        self.reset()

        for i in range(ephem_v2.num_vectors):
            et = ephem_v2.arr_et[i]
            x = ephem_v2.arr_x[i]
            y = ephem_v2.arr_y[i]
            vx = ephem_v2.arr_vx[i]
            vy = ephem_v2.arr_vy[i]
            m = ephem_v2.arr_m[i]
            x_target = ephem_v2.arr_x_target[i]
            y_target = ephem_v2.arr_y_target[i]
            vx_target = ephem_v2.arr_vx_target[i]
            vy_target = ephem_v2.arr_vy_target[i]
            TTG = ephem_v2.arr_TTG[i]
            alpha_x = ephem_v2.arr_alpha_x[i]
            alpha_y = ephem_v2.arr_alpha_y[i]
            u = ephem_v2.arr_u[i]

            self.add_data(
                et,
                x,
                y,
                vx,
                vy,
                m,
                x_target,
                y_target,
                vx_target,
                vy_target,
                TTG,
                alpha_x,
                alpha_y,
                u,
            )

    def get_interpolated_vector_at_time(self, next_t):
        # find the index of the next time
        next_index = np.where(self.arr_et >= next_t)[0][0]

        # perform linear interpolation
        if next_index == 0:
            return self.get_vector_at_index(0)
        else:
            t1 = self.arr_et[next_index - 1]
            t2 = self.arr_et[next_index]
            ratio = (next_t - t1) / (t2 - t1)

            vec1 = self.get_vector_at_index(next_index - 1)
            vec2 = self.get_vector_at_index(next_index)

            interpolated_vec = vec1 + ratio * (vec2 - vec1)

            return interpolated_vec

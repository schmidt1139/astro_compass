import numpy as np
import matplotlib.pyplot as plot
import os
import time
from constants.constants import Constants
from datetime import datetime, timezone


class Ephemeris:
    def reset(self):
        self.arr_et = np.array([])
        self.arr_x = np.array([])
        self.arr_y = np.array([])
        self.arr_vx = np.array([])
        self.arr_vy = np.array([])
        self.arr_m = np.array([])
        self.arr_alpha_x = np.array([])
        self.arr_alpha_y = np.array([])
        self.arr_u = np.array([])
        self.num_vectors = 0

    def __init__(self):
        # initialize an empty ephemeris object
        self.reset()

    def add_data(self, et, x, y, vx, vy, m, alpha_x=0.0, alpha_y=0.0, u=0.0):
        self.arr_et = np.append(self.arr_et, et)
        self.arr_x = np.append(self.arr_x, x)
        self.arr_y = np.append(self.arr_y, y)
        self.arr_vx = np.append(self.arr_vx, vx)
        self.arr_vy = np.append(self.arr_vy, vy)
        self.arr_m = np.append(self.arr_m, m)
        self.arr_alpha_x = np.append(self.arr_alpha_x, alpha_x)
        self.arr_alpha_y = np.append(self.arr_alpha_y, alpha_y)
        self.arr_u = np.append(self.arr_u, u)
        self.num_vectors = self.num_vectors + 1

    def add_polar_data(
        self, et, r, theta, r_dot, v_theta, m, alpha_x=0.0, alpha_y=0.0, u=0.0
    ):
        # convert polar coordinates to cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vx = r_dot * np.cos(theta) - v_theta * r * np.sin(theta)
        vy = r_dot * np.sin(theta) + v_theta * r * np.cos(theta)

        self.arr_et = np.append(self.arr_et, et)
        self.arr_x = np.append(self.arr_x, x)
        self.arr_y = np.append(self.arr_y, y)
        self.arr_vx = np.append(self.arr_vx, vx)
        self.arr_vy = np.append(self.arr_vy, vy)
        self.arr_m = np.append(self.arr_m, m)
        self.arr_alpha_x = np.append(self.arr_alpha_x, alpha_x)
        self.arr_alpha_y = np.append(self.arr_alpha_y, alpha_y)
        self.arr_u = np.append(self.arr_u, u)
        self.num_vectors = self.num_vectors + 1

    def plot_xy(
        self, radius_central_body=Constants.RADIUS_SUN_M, plot_label="Trajectory"
    ):
        
        #plot.style.use("data/support_files/dark_scientific.mplstyle");

        arr_x_cb = np.array([])
        arr_y_cb = np.array([])

        max_x = max(abs(self.arr_x))
        max_y = max(abs(self.arr_y))

        max_lim = 1.1 * max([max_x, max_y])

        pts = 1000

        # plot central body
        for i in range(0, pts):
            theta = 2 * np.pi * i / pts
            x_cb = radius_central_body * np.cos(theta)
            y_cb = radius_central_body * np.sin(theta)

            arr_x_cb = np.append(arr_x_cb, x_cb)
            arr_y_cb = np.append(arr_y_cb, y_cb)

        fig, ax = plot.subplots(figsize=(6, 6))

        ax.set_aspect("equal")

        # Get initial and final states
        x0 = self.arr_x[0]
        y0 = self.arr_y[0]
        xf = self.arr_x[-1]
        yf = self.arr_y[-1]

        if (plot.rcParams["figure.facecolor"] == "black"):
            markerfacecolor_in = "white"
            markeredgecolor_in = "white"
            background_color = "black"
        else:
            markerfacecolor_in = "black"
            markeredgecolor_in = "black"
            background_color = "white"

        ax.plot(
            x0,
            y0,
            label="Initial State",
            marker="o",
            color=background_color,
            linestyle=None,
            markerfacecolor=markerfacecolor_in,
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
        ax.plot(self.arr_x, self.arr_y, label="Trajectory")

        if (radius_central_body > 0.1* max_lim):
            ax.plot(arr_x_cb, arr_y_cb, label="Central Body",linewidth=4, color="gold")
        else:
            ax.plot(arr_x_cb, arr_y_cb, label="Central Body", color=background_color, markerfacecolor="gold", linestyle=None, marker="o", markersize=8)

        ax.set_title("Trajectory")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
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
                "7: Thrust Direction - X-hat [units: none]\n"
                "8: Thrust Direction - Y-hat [units: none]\n"
                "9: Thrust Throttle (ranges from 0-1) [units: none]\n"
                "\n"
                "<Ephemeris Start>\n"
            )

            f.write(header)

            for i in range(0, self.num_vectors - 1):
                modulo = i % mod_vector_write_frequency

                if modulo == 0:
                    str_ephem_out = (
                        f"{self.arr_et[i]: .16e},"
                        f"{self.arr_x[i]: .16e},"
                        f"{self.arr_y[i]: .16e},"
                        f"{self.arr_vx[i]: .16e},"
                        f"{self.arr_vy[i]: .16e},"
                        f"{self.arr_m[i]: .16e},"
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
                alpha_x = ephem_data[6]  # thrust unit vec - x
                alpha_y = ephem_data[7]  # thrust unit vec - y
                u = ephem_data[8]  # throttle

                self.add_data(et, x, y, vx, vy, m, alpha_x, alpha_y, u)

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
        alpha_x = self.arr_alpha_x[index]
        alpha_y = self.arr_alpha_y[index]
        u = self.arr_u[index]

        # construct output vector
        vector = np.array([et, x, y, vx, vy, m, alpha_x, alpha_y, u])

        return vector
    
    def overlay_ref_orbit(self, ephem, label, color_in="lime"):
        # Overlay a reference Keplerian orbit on the existing XY plot
        fig = self.fig_xy
        ax = self.ax_xy

        arr_x = np.array([])
        arr_y = np.array([])

        for i in range(0, ephem.num_vectors):
            x = ephem.arr_x[i]
            y = ephem.arr_y[i]

            arr_x = np.append(arr_x, x)
            arr_y = np.append(arr_y, y)

        ax.plot(arr_x, arr_y, label=label, color=color_in)
        ax.legend()  # Update legend to include the new plot

        self.fig_xy = fig
        self.ax_xy = ax

        return self.fig_xy
    
    def adjust_plot_limits(self):
        # Adjust the plot limits of the existing XY plot based on current data
        fig = self.fig_xy
        ax = self.ax_xy


        max_x = max(abs(self.arr_x))
        max_y = max(abs(self.arr_y))

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
    
    def add_target_icon(self, x_target, y_target, marker_in="+", color_in="red", size_in=12):
        # Add a target icon to the existing XY plot
        fig = self.fig_xy
        ax = self.ax_xy

        ax.plot(
            x_target,
            y_target,
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
    
    def compare_trajectories(self, other_ephem, position_tol=1e-12, velocity_tol=1e-6):
        # Compare this ephemeris trajectory to another ephemeris trajectory
        # Returns True if all corresponding states are within the specified tolerances
        
        if self.num_vectors != other_ephem.num_vectors:
            print(f"Different number of vectors: {self.num_vectors} vs {other_ephem.num_vectors}")
            return False  # Different number of vectors

        for i in range(self.num_vectors):
            dx = abs(self.arr_x[i] - other_ephem.arr_x[i])
            dy = abs(self.arr_y[i] - other_ephem.arr_y[i])
            dvx = abs(self.arr_vx[i] - other_ephem.arr_vx[i])
            dvy = abs(self.arr_vy[i] - other_ephem.arr_vy[i])

            if dx > position_tol or dy > position_tol or dvx > velocity_tol or dvy > velocity_tol:
                print(f"Difference at index {i}: x={self.arr_x[i]}, y={self.arr_y[i]}, vx={self.arr_vx[i]}, vy={self.arr_vy[i]}")
                print(f"                 vs x={other_ephem.arr_x[i]}, y={other_ephem.arr_y[i]}, vx={other_ephem.arr_vx[i]}, vy={other_ephem.arr_vy[i]}")
                return False  # States differ beyond tolerances

        return True  # All states are within tolerances

import numpy as np
import matplotlib.pyplot as plot
import os
import time
from datetime import datetime, timezone


class Ephemeris:
    def __init__(self):
        # initialize an empty ephemeris object

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

    def add_data(self, et, x, y, vx, vy, m, alpha_x, alpha_y, u):
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

    def add_polar_data(self, et, r, theta, r_dot, v_theta, m, alpha_x, alpha_y, u):
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

    def plot_xy(self, radius_central_body=6.957e8, plot_label="Trajectory"):
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

        ax.plot(
            x0,
            y0,
            label="Initial State",
            marker="o",
            color="white",
            linestyle=None,
            markerfacecolor="blue",
            markeredgecolor="blue",
        )
        ax.plot(
            xf,
            yf,
            label="Final State",
            marker="x",
            linestyle=None,
            markerfacecolor="black",
            markeredgecolor="black",
            color="white",
        )
        ax.plot(self.arr_x, self.arr_y, label="Trajectory", color="blue")
        ax.plot(arr_x_cb, arr_y_cb, label="Central Body")

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

    def plot_xy_ref_orbit(self, orbit_sma, label):
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

        ax.plot(arr_x_ref, arr_y_ref, label=label, linestyle="dashed")
        ax.legend(loc="upper left")

        self.fig_xy = fig
        self.ax_xy = ax

        return self.fig_xy

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

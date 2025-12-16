import os
import time
from datetime import datetime, timezone

import numpy as np


class Ephemeris_v2:
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
                f"Version: 2.0\n"
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

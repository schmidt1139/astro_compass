import os
import time
from datetime import datetime, timezone

import numpy as np

from astro_compass.core.ephemeris_v2 import Ephemeris_v2


class Ephemeris_v3(Ephemeris_v2):
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
        super().__init__()
        # initialize an empty ephemeris object
        self.reset()
        self.version = 3.0

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

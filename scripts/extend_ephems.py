import os

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.core.propagation import env_EOM_TBT_v2
from astro_compass.core.training_data_generation import read_ephems_from_dir


def extend_ephems():
    num_ephems_to_use = 100_000
    ephem_version = 2.0
    extension_scale = 1.5
    path_ephems = os.path.join(
        "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\temp\\"
    )
    path_output = os.path.join(
        "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\temp_out\\"
    )

    set_ephems, filenames = read_ephems_from_dir(
        path_ephems,
        num_ephems_to_use,
        version=ephem_version,
        flag_return_filenames=True,
    )

    print("Reading ephemerides from: " + path_ephems)
    set_ephems = set_ephems[:num_ephems_to_use]
    num_ephems = len(set_ephems)
    num_states = set_ephems[0].num_vectors * num_ephems
    print("Number of ephemerides: " + str(num_ephems))
    print("Number of total state vectors: " + str(num_states))

    counter = 0

    # collect all states and actions from ephemerides
    for eph in tqdm(set_ephems, desc="Processing ephemerides"):
        try:
            eph_output = Ephemeris_v2()

            # print("\nProcessing ephemeris " + str(counter+1) + " of " + str(num_ephems) )

            # print("Filename: " + filenames[counter])

            # update ephem filename
            ephem_filename = filenames[counter].replace(".txt", "_extended.txt")

            counter += 1

            # get first state vector
            first_state_vector = eph.get_vector_at_index(0)

            # extract last state vector
            last_index = eph.num_vectors - 1
            last_state_vector = eph.get_vector_at_index(last_index)
            x0 = last_state_vector[1]
            y0 = last_state_vector[2]
            vx0 = last_state_vector[3]
            vy0 = last_state_vector[4]
            mass0 = last_state_vector[5]
            elapsed_time = last_state_vector[0] - first_state_vector[0]

            # average step size
            step_size = elapsed_time / eph.num_vectors

            # time to extend to
            extend_time = elapsed_time * extension_scale
            delta_time = extend_time - elapsed_time
            additional_steps = int(delta_time / step_size)

            t0 = 0.0
            tf = delta_time

            # step the spacecraft forward
            t_span = (t0, tf)
            y0 = np.array([x0, y0, vx0, vy0, mass0])
            params = np.array(
                [
                    Constants.MU_SUN_M,
                    1.33,
                    3872.0,
                    Constants.G0,
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            )

            t_eval = np.linspace(t_span[0], t_span[1], additional_steps)
            solution = solve_ivp(
                env_EOM_TBT_v2, t_span, y0, method="RK45", args=(params,), t_eval=t_eval
            )

            # add original ephemeris states to output
            for i in range(0, eph.num_vectors):
                state_vector = eph.get_vector_at_index(i)
                et = state_vector[0]
                x = state_vector[1]
                y = state_vector[2]
                vx = state_vector[3]
                vy = state_vector[4]
                mass = state_vector[5]
                target_x = state_vector[6]
                target_y = state_vector[7]
                target_vx = state_vector[8]
                target_vy = state_vector[9]
                ttg = state_vector[10]
                u = state_vector[13]
                alpha_x = state_vector[11]
                alpha_y = state_vector[12]

                eph_output.add_data(
                    et,
                    x,
                    y,
                    vx,
                    vy,
                    mass,
                    target_x,
                    target_y,
                    target_vx,
                    target_vy,
                    ttg,
                    alpha_x,
                    alpha_y,
                    u,
                )

            # add extended states to output
            num_new_vectors = solution.y.shape[1]
            for i in range(1, num_new_vectors):
                et = last_state_vector[0] + solution.t[i]
                x = solution.y[0, i]
                y = solution.y[1, i]
                vx = solution.y[2, i]
                vy = solution.y[3, i]
                mass = solution.y[4, i]
                target_x = last_state_vector[6]
                target_y = last_state_vector[7]
                target_vx = last_state_vector[8]
                target_vy = last_state_vector[9]
                ttg = last_state_vector[10] - solution.t[i]
                u = 0.0
                alpha_x = 0.0
                alpha_y = 0.0

                eph_output.add_data(
                    et,
                    x,
                    y,
                    vx,
                    vy,
                    mass,
                    target_x,
                    target_y,
                    target_vx,
                    target_vy,
                    ttg,
                    alpha_x,
                    alpha_y,
                    u,
                )

            # write to file
            pathout = os.path.join(path_output, ephem_filename)
            eph_output.write_to_file(pathout, mod_vector_write_frequency=1)

            # print(" Average Step Size (days): " + str(step_size/86400) )
            # print(" Ephem Duration (days): " + str(elapsed_time/86400) )
            # print(" Extended Duration (days): " + str(extend_time/86400) )
            # print(" Additional steps: " + str(additional_steps) )

        except Exception as e:
            print(f"Error processing ephemeris {filenames[counter - 1]}: {e}")
            continue


extend_ephems()

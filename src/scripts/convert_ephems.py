import sys
import os
from core.propagation import env_EOM_TBT_v2
import numpy as np
from core.ephemeris_v2 import Ephemeris_v2
from core.ephemeris_v3 import Ephemeris_v3
from scipy.integrate import solve_ivp
from core.training_data_generation import read_ephems_from_dir
from tqdm import tqdm
from constants.constants import Constants
import concurrent.futures

def convert_ephem_wrapper(args):
    ephem, path_output, filename = args
    return convert_single_ephem(ephem, path_output, filename)

def convert_single_ephem(ephem, dir_out, filename_out):

    #out ephemeris
    ephem_v3 = Ephemeris_v3()
    
    #get initial state from ephem
    first_state_vector = ephem.get_vector_at_index(0)

    # unpack initial state vector
    t0 = first_state_vector[0]

    x0 = first_state_vector[1]
    y0 = first_state_vector[2]
    vx0 = first_state_vector[3]
    vy0 = first_state_vector[4]
    mass0 = first_state_vector[5]
    target_x0 = first_state_vector[6]
    target_y0 = first_state_vector[7]
    target_vx0 = first_state_vector[8]
    target_vy0 = first_state_vector[9]
    ttg0 = first_state_vector[10]
    u0 = first_state_vector[13]
    alpha_x0 = first_state_vector[11]
    alpha_y0 = first_state_vector[12]

    # determine the target location at the rendezvous time
    arr_target_y0 = np.array([target_x0, target_y0, target_vx0, target_vy0, 1000.0])
    time_to_prop_backward = ttg0

    # set up parameter array
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
    
    t_span = (0.0, -time_to_prop_backward)

    # # solve ODE
    solution = solve_ivp(
        env_EOM_TBT_v2, t_span, arr_target_y0, method="RK45", args=(params,)
    )

    # extract final state - this is the target location at the start of the ephemeris
    y_target_0 = (solution.y[:, -1]).astype(np.float32)

    # print(" Target at start of ephem: x: " + str(y_target_0[0]) + " y: " + str(y_target_0[1]) +
    #       " vx: " + str(y_target_0[2]) + " vy: " + str(y_target_0[3]) )
    
    # updated state vector
    ephem_v3.add_data( t0, x0, y0, vx0, vy0, mass0,
                      y_target_0[0], y_target_0[1], y_target_0[2], y_target_0[3],
                      ttg0, alpha_x0, alpha_y0, u0 )
    
    # step through remaining state vectors
    num_vectors = ephem.num_vectors

    #initial time and target position
    t_prev = t0
    arr_target_prev = y_target_0

    for i in range(1, num_vectors):
        state_vector = ephem.get_vector_at_index(i)

        # unpack state vector
        t = state_vector[0]
        dt = t - t_prev

        t_span = (0.0, dt)

        # propagate target position forward to current time
        solution = solve_ivp(
            env_EOM_TBT_v2, t_span, arr_target_prev, method="RK45", args=(params,)
        )

        arr_target_current = (solution.y[:, -1]).astype(np.float32)

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

        # add to new ephem with moving target
        ephem_v3.add_data( t, x, y, vx, vy, mass,
                          arr_target_current[0], arr_target_current[1], arr_target_current[2], arr_target_current[3],
                          ttg, alpha_x,  alpha_y, u )
        
        t_prev = t
        arr_target_prev = arr_target_current
    
    #write to file
    pathout = os.path.join(dir_out, filename_out)
    ephem_v3.write_to_file(pathout, mod_vector_write_frequency=1)


def convert_ephems():

    num_ephems_to_use = 100_000
    ephem_version = 2.0
    cores = 8
    params = {}
    params["num_vec_envs"] = cores
    path_ephems = os.path.join("C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\training_TBR_circular_20251110\\training_TBR_circular\\ephems_extended\\")
    path_output = os.path.join("C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\temp_out2\\")

    set_ephems, filenames = read_ephems_from_dir(path_ephems, num_ephems_to_use, version=ephem_version, 
                                      flag_return_filenames=True, params=params)

    print("Reading ephemerides from: " + path_ephems )
    set_ephems = set_ephems[:num_ephems_to_use]
    num_ephems = len(set_ephems)
    num_states = set_ephems[0].num_vectors * num_ephems
    print("Number of ephemerides: " + str(num_ephems) )
    print("Approx state vectors: " + str(num_states) )

    tasks = [(set_ephems[i], path_output, filenames[i].replace(".txt", "_v3.txt")) for i in range(num_ephems)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        list(tqdm(executor.map(convert_ephem_wrapper, tasks), total=len(tasks), desc="Converting ephemerides"))


if __name__ == "__main__":
    convert_ephems()
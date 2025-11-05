import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plot
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register
from sympy import true

# # Adding python src code directory
# current_dir = os.path.dirname(__file__)
# python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
# sys.path.append(python_src_dir)

from core.ephemeris import Ephemeris
from core.hamiltonian_control import Hamiltonian_Controller_TBT
from utils.log_utils import log
from envs.TwoBody_Orb2Orb_Transfer_Env import TwoBody_Orb2Orb_Transfer_Env

def test_Hamiltonians(flag_report_live=False):

    env = TwoBody_Orb2Orb_Transfer_Env()  # Create directly instead of gym.make()

    test_log = []
    test_log = log(
        "Test Environment Step with Thrust Action", test_log, flag_report_live
    )

    # The prescribed time of flight for the transfer trajectory [s]
    input_TOF = 1.1 * 365.25 * 24 * 60 * 60

    np.set_printoptions(precision=3)  # Limit to 3 decimal places

    # reset the TBT env
    seed = 111
    init_observation, init_info = env.reset(seed=seed)

    # ephemeris
    eph = Ephemeris()

    test_log = log("Test Hamiltonian Controller\n", test_log, flag_report_live)
    test_log = log(
        "Initial observation vector: " + str(init_observation),
        test_log,
        flag_report_live,
    )

    # extract some parameters of interest
    sun_rad = env.unwrapped.planet_radii[0]

    kwargs = {
        "flag_report_live": False,
    }

    # compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(
        env, init_observation, init_info, input_TOF, **kwargs
    )

    # compute solution
    flag_solved, h_sol, eps, sol, log_hsl = H_controller.hamiltonian_solution_finder()

    test_log = log(
        "Hamiltonian solution found: " + str(flag_solved), test_log, flag_report_live
    )

    for item in log_hsl:
        test_log = log(item, test_log, flag_report_live)

    # write output ephemeris
    eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
        H_controller.generate_output_ephemeris(eph)
    )

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_rho)
    # ax.plot(arr_time2, arr_rho2)
    fig.tight_layout()
    ax.set_title("Switching Function")

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_u)
    # ax.plot(arr_time2, arr_u2)
    fig.tight_layout()
    ax.set_title("Spacecraft Thrust Throttle over Time")

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_alpha_x, label="alpha_x")
    ax.plot(arr_time, arr_alpha_y, label="alpha_y")
    ax.set_title("Alpha Vector (Maneuver Direction) over Time")
    fig.tight_layout()
    ax.legend()

    # Ephemeris plotting
    sun_rad = 6.957e8
    sma_Earth = 149598023 * 1000  # m
    sma_Mars = 2.32495e8 * 1000  # m
    eph_out.plot_xy(sun_rad)
    eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
    eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")

    np.set_printoptions(precision=16)
    test_log = log(
        "Solution for initial co-states: " + str(h_sol), test_log, flag_report_live
    )
    test_log = log(
        "Final smoothing parameter used in solution generation: " + str(eps),
        test_log,
        flag_report_live,
    )
    # print(sol)

    eph_out.write_to_file(os.path.join("data", "test_data", "test_hamiltonians", "test_H_ephem.txt"))
    eph1 = Ephemeris()
    eph1.read_from_file(os.path.join("data", "test_data", "test_hamiltonians", "test_H_ephem.txt"))
    test_log = log("Wrote test ephem", test_log, flag_report_live)

    # compare to truth file
    eph2 = Ephemeris()
    eph2.read_from_file(os.path.join("data", "test_data", "test_hamiltonians", "test_H_ephem_truth.txt"))
    test_log = log("Read truth ephem", test_log, flag_report_live)

    # step through and compare each vector
    num_vectors_test = eph1.num_vectors
    num_vectors_truth = eph2.num_vectors
    flag_pass = True
    if num_vectors_test != num_vectors_truth:
        test_log = log("Test FAILED - vec num mismatch", test_log, flag_report_live)
        test_log = log("Vectors in eph_out: " + str(num_vectors_test), test_log, flag_report_live)
        test_log = log("Vectors in eph2: " + str(num_vectors_truth), test_log, flag_report_live)
        flag_pass = False

    for i in range(num_vectors_test - 1):
        vec1 = eph_out.get_vector_at_index(i)
        vec2 = eph2.get_vector_at_index(i)
        # print(f"Comparing vector {i}:")
        # print(f"  Test vec:  {vec1[0]}")
        # print(f"  Truth vec: {vec2[0]}")
        if not np.array_equal(vec1, vec2):
            test_log = log("Test FAILED - vector diff", test_log, flag_report_live)
            flag_pass = False
            break

    # if we made it here, the two ephemerides are the same
    if flag_pass:
        test_log = log("Test passed", test_log, flag_report_live)
    else:
        test_log = log("Test FAILED", test_log, flag_report_live)

    return flag_pass
    
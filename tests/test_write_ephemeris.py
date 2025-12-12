import os

import gymnasium as gym
import matplotlib.pyplot as plot
import numpy as np
from gymnasium import envs
from gymnasium.envs.registration import register

from astro_compass.Constants import Constants
from astro_compass.core.hamiltonian_control import Hamiltonian_Controller_TBT
from astro_compass.Ephemeris import Ephemeris
from astro_compass.utils.path_utils import DATA_ROOT

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


def test_write_ephemeris(env, filename_eph):
    # The prescribed time of flight for the transfer trajectory [s]
    input_TOF = 1.1 * 365.25 * 24 * 60 * 60

    np.set_printoptions(precision=3)  # Limit to 3 decimal places

    # reset the TBT env
    seed = 42
    init_observation, init_info = env.reset(seed=seed)

    # ephemeris
    eph = Ephemeris()

    print(init_observation)

    # compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(
        env, init_observation, init_info, input_TOF
    )

    # reduce smoothing for faster convergence
    H_controller.max_k = 2

    # compute solution
    flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

    # write output ephemeris
    eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
        H_controller.generate_output_ephemeris(eph)
    )

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_rho)
    ax.set_title("Switching Function")

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_u)
    ax.set_title("Spacecraft Thrust Throttle over Time")

    fig, ax = plot.subplots(figsize=(6, 6))
    ax.plot(arr_time, arr_alpha_x, label="alpha_x")
    ax.plot(arr_time, arr_alpha_y, label="alpha_y")
    ax.set_title("Alpha Vector (Maneuver Direction) over Time")
    ax.legend()

    # Ephemeris plotting
    sma_Earth = 149598023 * 1000  # m
    sma_Mars = 2.32495e8 * 1000  # m
    eph_out.plot_xy(Constants.RADIUS_SUN_M)
    eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
    eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")

    np.set_printoptions(precision=16)
    print("Solution for initial co-states: ", h_sol)
    print("Final smoothing parameter used in solution generation: ", eps)

    # write ephemeris file
    eph_out.write_to_file(filename_ephemeris_out, mod_vector_write_frequency=10)
    print("Ephemeris of trajectory written to: ", filename_ephemeris_out)


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")

# Ephemeris filename
dir_ephemeris_out = os.path.join(DATA_ROOT, "training_ephems")
filename_ephemeris_out = dir_ephemeris_out + "test_ephemeris.txt"

test_write_ephemeris(env, filename_ephemeris_out)

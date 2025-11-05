import os
import matplotlib.pyplot as plot
import time
import numpy as np

from constants.constants import Constants
from utils.log_utils import log
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from constants.constants import Constants
from utils.test_utils import compare_trajectories
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from utils.log_utils import write_log_to_file, write_config_file, read_config_file



def datagen_Hamiltonian_TBR_controller(flag_report_live):

    start_time = time.time()

    # define parameters
    params = {
        "mu": Constants.MU_SUN_M,  # sun mu [m^3/s^2]
        "max_T": 1.33/1000,  # max spacecraft thrust [kN]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M)) ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravitational acceleration at Earth surface [m/s^2]
        "env_step_size": 3600 * 24,  # environment step size [s]
        "seed_env_init": 44235,  # random seed for environment
        "num_trajs": 10,  # number of trajectories to simulate
        "max_steps": 1000,  # maximum number of steps per trajectory
        "tof_scale": [1.0, 1.0, 2.0],  # scale factor for time of flight
        "data_path": os.path.join("data", "z_script_output", "training_TBR_circular"),  # path to save data files
        "eps_threshold": 0.0001,
        "flag_plot_traj": True,  # flag to plot trajectories
        "init_costate_guesses": 4,  # number of initial costate guesses to try
        "root_max_iters": 800,  # maximum number of root finding iterations
        "flag_compare_to_truth": True,  # flag to compare output to truth data
        "a_min_env_nd": Constants.SMA_MERCURY,  # min semi-major axis for env [m]
        "a_max_env_nd": Constants.SMA_JUPITER,  # max semi-major axis for env [m]
        "e_min_env": 0.0001,  # min eccentricity for env
        "e_max_env": 0.0001,  # max eccentricity for env
        "w_min_env_deg": 0.0,  # min argument of periapsis for env [rad]
        "w_max_env_deg": 0.0,  # max argument of periapsis for env [rad]
        "flag_report_live": flag_report_live,
    }

    # Write configuration parameters to file
    path_config = os.path.join(params["data_path"], "test_TBR_hamiltonian_config.txt")
    write_config_file(params, path_config)

    test_log = []
    test_log = log(
        "Test Two-Body Rendezvous Hamiltonian Controller", test_log, flag_report_live
    )

    plot.style.use("data/support_files/dark_scientific.mplstyle")

    env = TwoBodyRendezvous_Env(
        mu=params["mu"],  # solar gravitational parameter in m^3/s^2
        max_T=params["max_T"],  # max thrust in N
        ISP=params["ISP"],  # ISP in seconds
        l_star=params["l_star"],  # characteristic length in m
        m_star=params["m_star"],  # characteristic mass in kg
        t_star=params["t_star"],  # characteristic time in s
        g0=params["g0"],  # gravitational acceleration at Earth surface in m/s^2
        step_size=params["env_step_size"],  # environment step size in seconds
        a_min_env_nd=params["a_min_env_nd"], # min semi-major axis for env [AU]
        a_max_env_nd=params["a_max_env_nd"], # max semi-major axis for env [AU]
        e_min_env=params["e_min_env"], # min eccentricity for env
        e_max_env=params["e_max_env"], # max eccentricity for env
        w_min_env_deg=params["w_min_env_deg"], # min argument of periapsis for env [deg]
        w_max_env_deg=params["w_max_env_deg"], # max argument of periapsis for env [deg]
    )

    seed_traj = params["seed_env_init"]
    arr_pass_count = []
    sa_output_ephems = []

    for traj_num in range(params["num_trajs"]):

        #str_gen_time = time.strftime("%Y%m%d_%H%M%S")
        test_log = log(f"\n\nStarting trajectory {traj_num+1} with seed {seed_traj}", test_log, flag_report_live)
        ephem_filename = "test_TBR_ephem_traj_seed_" + str(seed_traj)

        flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(env, seed_traj, params, ephem_filename)

        #check if solved
        if flag_solved == True:
            arr_pass_count.append(1)
            sa_output_ephems.append(os.path.join(params["data_path"], ephem_filename + ".txt"))
            if params["flag_plot_traj"] == True and eph_output is not None:
                eph_output.save_plots(params["data_path"], ephem_filename, params, env)

        else:
            arr_pass_count.append(0)

        seed_traj += 1  # increment trajectory seed

        #close any open plots
        plot.close('all')


    total_pass = sum(arr_pass_count)

    elapsed_time = time.time() - start_time
    test_log = log(f"Total trajectories attempted: {params['num_trajs']}", test_log, flag_report_live)
    test_log = log(f"Total trajectories solved: {total_pass}", test_log, flag_report_live)
    test_log = log(f"Total elapsed time: {elapsed_time/60.0:.2f} minutes", test_log, flag_report_live)
    if total_pass > 0:
        test_log = log("Average time per successful trajectory: " + str(elapsed_time/total_pass) + " seconds", test_log, flag_report_live) 

    #write test log to file
    path_log = os.path.join(params["data_path"], "datagen_Hamiltonian_TBR_controller_log.txt")
    write_log_to_file(path_log, test_log)


datagen_Hamiltonian_TBR_controller(True)
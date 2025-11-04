import matplotlib.pyplot as plot
import time
import numpy as np

from constants.constants import Constants
from utils.log_utils import log
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from constants.constants import Constants
from utils.test_utils import compare_trajectories
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from utils.log_utils import write_log_to_file

def write_config_file(params, path_config):
    """Write configuration parameters to a text file for record-keeping."""
    with open(path_config, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

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
        "seed_env_init": 43000,  # random seed for environment
        "num_trajs": 100,  # number of trajectories to simulate
        "max_steps": 1000,  # maximum number of steps per trajectory
        "tof_scale": [1.25, 1.0, 2.0],  # scale factor for time of flight
        "data_path": "data\\training_ephems\\training_TBR_venus_mars\\",  # path to save data files
        "eps_threshold": 0.0001,
        "flag_plot_traj": True,  # flag to plot trajectories
        "init_costate_guesses": 4,  # number of initial costate guesses to try
        "flag_compare_to_truth": True,  # flag to compare output to truth data
    }

    # Write configuration parameters to file
    path_config = params["data_path"] + "test_TBR_hamiltonian_config.txt"
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
            sa_output_ephems.append(params["data_path"] + ephem_filename + ".txt")
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
    path_log = params["data_path"] + "test_TBR_hamiltonian_log.txt"
    write_log_to_file(path_log, test_log)


datagen_Hamiltonian_TBR_controller(True)
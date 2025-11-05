import os
import matplotlib.pyplot as plot
import time

from constants.constants import Constants
from utils.log_utils import log
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from constants.constants import Constants
from utils.test_utils import compare_trajectories
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory

def write_config_file(params, path_config):
    """Write configuration parameters to a text file for record-keeping."""
    with open(path_config, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def test_Hamiltonian_TBR_Controller(flag_report_live):

    start_time = time.time()

    # define parameters
    params = {
        "mu": Constants.MU_SUN_M,  # sun mu [m^3/s^2]
        "max_T": 1.33/1000,  # max spacecraft thrust [kN]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M)) ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
        "env_step_size": 3600 * 24,  # environment step size [s]
        "seed_env_init": 48,  # random seed for environment
        "num_trajs": 2,  # number of trajectories to simulate
        "max_steps": 1000,  # maximum number of steps per trajectory
        "tof_scale": [1.0, 1.0, 2.0],  # scale factor for time of flight
        "data_path": os.path.join("data", "test_data", "test_TBR_hamiltonian"),  # path to save data files
        "eps_threshold": 0.0001,
        "flag_plot_traj": True,  # flag to plot trajectories
        "init_costate_guesses": 4,  # number of initial costate guesses to try
        "root_max_iters": 400,  # maximum number of root finding iterations
        "flag_compare_to_truth": True,  # flag to compare output to truth data
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
    )

    seed_traj = params["seed_env_init"]
    arr_pass_count = []
    sa_output_ephems = []

    for traj_num in range(params["num_trajs"]):

        #str_gen_time = time.strftime("%Y%m%d_%H%M%S")
        ephem_filename = "test_TBR_ephem_traj_seed_" + str(seed_traj)

        flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(env, seed_traj, params, ephem_filename)

        #check if solved
        if flag_solved == True:
            arr_pass_count.append(1)
            sa_output_ephems.append(os.path.join(params["data_path"], ephem_filename + ".txt"))
            if params["flag_plot_traj"] == True:
                eph_output.save_plots(params["data_path"], ephem_filename, params, env)

        else:
            arr_pass_count.append(0)

        seed_traj += 1  # increment trajectory seed

        #close any open plots
        plot.close('all')


    total_pass = sum(arr_pass_count)
    test_log = log(
        f"Hamiltonian Controller Test Complete: {total_pass} out of {params['num_trajs']} trajectories solved.",
        test_log,
        flag_report_live,
    )

    count = 1
    for item in arr_pass_count:
        test_log = log("Traj " + str(count) + " solution found: " + str(item), test_log, flag_report_live)
        count += 1

    elapsed_time = time.time() - start_time
    test_log = log(f"Total elapsed time: {elapsed_time/60.0:.2f} minutes", test_log, flag_report_live)
    if total_pass > 0:
        test_log = log("Average time per successful trajectory: " + str(elapsed_time/total_pass) + " seconds", test_log, flag_report_live)

    test_log = log("\n\n\nComparing trajectories to truth data...\n", test_log, flag_report_live)
    
    # Build truth file list to match the output ephemeris files that were actually generated
    # Extract seed numbers from output filenames
    sa_truth_ephems = []
    for output_file in sa_output_ephems:
        # Extract seed number from filename like "test_TBR_ephem_traj_seed_47.txt"
        seed_num = output_file.split("_seed_")[1].split(".txt")[0]
        truth_file = os.path.join(params["data_path"], f"test_TBR_ephem_traj_seed_{seed_num}_truth.txt")
        sa_truth_ephems.append(truth_file)
    
    flag_all_match = False
    if (params["flag_compare_to_truth"] == True):
        test_log, flag_all_match = compare_trajectories(sa_output_ephems, 
                                                        sa_truth_ephems, test_log, 2, flag_report_live)
        if flag_all_match:
            test_log = log("All trajectories match truth data!", test_log, flag_report_live)
        else:
            test_log = log("Some trajectories do not match truth data.", test_log, flag_report_live)

        return flag_all_match

test_Hamiltonian_TBR_Controller(True)

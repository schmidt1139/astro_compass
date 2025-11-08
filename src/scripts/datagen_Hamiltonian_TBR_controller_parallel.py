import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plot
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

from constants.constants import Constants
from utils.log_utils import log
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from constants.constants import Constants
from utils.test_utils import compare_trajectories
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from utils.log_utils import write_log_to_file, write_config_file, read_config_file
from core.process_single_trajectory import process_single_trajectory

def prepare_trajectory_tasks(params):
    """
    Prepare the list of trajectory tasks to be processed in parallel.
    
    Args:
        params: Dictionary of parameters

    Returns:
        List of trajectory tasks as tuples (traj_num, seed_traj, tof_scale)
    """
    trajectory_tasks = []
    for traj_num in range(params["num_trajs"]):
        if params["randomize_seeds"]:
            seed_traj = np.random.randint(0, 2**31 - 1)
        else:
            seed_traj = params["seed_env_init"] + traj_num

        if params["randomize_tofs"]:
            tof_scale = np.random.choice(params["tof_scales"])
        else:
            tof_scale = params["tof_scales"][0]

        trajectory_tasks.append((traj_num, seed_traj, tof_scale))
    
    return trajectory_tasks

def datagen_Hamiltonian_TBR_controller_parallel(flag_report_live):

    start_time = time.time()

    path_config = os.path.join("data", "config", "datagen_Hamiltonian_TBR_controller_parallel_config.txt")
    params = read_config_file(path_config)

    # Write configuration parameters to file
    time_str = time.strftime("%Y%m%d_%H%M%S")
    path_config = os.path.join(params["data_path"], "datagen_Hamiltonian_TBR_controller_parallel_config_" + time_str + ".txt")
    write_config_file(params, path_config)

    test_log = []
    test_log = log(
        "Test Two-Body Rendezvous Hamiltonian Controller", test_log, flag_report_live
    )

    plot.style.use(os.path.join("data", "support_files", "dark_scientific.mplstyle"))

    # Ensure boolean parameters are actually booleans (config might read as strings)
    if isinstance(params.get("randomize_seeds"), str):
        params["randomize_seeds"] = params["randomize_seeds"].lower() in ['true', '1', 'yes']
    if isinstance(params.get("randomize_tofs"), str):
        params["randomize_tofs"] = params["randomize_tofs"].lower() in ['true', '1', 'yes']

    # Determine number of processes to use
    num_processes = min(cpu_count(), params.get("num_cores", cpu_count()))
    print("CPU count:", cpu_count())
    print(f"Using {num_processes} parallel processes to generate {params['num_trajs']} trajectories")

    # Prepare list of trajectories to process
    trajectory_tasks = prepare_trajectory_tasks(params)

    # Process trajectories in parallel
    arr_pass_count = []
    sa_output_ephems = []
    completed = 0
    
    with Pool(processes=num_processes) as pool:
        # Use partial to pass params to each worker
        process_func = partial(process_single_trajectory, params=params)
        
        # Process all trajectories in parallel with live updates
        for result in pool.imap_unordered(process_func, trajectory_tasks):
            flag_solved, ephem_path, seed_traj, str_gen_time = result
            completed += 1
            
            if flag_solved:
                print(f"[{str_gen_time}] [{completed}/{params['num_trajs']}] Trajectory seed {seed_traj} solved successfully.")
                arr_pass_count.append(1)
                sa_output_ephems.append(ephem_path)
            else:
                print(f"[{str_gen_time}] [{completed}/{params['num_trajs']}] Trajectory seed {seed_traj} failed to solve.")
                arr_pass_count.append(0)


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


if __name__ == "__main__":
    datagen_Hamiltonian_TBR_controller_parallel(True)  # Set to True for verbose output
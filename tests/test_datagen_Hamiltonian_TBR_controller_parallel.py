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



def test_datagen_Hamiltonian_TBR_parallel(flag_report_live):

    start_time = time.time()

    # Get the workspace root (parent of tests directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    path_config = os.path.join(workspace_root, "data", "test_data", "test_datagen_Hamiltonian_TBR_parallel", "test_datagen_Hamiltonian_TBR_controller_parallel_config.txt")
    params = read_config_file(path_config)

    # Convert data_path to absolute path if it's relative (relative to config file location)
    if not os.path.isabs(params["data_path"]):
        config_dir = os.path.dirname(path_config)
        params["data_path"] = os.path.abspath(os.path.join(config_dir, params["data_path"]))

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

        test_log = log(f"Queuing trajectory {traj_num+1} with seed {seed_traj} and tof scale {tof_scale}", test_log, flag_report_live)
        trajectory_tasks.append((traj_num, seed_traj, tof_scale))

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
                # Don't append to sa_output_ephems when trajectory fails


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

    # Build truth file list based on what was actually generated successfully
    sa_truth_ephems = []
    for output_path in sa_output_ephems:
        # Extract filename and replace 'test_' with 'truth_'
        filename = os.path.basename(output_path)
        truth_filename = filename.replace('test_', 'truth_')
        truth_path = os.path.join(params["data_path"], truth_filename)
        sa_truth_ephems.append(truth_path)

    #sort output files to match truth files
    sa_output_ephems.sort()
    sa_truth_ephems.sort()

    flag_all_match = False

    test_log, flag_all_match = compare_trajectories(sa_output_ephems, 
                                                    sa_truth_ephems, test_log, 2, flag_report_live)
    if flag_all_match:
        test_log = log("All trajectories match truth data!", test_log, flag_report_live)
    else:
        test_log = log("Some trajectories do not match truth data.", test_log, flag_report_live)

    return flag_all_match


if __name__ == "__main__":
    test_datagen_Hamiltonian_TBR_parallel(True)  # Set to True for verbose output
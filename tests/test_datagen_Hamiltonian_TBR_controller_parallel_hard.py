import os
from core.datagen import run_parallel_trajectory_generation
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



def test_datagen_Hamiltonian_TBR_parallel_hard(flag_report_live):

    start_time = time.time()

    # Get the workspace root (parent of tests directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    path_config = os.path.join(workspace_root, "data", "test_data", "test_datagen_Hamiltonian_TBR_parallel_hard", "test_datagen_Hamiltonian_TBR_controller_parallel_config_hard.txt")
    params = read_config_file(path_config)

    # Run parallel trajectory generation
    test_log, arr_pass_count, sa_output_ephems = run_parallel_trajectory_generation(params)

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
    test_datagen_Hamiltonian_TBR_parallel_hard(True)  # Set to True for verbose output
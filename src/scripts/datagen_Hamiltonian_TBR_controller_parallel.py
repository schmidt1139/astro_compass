import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plot
import time

from multiprocessing import Pool, cpu_count

from utils.log_utils import log
from utils.log_utils import write_log_to_file, write_config_file, read_config_file
from core.datagen import run_parallel_trajectory_generation


def datagen_Hamiltonian_TBR_controller_parallel():

    start_time = time.time()

    path_config = os.path.join("data", "config", "datagen_Hamiltonian_TBR_controller_parallel_config.txt")
    params = read_config_file(path_config)

    # Run parallel trajectory generation
    test_log, arr_pass_count, sa_output_ephems = run_parallel_trajectory_generation(params)
    
    flag_report_live = params.get("flag_report_live", False)

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
    datagen_Hamiltonian_TBR_controller_parallel()
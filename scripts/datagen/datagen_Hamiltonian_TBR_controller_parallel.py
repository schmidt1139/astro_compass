import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import cProfile
import pstats
import time

from astro_compass.core.datagen import run_parallel_trajectory_generation
from astro_compass.utils.log_utils import log, read_config_file, write_log_to_file


def datagen_Hamiltonian_TBR_controller_parallel():
    start_time = time.time()
    time_str = time.strftime("%Y%m%d_%H%M%S")

    path_config = os.path.join(
        "data", "config", "datagen_Hamiltonian_TBR_controller_parallel_config.txt"
    )
    params = read_config_file(path_config)

    # Validate and normalize data_path
    if "data_path" not in params:
        raise ValueError("data_path must be specified in configuration file")

    # Remove leading/trailing whitespace
    data_path = params["data_path"].strip()

    # If it's not already an absolute path, make it relative to the current working directory
    if not os.path.isabs(data_path):
        # Convert to absolute path relative to current directory
        data_path = os.path.abspath(data_path)

    # Normalize the path (handles trailing slashes, converts to OS-specific separators)
    params["data_path"] = os.path.normpath(data_path)

    # Verify the base directory exists
    if not os.path.exists(params["data_path"]):
        raise FileNotFoundError(
            f"data_path directory does not exist: '{params['data_path']}'. Please create it before running."
        )

    if not os.path.isdir(params["data_path"]):
        raise NotADirectoryError(
            f"data_path is not a directory: '{params['data_path']}'"
        )

    print(f"Using data_path: {params['data_path']}")

    # generate subdirectories if they do not exist
    subdirs = ["ephems", "plots", "logs", "summaries", "plots"]
    for subdir in subdirs:
        dir_path = os.path.join(params["data_path"], subdir)
        os.makedirs(dir_path, exist_ok=True)

    # Run parallel trajectory generation
    test_log, arr_pass_count, sa_output_ephems, sa_summary = (
        run_parallel_trajectory_generation(params)
    )

    flag_report_live = params.get("flag_report_live", False)

    total_pass = sum(arr_pass_count)

    trajs_per_core = (
        params["num_trajs"] / params["num_cores"] if params["num_cores"] > 0 else 0
    )
    sa_summary.append(f"Trajectories per core: {trajs_per_core:.2f}")
    elapsed_time = time.time() - start_time
    test_log = log(
        f"Total trajectories attempted: {params['num_trajs']}",
        test_log,
        flag_report_live,
    )
    test_log = log(
        f"Total trajectories solved: {total_pass}", test_log, flag_report_live
    )
    test_log = log(
        f"Total elapsed wall time: {elapsed_time / 60.0:.2f} minutes",
        test_log,
        flag_report_live,
    )
    test_log = log(
        f"Total time x cores: {elapsed_time * params['num_cores'] / 60.0:.2f} minutes",
        test_log,
        flag_report_live,
    )
    test_log = log(
        f"Trajectories per core: {trajs_per_core:.2f}", test_log, flag_report_live
    )
    sa_summary.append(f"Total trajectories attempted: {params['num_trajs']}")
    sa_summary.append(f"Total trajectories solved: {total_pass}")
    sa_summary.append(f"Total elapsed wall time: {elapsed_time / 60.0:.2f} minutes")
    sa_summary.append(
        f"Total time x cores: {elapsed_time * params['num_cores'] / 60.0:.2f} minutes"
    )
    sa_summary.append(f"Trajectories per core: {trajs_per_core:.2f}")

    time_per_requested = (
        elapsed_time / params["num_trajs"] if params["num_trajs"] > 0 else 0
    )
    time_per_requested_per_core = time_per_requested / params["num_cores"]
    test_log = log(
        f"Average time per requested trajectory: {time_per_requested:.2f} seconds",
        test_log,
        flag_report_live,
    )
    test_log = log(
        f"Average time per requested trajectory per core: {elapsed_time / trajs_per_core:.2f} seconds",
        test_log,
        flag_report_live,
    )
    sa_summary.append(
        f"Average time per requested trajectory: {time_per_requested:.2f} seconds"
    )
    sa_summary.append(
        f"Average time per requested trajectory per core: {elapsed_time / trajs_per_core:.2f} seconds"
    )
    if total_pass > 0:
        test_log = log(
            f"Average time per successful trajectory: {elapsed_time / total_pass:.2f} seconds",
            test_log,
            flag_report_live,
        )
        sa_summary.append(
            f"Average time per successful trajectory: {elapsed_time / total_pass:.2f} seconds"
        )

    # write test log to file
    path_log = os.path.join(
        params["data_path"],
        "logs",
        "datagen_Hamiltonian_TBR_controller_log_" + time_str + ".txt",
    )
    path_summary = os.path.join(
        params["data_path"],
        "summaries",
        "datagen_Hamiltonian_TBR_controller_parallel_summary_" + time_str + ".txt",
    )
    write_log_to_file(path_log, test_log)
    write_log_to_file(path_summary, sa_summary)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    datagen_Hamiltonian_TBR_controller_parallel()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")

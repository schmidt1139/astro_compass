import os

import matplotlib

from astro_compass.core.datagen import run_parallel_trajectory_generation

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import time

from astro_compass.utils.log_utils import log, read_config_file
from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.utils.test_utils import compare_trajectories


def test_datagen_Hamiltonian_TBR_parallel(flag_report_live):
    start_time = time.time()

    path_config = os.path.join(
        DATA_ROOT,
        "test_data",
        "test_datagen_Hamiltonian_TBR_parallel",
        "test_datagen_Hamiltonian_TBR_controller_parallel_config3.txt",
    )
    params = read_config_file(path_config)

    # Validate and normalize data_path
    if "data_path" not in params:
        raise ValueError("data_path must be specified in configuration file")

    # Strip whitespace and normalize path
    data_path = params["data_path"].strip()

    # Convert to absolute path if relative
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)

    params["data_path"] = os.path.normpath(data_path)

    # Run parallel trajectory generation
    test_log, arr_pass_count, sa_output_ephems, sa_summary = (
        run_parallel_trajectory_generation(params)
    )

    # Build truth file list based on what was actually generated successfully
    sa_truth_ephems = []
    for output_path in sa_output_ephems:
        # Extract filename and replace 'test_' with 'truth_'
        filename = os.path.basename(output_path)
        truth_filename = filename.replace("test_", "truth_")
        truth_path = os.path.join(params["data_path"], "ephems", truth_filename)
        sa_truth_ephems.append(truth_path)

    # sort output files to match truth files
    sa_output_ephems.sort()
    sa_truth_ephems.sort()

    flag_all_match = False

    for item in sa_summary:
        test_log = log(item, test_log, flag_report_live)

    test_log, flag_all_match = compare_trajectories(
        sa_output_ephems, sa_truth_ephems, test_log, 2, flag_report_live
    )
    if flag_all_match:
        test_log = log("All trajectories match truth data!", test_log, flag_report_live)
    else:
        test_log = log(
            "Some trajectories do not match truth data.", test_log, flag_report_live
        )

    return flag_all_match


if __name__ == "__main__":
    test_datagen_Hamiltonian_TBR_parallel(True)  # Set to True for verbose output

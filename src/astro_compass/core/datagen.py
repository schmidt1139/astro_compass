import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plot
import numpy as np
from core.process_single_trajectory import process_single_trajectory

from astro_compass.utils.log_utils import log, write_config_file


def record_pass_stats(arr_pass_count, tof_index, scenario_index):
    """
    Record pass statistics for TOF scales and scenarios.

    Args:
        arr_pass_count: List of pass counts
        tof_index: Index of the TOF scale
        scenario_index: Index of the scenario
    """
    arr_pass_count[tof_index, scenario_index] += 1


def prepare_trajectory_tasks(params):
    """
    Prepare the list of trajectory tasks to be processed in parallel.

    Args:
        params: Dictionary of parameters

    Returns:
        List of trajectory tasks as tuples (traj_num, seed_traj, tof_scale)
    """
    # Seed the random number generator
    # If seed_env_init is None or "None", use time-based seed for different trajectories each run
    seed = params.get("seed_env_init", None)
    if seed is None or (isinstance(seed, str) and seed.lower() == "none"):
        seed = int(time.time() * 1000) % (2**31)  # Use current time as seed
    np.random.seed(int(seed))

    trajectory_tasks = []
    list_params = []
    counts_tof_scale = np.zeros(len(params["tof_scales"]), dtype=int)
    counts_scenario = np.zeros(len(params["kep_scenario_weights"]), dtype=int)

    for traj_num in range(params["num_trajs"]):
        # Create a copy of params for this trajectory to avoid modifying the original
        params_i = params.copy()
        params_i["traj_num"] = traj_num

        if params["randomize_seeds"]:
            seed_traj = np.random.randint(0, 2**31 - 1)
        else:
            seed_traj = params["seed_env_init"] + traj_num
        params_i["seed_traj"] = seed_traj

        if params["randomize_tofs"]:
            tof_weights = params.get("tof_weights", None)
            if tof_weights is not None:
                tof_index = np.random.choice(len(params["tof_scales"]), p=tof_weights)
            else:
                tof_index = np.random.choice(len(params["tof_scales"]))
            tof_scale = params["tof_scales"][tof_index]
        else:
            tof_index = 0
            tof_scale = params["tof_scales"][0]

        params_i["tof_scale"] = tof_scale
        params_i["tof_index"] = tof_index

        # Set randomized orbital element limits if specified
        if params["randomize_limits"]:
            # select limits based on one random scenario
            scenario_index = np.random.choice(
                len(params["kep_scenario_weights"]), p=params["kep_scenario_weights"]
            )
            params_i["scenario_index"] = scenario_index
            a_min_init_env_nd = params["a_min_init_env_nd"][scenario_index]
            a_max_init_env_nd = params["a_max_init_env_nd"][scenario_index]
            e_min_init_env = params["e_min_init_env"][scenario_index]
            e_max_init_env = params["e_max_init_env"][scenario_index]
            w_min_init_env_deg = params["w_min_init_env_deg"][scenario_index]
            w_max_init_env_deg = params["w_max_init_env_deg"][scenario_index]

            a_min_final_env_nd = params["a_min_final_env_nd"][scenario_index]
            a_max_final_env_nd = params["a_max_final_env_nd"][scenario_index]
            e_min_final_env = params["e_min_final_env"][scenario_index]
            e_max_final_env = params["e_max_final_env"][scenario_index]
            w_min_final_env_deg = params["w_min_final_env_deg"][scenario_index]
            w_max_final_env_deg = params["w_max_final_env_deg"][scenario_index]

            params_i["a_min_init_env_nd"] = a_min_init_env_nd
            params_i["a_max_init_env_nd"] = a_max_init_env_nd
            params_i["e_min_init_env"] = e_min_init_env
            params_i["e_max_init_env"] = e_max_init_env
            params_i["w_min_init_env_deg"] = w_min_init_env_deg
            params_i["w_max_init_env_deg"] = w_max_init_env_deg

            params_i["a_min_final_env_nd"] = a_min_final_env_nd
            params_i["a_max_final_env_nd"] = a_max_final_env_nd
            params_i["e_min_final_env"] = e_min_final_env
            params_i["e_max_final_env"] = e_max_final_env
            params_i["w_min_final_env_deg"] = w_min_final_env_deg
            params_i["w_max_final_env_deg"] = w_max_final_env_deg

        else:
            scenario_index = 0
            params_i["scenario_index"] = 0
            params_i["a_min_init_env_nd"] = params["a_min_init_env_nd"][0]
            params_i["a_max_init_env_nd"] = params["a_max_init_env_nd"][0]
            params_i["e_min_init_env"] = params["e_min_init_env"][0]
            params_i["e_max_init_env"] = params["e_max_init_env"][0]
            params_i["w_min_init_env_deg"] = params["w_min_init_env_deg"][0]
            params_i["w_max_init_env_deg"] = params["w_max_init_env_deg"][0]

            params_i["a_min_final_env_nd"] = params["a_min_final_env_nd"][0]
            params_i["a_max_final_env_nd"] = params["a_max_final_env_nd"][0]
            params_i["e_min_final_env"] = params["e_min_final_env"][0]
            params_i["e_max_final_env"] = params["e_max_final_env"][0]
            params_i["w_min_final_env_deg"] = params["w_min_final_env_deg"][0]
            params_i["w_max_final_env_deg"] = params["w_max_final_env_deg"][0]

        # update counters
        counts_tof_scale[tof_index] += 1
        counts_scenario[scenario_index] += 1
        list_params.append(params_i)

    arr_pass_count_stats = np.zeros(
        (len(params["tof_scales"]), len(params["kep_scenario_weights"])), dtype=int
    )

    return list_params, counts_tof_scale, counts_scenario, arr_pass_count_stats


def run_parallel_trajectory_generation(params):
    """
    Run parallel trajectory generation with live updates and timeout support.

    Args:
        params: Dictionary of parameters containing all configuration settings

    Returns:
        tuple: (test_log, arr_pass_count, sa_output_ephems)
    """
    flag_report_live = params.get("flag_report_live", False)

    # Write configuration parameters to file
    start_time = time.time()  # Store start time for elapsed time calculations
    time_str = time.strftime("%Y%m%d_%H%M%S")
    config_dir = os.path.join(params["data_path"], "configs")
    path_config = os.path.join(
        config_dir,
        "datagen_Hamiltonian_TBR_controller_parallel_config_" + time_str + ".txt",
    )
    # create configs directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    write_config_file(params, path_config)

    test_log = []
    test_log = log(
        "Test Two-Body Rendezvous Hamiltonian Controller", test_log, flag_report_live
    )

    test_log = log(f"Starting at {time_str}", test_log, True)

    plot.style.use(os.path.join("data", "support_files", "dark_scientific.mplstyle"))

    # Determine number of processes to use
    num_processes = min(cpu_count(), params.get("num_cores", cpu_count()))
    print("CPU count:", cpu_count())
    print(
        f"Using {num_processes} parallel processes to generate {params['num_trajs']} trajectories"
    )

    # Get timeout per trajectory (in seconds)
    timeout_per_trajectory = params.get(
        "timeout_per_trajectory", 300
    )  # Default 5 minutes
    print(f"Timeout per trajectory: {timeout_per_trajectory} seconds")

    # Prepare list of trajectories to process
    trajectory_tasks, counts_tof_scale, counts_scenario, arr_pass_count_stats = (
        prepare_trajectory_tasks(params)
    )
    for param in trajectory_tasks:
        test_log = log(
            f"Queuing trajectory {param['traj_num'] + 1} with seed {param['seed_traj']} and tof scale {param['tof_scale']} and scenario {param['scenario_index']}",
            test_log,
            flag_report_live,
        )

    # Process trajectories in parallel
    arr_pass_count = []
    sa_output_ephems = []
    completed = 0

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better handling - tasks only start when a worker is available
        async_results = pool.imap_unordered(
            process_single_trajectory, trajectory_tasks, chunksize=1
        )

        # Process results as they complete
        for result in async_results:
            (
                flag_solved,
                ephem_path,
                result_seed,
                str_gen_time,
                timed_out,
                processing_time,
                params_result,
            ) = result
            completed += 1

            elapsed_seconds = time.time() - start_time

            if timed_out:
                print(
                    f"[{str_gen_time}  {elapsed_seconds:.1f}s] [{completed}/{params['num_trajs']}] Trajectory seed {result_seed} TIMED OUT after {processing_time:.1f}s (limit: {timeout_per_trajectory}s)."
                )
                arr_pass_count.append(0)
            elif flag_solved:
                print(
                    f"[{str_gen_time}  {elapsed_seconds:.1f}s] [{completed}/{params['num_trajs']}] Trajectory seed {result_seed} solved successfully."
                )
                arr_pass_count.append(1)
                tof_index = params_result["tof_index"]
                scenario_index = params_result["scenario_index"]
                arr_pass_count_stats[tof_index, scenario_index] += 1
                sa_output_ephems.append(ephem_path)
            else:
                print(
                    f"[{str_gen_time}  {elapsed_seconds:.1f}s] [{completed}/{params['num_trajs']}] Trajectory seed {result_seed} failed to solve."
                )
                arr_pass_count.append(0)

    # summarize counts
    sa_summary = []
    sa_summary.append("\nTrajectory Generation Summary:\n")
    sa_summary.append("------------------------------\n")
    print("Trajectory generation summary:")
    for i, count in enumerate(counts_tof_scale):
        print(f"  TOF scale {params['tof_scales'][i]}: {count} trajectories")
        sa_summary.append(
            f"  TOF scale {params['tof_scales'][i]}: {count} trajectories\n"
        )
    for i, count in enumerate(counts_scenario):
        print(f"  Scenario {i}: {count} trajectories")
        sa_summary.append(f"  Scenario {i}: {count} trajectories\n")
    sa_summary.append("------------------------------\n")
    # summarize success rates
    print("Trajectory generation success rates:")
    sa_summary.append("\nTrajectory generation success rates:\n")
    for i, count in enumerate(counts_tof_scale):
        print(
            f"  TOF scale {params['tof_scales'][i]}: {arr_pass_count_stats[i, :].sum()} solved out of {count} trajectories"
        )
        sa_summary.append(
            f"  TOF scale {params['tof_scales'][i]}: {arr_pass_count_stats[i, :].sum()} solved out of {count} trajectories\n"
        )
    for i, count in enumerate(counts_scenario):
        print(
            f"  Scenario {i}: {arr_pass_count_stats[:, i].sum()} solved out of {count} trajectories"
        )
        sa_summary.append(
            f"  Scenario {i}: {arr_pass_count_stats[:, i].sum()} solved out of {count} trajectories\n"
        )

    # determine success rate for TOF scales
    total_solved = sum(arr_pass_count)

    return test_log, arr_pass_count, sa_output_ephems, sa_summary

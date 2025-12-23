import os
import time
import signal
import numpy as np
from core.gen_Hamiltonian_transfer import gen_Hamiltonian_transfer
import matplotlib.pyplot as plot
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from utils.log_utils import write_log_to_file, log, log_parameters
from core.exceptions import TimeoutException, timeout_handler
from utils.env_utils import gen_rl_environment
from utils.plotting_utils import plot_overlay_ballistic_orbit, plot_rendezvous_traj

def process_single_trajectory(params):
    """
    Process a single trajectory. This function will be called in parallel.

    Args:
        params: Dictionary of parameters containing all trajectory-specific settings
                including 'traj_num', 'seed_traj', and 'tof_scale'

    Returns:
        Tuple of (success, ephem_path, seed_traj, str_gen_time)
    """

    seed_traj = params["seed_traj"]
    tof_scale = params["tof_scale"]
    flag_report_live = params.get("flag_report_live", False)
    scenario_index = params["scenario_index"]
    flag_report_logs = params.get("report_logs", False)
    flag_debug_h_targeter = params.get("flag_debug_h_targeter", False)
    flag_report_live = params.get("flag_report_live", False)
    flag_print_targeter_output = False

    if flag_report_live and flag_debug_h_targeter:
        flag_print_targeter_output = True

    # Create environment for this process
    env = gen_rl_environment(params)

    # Generate filename
    tof_scale_str = str(tof_scale).replace(".", "p")
    ephem_dir = os.path.join(params["data_path"], "ephems")
    plot_dir = os.path.join(params["data_path"], "plots")
    ephem_filename = f"test_TBR_ephem_traj_seed_{seed_traj}_tof_{tof_scale_str}_scenario_{scenario_index}"

    # Record when this worker actually starts processing
    worker_start_time = time.time()

    # Get timeout setting
    timeout_per_trajectory = params.get("timeout_per_trajectory", 300)

    # Generate trajectory with timeout enforcement
    test_log = []
    timed_out = False
    flag_solved = False
    eph_output = None

    # Set up timeout signal (only works on Unix-like systems)
    try:
        # Set the timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_per_trajectory)

        try:
            flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(
                env,
                seed_traj,
                tof_scale,
                params,
                ephem_filename,
                test_log,
                flag_report_live=flag_print_targeter_output,
            )
        except TimeoutException:
            timed_out = True
            flag_solved = False
        finally:
            # Cancel the alarm
            signal.alarm(0)
    except (AttributeError, ValueError):
        # signal.SIGALRM not available on Windows, fall back to no timeout enforcement
        try:
            flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(
                env,
                seed_traj,
                tof_scale,
                params,
                ephem_filename,
                test_log,
                flag_report_live=flag_print_targeter_output,
            )
        except TimeoutException:
            timed_out = True
            flag_solved = False

    str_gen_time = time.strftime("%b %d %Y %H:%M:%S")

    # Calculate actual processing time
    processing_time = time.time() - worker_start_time

    # Check if we exceeded timeout (fallback for systems without signal support)
    if not timed_out and processing_time > timeout_per_trajectory:
        timed_out = True

    # Save plots if solved
    if flag_solved and params["flag_plot_traj"] and eph_output is not None:
        eph_output.save_plots(plot_dir, ephem_filename, params, env)

    # write ephemeris if solved
    ephem_path = (
        os.path.join(ephem_dir, ephem_filename + ".txt") if flag_solved else None
    )
    if flag_solved and eph_output is not None:
        eph_output.write_to_file(os.path.join(ephem_dir, ephem_filename + ".txt"))

    # Add parameters to the log after trajectory generation
    test_log = log_parameters(params, test_log, False)

    # record log to file if requested
    if flag_report_logs == True:
        # flag solved string
        flag_solved_str = "Solved" if flag_solved else "Failed"

        # make log directory if it doesn't exist
        # Convert to absolute path to ensure it works in multiprocessing workers
        data_path_abs = os.path.abspath(params["data_path"])
        log_dir = os.path.join(data_path_abs, "logs")

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(
                log_dir, f"process_single_trajectory_{seed_traj}_{flag_solved_str}.log"
            )
            write_log_to_file(log_file_path, test_log)
            # print(f"[DEBUG] Log written to: {log_file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write log: {type(e).__name__}: {e}")
            print(f"[ERROR] Attempted path: {log_dir}")
            print(f"[ERROR] data_path from params: {params['data_path']}")
            print(f"[ERROR] data_path_abs: {data_path_abs}")

    # Close any plots to free memory
    plot.close("all")

    return (
        flag_solved,
        ephem_path,
        seed_traj,
        str_gen_time,
        timed_out,
        processing_time,
        params,
    )




def process_single_transfer(params):
    """
    Process a single trajectory. This function will be called in parallel.

    Args:
        params: Dictionary of parameters containing all trajectory-specific settings
                including 'traj_num', 'seed_traj', and 'tof_scale'

    Returns:
        Tuple of (success, ephem_path, seed_traj, str_gen_time)
    """

    seed_traj = params["seed_traj"]
    tof_scale = params["tof_scale"]
    flag_report_live = params.get("flag_report_live", False)
    scenario_index = params["scenario_index"]
    flag_report_logs = params.get("report_logs", False)
    flag_debug_h_targeter = params.get("flag_debug_h_targeter", False)
    flag_report_live = params.get("flag_report_live", False)
    flag_print_targeter_output = False

    if flag_report_live and flag_debug_h_targeter:
        flag_print_targeter_output = True

    # Create environment for this process
    env = gen_rl_environment(params)

    # Generate filename
    tof_scale_str = str(tof_scale).replace(".", "p")
    ephem_dir = os.path.join(params["data_path"], "ephems")
    plot_dir = os.path.join(params["data_path"], "plots")
    ephem_filename = f"test_TBR_ephem_traj_seed_{seed_traj}_tof_{tof_scale_str}_scenario_{scenario_index}"

    # Record when this worker actually starts processing
    worker_start_time = time.time()

    # Get timeout setting
    timeout_per_trajectory = params.get("timeout_per_trajectory", 300)

    # Generate trajectory with timeout enforcement
    test_log = []
    timed_out = False
    flag_solved = False
    eph_output = None

    # Set up timeout signal (only works on Unix-like systems)

    try:
        flag_solved, test_log, eph_output = gen_Hamiltonian_transfer(
            env,
            seed_traj,
            tof_scale,
            params,
            ephem_filename,
            test_log,
            flag_report_live=flag_print_targeter_output,
        )
    except TimeoutException:
        timed_out = True
        flag_solved = False
    except (AttributeError, ValueError):
        flag_solved = False


    str_gen_time = time.strftime("%b %d %Y %H:%M:%S")

    # Calculate actual processing time
    processing_time = time.time() - worker_start_time

    # Check if we exceeded timeout (fallback for systems without signal support)
    if not timed_out and processing_time > timeout_per_trajectory:
        timed_out = True

    # Save plots if solved
    if flag_solved and params["flag_plot_traj"] and eph_output is not None:

        eph_output.save_plots(plot_dir, ephem_filename, params, env)

        state = eph_output.get_vector_at_index(0)
        x_init = state[1]
        y_init = state[2]
        vx_init = state[3]
        vy_init = state[4]
        x_target_init = state[6]
        y_target_init = state[7]
        vx_target_init = state[8]
        vy_target_init = state[9]

        env.reset()
        fig_orb = eph_output.plot_xy(color_in="#7e03a8")
        fig_orb = plot_overlay_ballistic_orbit(
            x_init,
            y_init,
            vx_init,
            vy_init,
            env,
            fig_orb,
            params,
            eph_output,
            label_in="Initial Orbit",
            color_in="#0d0887",
        )
        fig_orb = plot_overlay_ballistic_orbit(
            x_target_init,
            y_target_init,
            vx_target_init,
            vy_target_init,
            env,
            fig_orb,
            params,
            eph_output,
            label_in="Target Orbit",
            color_in="#cc4778",
            )
        fig_orb = eph_output.adjust_plot_limits()

        fig_orb.savefig(os.path.join(plot_dir, ephem_filename + "_trajectory.png"), dpi=300)

    # write ephemeris if solved
    ephem_path = (
        os.path.join(ephem_dir, ephem_filename + ".txt") if flag_solved else None
    )
    if flag_solved and eph_output is not None:
        eph_output.write_to_file(os.path.join(ephem_dir, ephem_filename + ".txt"), mod_vector_write_frequency=10)

    # Add parameters to the log after trajectory generation
    test_log = log_parameters(params, test_log, False)

    # record log to file if requested
    if flag_report_logs == True:
        # flag solved string
        flag_solved_str = "Solved" if flag_solved else "Failed"

        # make log directory if it doesn't exist
        # Convert to absolute path to ensure it works in multiprocessing workers
        data_path_abs = os.path.abspath(params["data_path"])
        log_dir = os.path.join(data_path_abs, "logs")

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(
                log_dir, f"process_single_trajectory_{seed_traj}_{flag_solved_str}.log"
            )
            write_log_to_file(log_file_path, test_log)
            # print(f"[DEBUG] Log written to: {log_file_path}")

        except Exception as e:
            print(f"[ERROR] Failed to write log: {type(e).__name__}: {e}")
            print(f"[ERROR] Attempted path: {log_dir}")
            print(f"[ERROR] data_path from params: {params['data_path']}")
            print(f"[ERROR] data_path_abs: {data_path_abs}")

    # Close any plots to free memory
    plot.close("all")

    return (
        flag_solved,
        ephem_path,
        seed_traj,
        str_gen_time,
        timed_out,
        processing_time,
        params,
    )

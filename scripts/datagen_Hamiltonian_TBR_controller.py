import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import time

import matplotlib.pyplot as plot
import numpy as np
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env

from astro_compass.core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from astro_compass.utils.log_utils import (
    log,
    read_config_file,
    write_config_file,
    write_log_to_file,
)


def datagen_Hamiltonian_TBR_controller():
    start_time = time.time()

    path_config = os.path.join(
        "data", "config", "datagen_Hamiltonian_TBR_controller_config.txt"
    )
    params = read_config_file(path_config)

    flag_report_live = params.get("flag_report_live", False)

    # Write configuration parameters to file
    time_str = time.strftime("%Y%m%d_%H%M%S")
    path_config = os.path.join(
        params["data_path"],
        "datagen_Hamiltonian_TBR_controller_config_" + time_str + ".txt",
    )
    write_config_file(params, path_config)

    test_log = []
    test_log = log(
        "Test Two-Body Rendezvous Hamiltonian Controller", test_log, flag_report_live
    )

    plot.style.use(os.path.join("data", "support_files", "dark_scientific.mplstyle"))

    env = TwoBodyRendezvous_Env(
        mu=params["mu"],  # solar gravitational parameter in m^3/s^2
        max_T=params["max_T"],  # max thrust in N
        ISP=params["ISP"],  # ISP in seconds
        l_star=params["l_star"],  # characteristic length in m
        m_star=params["m_star"],  # characteristic mass in kg
        t_star=params["t_star"],  # characteristic time in s
        g0=params["g0"],  # gravitational acceleration at Earth surface in m/s^2
        step_size=params["env_step_size"],  # environment step size in seconds
        a_min_env_nd=params["a_min_env_nd"],  # min semi-major axis for env [AU]
        a_max_env_nd=params["a_max_env_nd"],  # max semi-major axis for env [AU]
        e_min_env=params["e_min_env"],  # min eccentricity for env
        e_max_env=params["e_max_env"],  # max eccentricity for env
        w_min_env_deg=params[
            "w_min_env_deg"
        ],  # min argument of periapsis for env [deg]
        w_max_env_deg=params[
            "w_max_env_deg"
        ],  # max argument of periapsis for env [deg]
    )

    arr_pass_count = []
    sa_output_ephems = []

    for traj_num in range(params["num_trajs"]):
        print(f"\n\nStarting trajectory {traj_num + 1} of {params['num_trajs']}")

        if params["randomize_seeds"]:
            seed_traj = np.random.randint(0, 2**31 - 1)
        else:
            seed_traj = params["seed_env_init"] + traj_num

        if params["randomize_tofs"]:
            tof_scale = np.random.choice(params["tof_scales"])
        else:
            tof_scale = params["tof_scales"][0]

        # str_gen_time = time.strftime("%Y%m%d_%H%M%S")
        test_log = log(
            f"\n\nStarting trajectory {traj_num + 1} with seed {seed_traj}",
            test_log,
            flag_report_live,
        )
        tof_scale_str = str(tof_scale).replace(".", "p")
        ephem_filename = (
            "test_TBR_ephem_traj_seed_" + str(seed_traj) + "_tof_" + tof_scale_str
        )

        flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(
            env,
            seed_traj,
            tof_scale,
            params,
            ephem_filename,
            test_log,
            flag_report_live=flag_report_live,
        )

        # check if solved
        str_gen_time = time.strftime("%b %d %Y %H:%M:%S")
        if flag_solved == True:
            print(
                f"[{str_gen_time}] Trajectory seed {seed_traj} solved successfully.\n"
            )
            arr_pass_count.append(1)
            sa_output_ephems.append(
                os.path.join(params["data_path"], ephem_filename + ".txt")
            )
            if params["flag_plot_traj"] == True and eph_output is not None:
                eph_output.save_plots(params["data_path"], ephem_filename, params, env)

        else:
            print(
                f"[{str_gen_time}] Trajectory seed {seed_traj} failed to solve, moving on.\n"
            )
            arr_pass_count.append(0)

        seed_traj += 1  # increment trajectory seed

        # close any open plots
        plot.close("all")

    total_pass = sum(arr_pass_count)

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
        f"Total elapsed time: {elapsed_time / 60.0:.2f} minutes",
        test_log,
        flag_report_live,
    )
    if total_pass > 0:
        test_log = log(
            "Average time per successful trajectory: "
            + str(elapsed_time / total_pass)
            + " seconds",
            test_log,
            flag_report_live,
        )

    # write test log to file
    path_log = os.path.join(
        params["data_path"], "datagen_Hamiltonian_TBR_controller_log.txt"
    )
    write_log_to_file(path_log, test_log)


datagen_Hamiltonian_TBR_controller()  # Set to True for verbose output

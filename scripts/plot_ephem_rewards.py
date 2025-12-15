import os

import matplotlib.pyplot as plt

from astro_compass.core.training_data_generation import read_ephems_from_dir
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.h_rl_fusion import calc_rewards_from_H_ephem
from astro_compass.utils.log_utils import read_config_file
from astro_compass.utils.plotting_utils import plot_rendezvous_traj


def plot_H_ephem_rewards():
    print("Plotting H ephemeris rewards")

    num_ephems = 2
    number_of_vectors = 1500
    dir_ephems = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\temp\\"
    print("Reading ephemerides from directory: ", dir_ephems)
    set_ephems = read_ephems_from_dir(dir_ephems, num_ephems, version=2.0)
    print(f"Read {len(set_ephems)} ephemerides")
    # config path
    path_config = os.path.join(
        "data", "config", "SAC_training_TBR_polar__env2_config.txt"
    )

    # define normalization parameters (for NN)
    params = read_config_file(path_config)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    params["number_of_vectors_plot"] = number_of_vectors

    # Plot rewards
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax1.set_xlabel("Elapsed Time (days)")
    ax1.set_ylabel("Reward")
    ax1.set_title("Ephemeris Rewards Over Time")
    ax1.legend()
    ax2.set_xlabel("Elapsed Time (days)")
    ax2.set_ylabel("Cumulative Reward")
    plt.grid()

    # Plot components
    fig2, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    ax3, ax4, ax5, ax6 = axes.flatten()
    ax3.set_xlabel("Elapsed Time (days)")
    ax3.set_ylabel("Position Reward Component")
    ax4.set_xlabel("Elapsed Time (days)")
    ax4.set_ylabel("Velocity Reward Component")
    ax5.set_xlabel("Elapsed Time (days)")
    ax5.set_ylabel("Throttle Reward Component")
    ax6.set_xlabel("Elapsed Time (days)")
    ax6.set_ylabel("Overlayed Reward Components")

    # plot residual
    fig3, ax7 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax7.set_xlabel("Elapsed Time (days)")
    ax7.set_ylabel("Position Residual")

    # plot residual
    fig4, ax8 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax8.set_xlabel("Elapsed Time (days)")
    ax8.set_ylabel("Time Component Reward")

    fig5, ax9 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax9.set_xlabel("Target X")
    ax9.set_ylabel("Target Y")
    ax9.axis("equal")

    fig6, ax10 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax10.set_xlabel("Elapsed Time (days)")
    ax10.set_ylabel("Time to Go (days)")

    fig7, ax11 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax11.set_xlabel("Elapsed Time (days)")
    ax11.set_ylabel("Terminated")

    for i in range(len(set_ephems)):
        ephem_H = set_ephems[i]
        print(
            f"Processing ephemeris {i + 1}/{len(set_ephems)} with {ephem_H.num_vectors} vectors"
        )

        # Create environment

        env = gen_rl_environment(params)

        # Calculate rewards from ephemeris
        [
            arr_elapsed_time,
            arr_rewards,
            arr_pos_rewards,
            arr_vel_rewards,
            arr_mass_rewards,
            arr_throttle_rewards,
            arr_time_rewards,
            arr_r_tot,
            arr_position_res,
            arr_target_x_current,
            arr_target_y_current,
            arr_target_vx_current,
            arr_target_vy_current,
            arr_ttg,
            arr_x_current,
            arr_y_current,
            arr_terminated,
        ] = calc_rewards_from_H_ephem(ephem_H, env, params)

        # Plot total rewards
        current_color = colors[i % len(colors)]
        ax1.plot(
            arr_elapsed_time, arr_rewards, current_color, label=f"Ephem {i + 1} Reward"
        )
        ax2.plot(
            arr_elapsed_time,
            arr_r_tot,
            current_color,
            label=f"Ephem {i + 1} Cumulative Reward",
        )

        # Plot components
        ax3.plot(
            arr_elapsed_time,
            arr_pos_rewards,
            label=f"Ephem {i + 1} Pos Reward",
            color=current_color,
        )
        ax4.plot(
            arr_elapsed_time,
            arr_vel_rewards,
            label=f"Ephem {i + 1} Vel Reward",
            color=current_color,
        )
        ax5.plot(
            arr_elapsed_time,
            arr_throttle_rewards,
            label=f"Ephem {i + 1} Throttle Reward",
            color=current_color,
        )
        ax6.plot(
            arr_elapsed_time,
            arr_pos_rewards,
            label=f"Ephem {i + 1} Pos Reward",
            color=current_color,
            linestyle="dashed",
        )
        ax6.plot(
            arr_elapsed_time,
            arr_vel_rewards,
            label=f"Ephem {i + 1} Vel Reward",
            color=current_color,
            linestyle="dotted",
        )
        ax6.plot(
            arr_elapsed_time,
            arr_throttle_rewards,
            label=f"Ephem {i + 1} Throttle Reward",
            color=current_color,
        )
        ax6.plot(
            arr_elapsed_time,
            arr_time_rewards,
            label=f"Ephem {i + 1} Time Multiplier",
            color="black",
            linestyle="dashdot",
        )

        ax7.plot(
            arr_elapsed_time,
            arr_position_res,
            label=f"Ephem {i + 1} Position Residual",
            color=current_color,
        )

        ax8.plot(
            arr_elapsed_time,
            arr_time_rewards,
            label=f"Ephem {i + 1} Time Component Reward",
            color=current_color,
        )

        ax9.plot(
            arr_x_current,
            arr_y_current,
            label=f"Ephem {i + 1} Chaser Traj",
            color=current_color,
            linestyle="dashed",
            linewidth=2.0,
        )
        ax9.plot(
            arr_target_x_current,
            arr_target_y_current,
            label=f"Ephem {i + 1} Target Traj",
            color=current_color,
        )

        ax10.plot(
            arr_elapsed_time,
            arr_ttg,
            label=f"Ephem {i + 1} Time to Go",
            color=current_color,
        )

        ax11.plot(
            arr_elapsed_time,
            arr_terminated,
            label=f"Ephem {i + 1} Terminated",
            color=current_color,
        )

    fig_orb = plot_rendezvous_traj(ephem_H, env, params)
    plt.show()


plot_H_ephem_rewards()

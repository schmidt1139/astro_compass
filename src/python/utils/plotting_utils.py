import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants.constants import Constants
from utils.h_rl_fusion import calc_rewards_from_H_ephem
from utils.state_vector_utils import convert_alpha_from_cart_to_fpa


def format_plots():
    matplotlib.rcParams.update(
        {
            "text.usetex": False,  # Use LaTeX for all text
            "font.family": "serif",  # Use serif font
            "font.size": 10,  # Match AIAA body font size
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "figure.figsize": (3.5, 2.5),  # Single-column figure
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,  # No gridlines in AIAA style
        }
    )


def plot_training_loss(arr_epochs, arr_loss_train, arr_loss, path_plot, params):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(arr_epochs, arr_loss, label="Eval Loss", color="orange")
    ax.semilogy(
        arr_epochs, arr_loss_train, label="Training Loss", color="blue", linewidth=2
    )
    ax.set_xlabel(r"Training Epochs")
    ax.set_ylabel(r"Loss (" + params["loss"] + ")")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_plot)


class SACRolloutData:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []
        self.arr_throttle = []
        self.arr_alpha_x = []
        self.arr_alpha_y = []
        self.arr_x = []
        self.arr_y = []
        self.arr_vx = []
        self.arr_vy = []
        self.arr_sma = []
        self.arr_sma_target = []
        self.arr_ecc = []
        self.arr_ecc_target = []
        self.arr_ecc_max = []
        self.arr_reward_mass = []
        self.arr_reward_distance = []
        self.sum_reward = 0.0

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_x,
        alpha_y,
        x,
        y,
        vx,
        vy,
        sma,
        sma_target,
        ecc,
        ecc_target,
        ecc_max,
        reward_mass,
        reward_distance,
    ):
        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_x.append(alpha_x)
        self.arr_alpha_y.append(alpha_y)
        self.arr_x.append(x)
        self.arr_y.append(y)
        self.arr_vx.append(vx)
        self.arr_vy.append(vy)
        self.arr_sma.append(sma)
        self.arr_sma_target.append(sma_target)
        self.arr_ecc.append(ecc)
        self.arr_ecc_target.append(ecc_target)
        self.arr_ecc_max.append(ecc_max)
        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)
        self.arr_reward_mass.append(reward_mass)
        self.arr_reward_distance.append(reward_distance)


class SACRolloutData_TBR:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []
        self.arr_throttle = []
        self.arr_alpha_x = []
        self.arr_alpha_y = []
        self.arr_x = []
        self.arr_y = []
        self.arr_vx = []
        self.arr_vy = []
        self.arr_m = []
        self.arr_x_target = []
        self.arr_y_target = []
        self.arr_vx_target = []
        self.arr_vy_target = []
        self.arr_ttg = []
        self.arr_pos_reward = []
        self.arr_vel_reward = []
        self.arr_mass_reward = []
        self.sum_reward = 0.0

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_x,
        alpha_y,
        x,
        y,
        vx,
        vy,
        m,
        x_target,
        y_target,
        vx_target,
        vy_target,
        ttg,
        pos_reward,
        vel_reward,
        mass_reward,
    ):
        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_x.append(alpha_x)
        self.arr_alpha_y.append(alpha_y)
        self.arr_x.append(x)
        self.arr_y.append(y)
        self.arr_vx.append(vx)
        self.arr_vy.append(vy)
        self.arr_m.append(m)
        self.arr_x_target.append(x_target)
        self.arr_y_target.append(y_target)
        self.arr_vx_target.append(vx_target)
        self.arr_vy_target.append(vy_target)
        self.arr_ttg.append(ttg)
        self.arr_pos_reward.append(pos_reward)
        self.arr_vel_reward.append(vel_reward)
        self.arr_mass_reward.append(mass_reward)
        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)


class SACRolloutData_TBR_polar:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []

        self.arr_throttle = []
        self.arr_alpha_r = []
        self.arr_alpha_theta = []

        self.arr_rad = []
        self.arr_cos_theta = []
        self.arr_sin_theta = []
        self.arr_v = []
        self.arr_cos_fpa = []
        self.arr_sin_fpa = []
        self.arr_mass = []

        self.arr_rad_f = []
        self.arr_cos_theta_f = []
        self.arr_sin_theta_f = []
        self.arr_v_f = []
        self.arr_cos_fpa_f = []
        self.arr_sin_fpa_f = []
        self.arr_ttg = []

        self.arr_pos_reward = []
        self.arr_vel_reward = []
        self.arr_mass_reward = []
        self.arr_throttle_reward = []
        self.sum_reward = 0.0

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_r,
        alpha_theta,
        rad,
        cos_theta,
        sin_theta,
        v,
        cos_fpa,
        sin_fpa,
        m,
        rad_f,
        cos_theta_f,
        sin_theta_f,
        v_f,
        cos_fpa_f,
        sin_fpa_f,
        ttg,
        pos_reward,
        vel_reward,
        mass_reward,
        throttle_reward,
    ):
        alpha_fpa = np.sqrt(alpha_r**2 + alpha_theta**2)
        alpha_r /= alpha_fpa
        alpha_theta /= alpha_fpa

        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_r.append(alpha_r)
        self.arr_alpha_theta.append(alpha_theta)
        self.arr_rad.append(rad)
        self.arr_cos_theta.append(cos_theta)
        self.arr_sin_theta.append(sin_theta)
        self.arr_v.append(v)
        self.arr_cos_fpa.append(cos_fpa)
        self.arr_sin_fpa.append(sin_fpa)
        self.arr_mass.append(m)
        self.arr_rad_f.append(rad_f)
        self.arr_cos_theta_f.append(cos_theta_f)
        self.arr_sin_theta_f.append(sin_theta_f)
        self.arr_v_f.append(v_f)
        self.arr_cos_fpa_f.append(cos_fpa_f)
        self.arr_sin_fpa_f.append(sin_fpa_f)
        self.arr_ttg.append(ttg)
        self.arr_pos_reward.append(pos_reward)
        self.arr_vel_reward.append(vel_reward)
        self.arr_mass_reward.append(mass_reward)
        self.arr_throttle_reward.append(throttle_reward)

        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)


def plot_SAC_training(
    SACRolloutData, arr_episode_numbers, arr_episode_rs, path_output, eph, eph_h=None
):
    # plot reward over time
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_reward_tot, label="Reward")
    plt.xlabel("Time [days]")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward.png"), dpi=300)

    # plot reward over time per step
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_reward, label="Reward")
    plt.plot(
        SACRolloutData.arr_time,
        SACRolloutData.arr_reward_mass,
        label="Reward Mass Component",
    )
    plt.plot(
        SACRolloutData.arr_time,
        SACRolloutData.arr_reward_distance,
        label="Reward Distance Component",
    )
    plt.xlabel("Time [days]")
    plt.ylabel("Reward per Step")
    plt.title("SAC Training Reward Per Step over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300)

    # plot throttle over time
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_throttle, label="Throttle")
    plt.xlabel("Time [days]")
    plt.ylabel("Throttle")
    plt.title("SAC Training Throttle over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Throttle.png"), dpi=300)

    # plot attitude over time
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_alpha_x, label="alpha_x")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_alpha_y, label="alpha_y")
    plt.xlabel("Time [days]")
    plt.ylabel("Attitude")
    plt.title("SAC Training Burn Attitude over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Alpha.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_x, label="x")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_y, label="y")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_vx, label="vx")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_vy, label="vy")
    plt.xlabel("Time [days]")
    plt.ylabel("ND state")
    plt.title("SAC Training ND State over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_ND_State.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_sma, label="sma")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_sma_target, label="sma_target")
    plt.xlabel("Time [days]")
    plt.ylabel("SMA Achieved [m]")
    plt.title("SAC Achieved SMA over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_SMA_Achieved.png"), dpi=300)

    plt.figure()
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_ecc, label="ecc")
    plt.plot(SACRolloutData.arr_time, SACRolloutData.arr_ecc_target, label="ecc_target")
    plt.plot(
        SACRolloutData.arr_time,
        SACRolloutData.arr_ecc_max,
        label="ecc_max",
        linestyle="--",
        color="red",
    )
    plt.xlabel("Time [days]")
    plt.ylabel("ECC Achieved")
    plt.title("SAC Achieved ECC over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_ECC_Achieved.png"), dpi=300)

    plt.figure()
    plt.plot(arr_episode_numbers, arr_episode_rs, label="Training Reward per Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("SAC Reward Per Episode During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(
        os.path.join(path_output, "SAC_Training_reward_per_episode.png"), dpi=300
    )

    # generate and save figures
    fig_orb = eph.plot_xy()
    if eph_h is not None:
        fig_orb = eph.overlay_ref_orbit(
            ephem=eph_h, label="Hamiltonian Trajectory", color_in="#f89540"
        )
    fig_orb = eph.plot_xy_ref_orbit(Constants.SMA_MARS, "Mars", "#b7410e")
    fig_orb = eph.plot_xy_ref_orbit(Constants.SMA_EARTH, "Earth")
    fig_orb.savefig(os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300)


def plot_SAC_training_TBR(
    SACRolloutData_TBR,
    arr_episode_numbers,
    arr_episode_rs,
    path_output,
    eph,
    params,
    env,
    arr_actor_loss_pt,
    arr_critic_loss_pt,
    ephem_H=None,
):
    if ephem_H is not None:
        results = calc_rewards_from_H_ephem(ephem_H, env, params)
        arr_time_H = results[0]
        arr_reward_H = results[1]
        arr_r_pos_H = results[2]
        arr_r_vel_H = results[3]
        arr_r_mass_H = results[4]
        arr_r_tot = results[5]
        arr_u_H = ephem_H.arr_u
        arr_alpha_x_H = ephem_H.arr_alpha_x
        arr_alpha_y_H = ephem_H.arr_alpha_y

    # plot reward over time
    plt.figure()
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_reward_tot,
        label="Reward",
    )
    if ephem_H is not None:
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_r_tot,
            label="Hamiltonian Ephem Reward",
            linestyle="--",
            color="red",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward.png"), dpi=300)

    plt.figure()
    arr_ttg_days = [
        ttg * params["t_star"] / Constants.DAYS_TO_SEC
        for ttg in SACRolloutData_TBR.arr_ttg
    ]
    arr_zeros = [0.0 for ttg in SACRolloutData_TBR.arr_ttg]

    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        np.array(arr_ttg_days) / 365.25,
        label="Time to Target",
        color="magenta",
    )
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        arr_zeros,
        label="Target Reached",
        linestyle="--",
        color="orange",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("Time to Target [years]")
    plt.title("SAC Training Time to Target over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Time_to_Target.png"), dpi=300)

    # plot reward over time per step
    plt.figure()
    percentile_95_SAC = np.percentile(SACRolloutData_TBR.arr_reward, 95)
    percentile_5_SAC = np.percentile(SACRolloutData_TBR.arr_reward, 5)
    if ephem_H is not None:
        percentile_95_H = np.percentile(arr_reward_H, 95)
        percentile_5_H = np.percentile(arr_reward_H, 5)
        max_percentile = max(percentile_95_SAC, percentile_95_H)
        min_percentile = min(percentile_5_SAC, percentile_5_H)
    else:
        max_percentile = percentile_95_SAC
        min_percentile = percentile_5_SAC

    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_reward,
        label="Reward",
    )
    if ephem_H is not None:
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_reward_H,
            label="Hamiltonian Ephem Reward",
            linestyle="--",
            color="red",
        )
    plt.ylim(min_percentile, 1.1 * max_percentile)
    plt.xlabel("Time [years]")
    plt.ylabel("Reward per Step")
    plt.title("SAC Training Reward Per Step over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300)

    plt.figure()
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_pos_reward,
        label="Position r component",
    )
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_vel_reward,
        label="Velocity r component",
    )
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_mass_reward,
        label="Mass r component",
    )
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_reward,
        label="Composite Reward",
    )
    if ephem_H is not None:
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_r_pos_H,
            label="Hamiltonian Ephem Position r component",
            linestyle="--",
            color="red",
        )
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_r_vel_H,
            label="Hamiltonian Ephem Velocity r component",
            linestyle="--",
            color="blue",
        )
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_r_mass_H,
            label="Hamiltonian Ephem Mass r component",
            linestyle="--",
            color="green",
        )
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_reward_H,
            label="Hamiltonian Ephem Composite Reward",
            linestyle="--",
            color="orange",
        )
    percentile_95_H = np.percentile(arr_reward_H, 95)
    percentile_95_SAC = np.percentile(SACRolloutData_TBR.arr_reward, 95)
    max_percentile = max(percentile_95_SAC, percentile_95_H)
    if ephem_H is not None:
        percentile_95_H = np.percentile(arr_reward_H, 95)
        percentile_5_H = np.percentile(arr_reward_H, 5)
        max_percentile = max(percentile_95_SAC, percentile_95_H)
        min_percentile = min(percentile_5_SAC, percentile_5_H)
    else:
        max_percentile = percentile_95_SAC
        min_percentile = percentile_5_SAC
    # plt.ylim(min_percentile, 1.1*max_percentile)
    plt.xlabel("Time [years]")
    plt.ylabel("Reward Contribution")
    plt.title("SAC Training Reward Contribution over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(
        os.path.join(path_output, "SAC_Training_Reward_Contribution.png"), dpi=300
    )

    # plot throttle over time
    plt.figure()
    plt.plot(
        np.array(SACRolloutData_TBR.arr_time) / 365.25,
        SACRolloutData_TBR.arr_throttle,
        label="Throttle",
    )
    if ephem_H is not None:
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_u_H,
            label="Hamiltonian Ephem Throttle",
            linestyle="--",
            color="red",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Throttle")
    plt.title("SAC Training Throttle over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Throttle.png"), dpi=300)

    # plot attitude over time
    plt.figure()
    arr_time_years = np.array(SACRolloutData_TBR.arr_time) / 365.25
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_alpha_x, label="alpha_x")
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_alpha_y, label="alpha_y")
    if ephem_H is not None:
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_alpha_x_H,
            label="Hamiltonian Ephem alpha_x",
            linestyle="--",
            color="red",
        )
        plt.plot(
            np.array(arr_time_H) / 365.25,
            arr_alpha_y_H,
            label="Hamiltonian Ephem alpha_y",
            linestyle="--",
            color="blue",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Attitude")
    plt.title("SAC Training Burn Attitude over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Alpha.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_x, label="x", color="cyan")
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_y, label="y", color="magenta")
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_vx, label="vx", color="orange")
    plt.plot(arr_time_years, SACRolloutData_TBR.arr_vy, label="vy", color="pink")
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR.arr_x_target,
        label="x_target",
        linestyle="--",
        color="cyan",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR.arr_y_target,
        label="y_target",
        linestyle="--",
        color="magenta",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR.arr_vx_target,
        label="vx_target",
        linestyle="--",
        color="orange",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR.arr_vy_target,
        label="vy_target",
        linestyle="--",
        color="pink",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("ND state")
    plt.title("SAC Training ND State over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_ND_State.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    plt.plot(arr_episode_numbers, arr_episode_rs, label="Training Reward per Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("SAC Reward Per Episode During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(
        os.path.join(path_output, "SAC_Training_reward_per_episode.png"), dpi=300
    )

    # Create 2x1 subplot for actor and critic losses
    if len(arr_critic_loss_pt) == 0 or len(arr_actor_loss_pt) == 0:
        print("No pre-training loss data to plot.")
    else:
        print("Plotting pre-training actor and critic losses.")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

        # Plot critic loss on top
        ax1.plot(arr_critic_loss_pt, label="Critic Loss", color="blue")
        ax1.set_xlabel(
            "Iterations (updated every " + str(params["loss_report_rate"]) + " steps)"
        )
        ax1.set_ylabel("Critic Loss")
        ax1.set_title("Pre-Training Critic Loss vs Iterations")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot actor loss on bottom
        ax2.plot(arr_actor_loss_pt, label="Actor Loss", color="orange")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Actor Loss")
        ax2.set_title("Pre-Training Actor Loss vs Iterations")
        ax2.set_xscale("log")
        # ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.savefig(os.path.join(path_output, "SAC_Actor_Critic_Losses.png"), dpi=300)
        plt.close(fig)

    # generate and save figures
    fig_orb = eph.plot_xy(color_in="#7e03a8")
    x_target = SACRolloutData_TBR.arr_x_target[-1] * params["l_star"]
    y_target = SACRolloutData_TBR.arr_y_target[-1] * params["l_star"]
    vx_target = (
        SACRolloutData_TBR.arr_vx_target[-1] * params["l_star"] / params["t_star"]
    )
    vy_target = (
        SACRolloutData_TBR.arr_vy_target[-1] * params["l_star"] / params["t_star"]
    )
    fig_orb = plot_overlay_ballistic_orbit(
        x_target,
        y_target,
        vx_target,
        vy_target,
        env,
        fig_orb,
        params,
        eph,
        label_in="Target Orbit",
        color_in="#cc4778",
    )
    fig_orb = plot_overlay_ballistic_orbit(
        SACRolloutData_TBR.arr_x[0] * params["l_star"],
        SACRolloutData_TBR.arr_y[0] * params["l_star"],
        SACRolloutData_TBR.arr_vx[0] * params["l_star"] / params["t_star"],
        SACRolloutData_TBR.arr_vy[0] * params["l_star"] / params["t_star"],
        env,
        fig_orb,
        params,
        eph,
        label_in="Initial Orbit",
        color_in="#0d0887",
    )
    fig_orb = eph.add_target_icon(x_target, y_target, color_in="#cc4778")

    if params.get("flag_gen_H_traj", False) and (ephem_H is not None):
        fig_orb = eph.overlay_ref_orbit(
            ephem=ephem_H, label="Hamiltonian Trajectory", color_in="#f89540"
        )

    fig_orb = eph.adjust_plot_limits()
    fig_orb.savefig(
        os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig_orb)

    # write reward data to csv - handle different length arrays
    sac_time = np.array(SACRolloutData_TBR.arr_time) / 365.25
    sac_reward = SACRolloutData_TBR.arr_reward
    h_time = np.array(arr_time_H) / 365.25
    h_reward = arr_reward_H

    # downsample to every 10th point for saving
    sac_time = sac_time[::10]
    sac_reward = sac_reward[::10]
    h_time = h_time[::10]
    h_reward = h_reward[::10]
    arr_r_pos_H = arr_r_pos_H[::10]
    arr_r_vel_H = arr_r_vel_H[::10]
    arr_r_mass_H = arr_r_mass_H[::10]

    # Find max length AFTER downsampling
    max_len = max(len(sac_time), len(h_time))

    df = pd.DataFrame(
        {
            "Time_years": list(sac_time) + [np.nan] * (max_len - len(sac_time)),
            "SAC_Reward_per_Step": list(sac_reward)
            + [np.nan] * (max_len - len(sac_reward)),
            "arr_time_H_years": list(h_time) + [np.nan] * (max_len - len(h_time)),
            "arr_r_pos_H_per_step": list(arr_r_pos_H)
            + [np.nan] * (max_len - len(arr_r_pos_H)),
            "arr_r_vel_H_per_step": list(arr_r_vel_H)
            + [np.nan] * (max_len - len(arr_r_vel_H)),
            "arr_r_mass_H_per_step": list(arr_r_mass_H)
            + [np.nan] * (max_len - len(arr_r_mass_H)),
            "Hamiltonian_Ephem_Reward_per_Step": list(h_reward)
            + [np.nan] * (max_len - len(h_reward)),
            "Cumulative_Hamiltonian_Ephem_Reward": list(np.cumsum(h_reward))
            + [np.nan] * (max_len - len(h_reward)),
        }
    )
    df.to_csv(os.path.join(path_output, "SAC_Rewards.csv"), index=False)


def plot_SAC_training_TBR_polar(
    SACRolloutData_TBR_polar,
    path_output,
    eph,
    params,
    env,
    arr_episode_numbers=None,
    arr_episode_rs=None,
    arr_actor_loss_pt=None,
    arr_critic_loss_pt=None,
    ephem_H=None,
):
    if ephem_H is not None:
        results = calc_rewards_from_H_ephem(ephem_H, env, params)
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
        ] = results


    #SAC rollout data
    elapsed_time_SAC = SACRolloutData_TBR_polar.arr_time[-1]*Constants.DAYS_TO_SEC
    num_steps_SAC = len(SACRolloutData_TBR_polar.arr_time)
    average_step_size_SAC = elapsed_time_SAC / num_steps_SAC
    print(f"SAC rollout total elapsed time: {elapsed_time_SAC/Constants.DAYS_TO_SEC/365.25:.2f} years over {num_steps_SAC} steps.")
    print(f"Average step size: {average_step_size_SAC} s")

    if ephem_H is not None:
        elapsed_time_h_ephem = ephem_H.arr_et[-1];
        num_vectors = ephem_H.num_vectors
        average_step_size_h = elapsed_time_h_ephem / num_vectors
        print(f"Hamiltonian ephemeris total elapsed time: {elapsed_time_h_ephem/Constants.DAYS_TO_SEC/365.25:.2f} years over {num_vectors} vectors.")
        print(f"Average step size: {average_step_size_h} s")
        reward_reduction_factor = average_step_size_h / average_step_size_SAC
        arr_r_tot = [r * reward_reduction_factor for r in arr_r_tot]
        print(f"Applied reward reduction factor of {reward_reduction_factor:.4f} to Hamiltonian ephemeris rewards to account for differing step sizes.")
        
    # plot reward over time
    plt.figure()
    plt.plot(
        np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
        SACRolloutData_TBR_polar.arr_reward_tot,
        label="Reward",
    )
    if ephem_H is not None:
        plt.plot(np.array(arr_elapsed_time)/365.25, arr_r_tot, label="Hamiltonian Ephem Total Reward", linestyle="--", color="red")
    plt.xlabel("Time [years]")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward.png"), dpi=300)

    plt.figure()
    arr_ttg_days = [
        ttg * params["t_star"] / Constants.DAYS_TO_SEC
        for ttg in SACRolloutData_TBR_polar.arr_ttg
    ]
    arr_zeros = [0.0 for ttg in SACRolloutData_TBR_polar.arr_ttg]

    plt.plot(
        np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
        np.array(arr_ttg_days) / 365.25,
        label="Time to Target",
        color="magenta",
    )
    plt.plot(
        np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
        arr_zeros,
        label="Target Reached",
        linestyle="--",
        color="orange",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("Time to Target [years]")
    plt.title("SAC Training Time to Target over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Time_to_Target.png"), dpi=300)

    # plot reward over time per step
    try:
        plt.figure()
        percentile_95_SAC = np.percentile(SACRolloutData_TBR_polar.arr_reward, 99)
        percentile_5_SAC = np.percentile(SACRolloutData_TBR_polar.arr_reward, 1)
        # if ephem_H is not None:
        #     percentile_95_H = np.percentile(arr_rewards_H, 99)
        #     percentile_5_H = np.percentile(arr_rewards_H, 1)
        #     max_percentile = max(percentile_95_SAC, percentile_95_H)
        #     min_percentile = min(percentile_5_SAC, percentile_5_H)
        # else:
        #     max_percentile = percentile_95_SAC
        #     min_percentile = percentile_5_SAC

        plt.plot(
            np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
            SACRolloutData_TBR_polar.arr_reward,
            label="Reward",
        )
        if ephem_H is not None:
            plt.plot(
                np.array(arr_elapsed_time) / 365.25,
                arr_rewards,
                label="Hamiltonian Ephem Reward",
                linestyle="--",
                color="red",
            )
        # plt.ylim(min_percentile, 1.1*max_percentile)
        plt.xlabel("Time [years]")
        plt.ylabel("Reward per Step")
        plt.title("SAC Training Reward Per Step over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300
        )

        plt.figure()
        plt.plot(
            np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
            SACRolloutData_TBR_polar.arr_pos_reward,
            label="Position r component",
        )
        plt.plot(
            np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
            SACRolloutData_TBR_polar.arr_vel_reward,
            label="Velocity r component",
        )
        plt.plot(
            np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
            SACRolloutData_TBR_polar.arr_throttle_reward,
            label="Throttle r component",
        )
        plt.plot(
            np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
            SACRolloutData_TBR_polar.arr_reward,
            label="Composite Reward",
        )
        if ephem_H is not None:
            plt.plot(
                np.array(arr_elapsed_time) / 365.25,
                arr_pos_rewards,
                label="Hamiltonian Ephem Position r component",
                linestyle="--",
                color="red",
            )
            plt.plot(
                np.array(arr_elapsed_time) / 365.25,
                arr_vel_rewards,
                label="Hamiltonian Ephem Velocity r component",
                linestyle="--",
                color="blue",
            )
            plt.plot(
                np.array(arr_elapsed_time) / 365.25,
                arr_throttle_rewards,
                label="Hamiltonian Ephem Throttle r component",
                linestyle="--",
                color="green",
            )
            plt.plot(
                np.array(arr_elapsed_time) / 365.25,
                arr_rewards,
                label="Hamiltonian Ephem Composite Reward",
                linestyle="--",
                color="orange",
            )
            # percentile_95_H = np.percentile(arr_rewards_H, 99)
            # percentile_95_SAC = np.percentile(SACRolloutData_TBR_polar.arr_reward, 99)
            # max_percentile = max(percentile_95_SAC, percentile_95_H)
        # else:
        # max_percentile = percentile_95_SAC
        # min_percentile = percentile_5_SAC

        # if ephem_H is not None:
        #     percentile_95_H = np.percentile(arr_rewards_H, 99)
        #     percentile_5_H = np.percentile(arr_rewards_H, 1)
        #     max_percentile = max(percentile_95_SAC, percentile_95_H)
        #     min_percentile = min(percentile_5_SAC, percentile_5_H)
        # else:
        #     max_percentile = percentile_95_SAC
        #     min_percentile = percentile_5_SAC

        # plt.ylim(min_percentile, 1.1*max_percentile)
        plt.xlabel("Time [years]")
        plt.ylabel("Reward Contribution")
        plt.title("SAC Training Reward Contribution over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(path_output, "SAC_Training_Reward_Contribution.png"), dpi=300
        )

    except Exception as e:
        print(f"Error saving SAC Training Reward plots: {e}")

    # plot throttle over time
    plt.figure()
    plt.plot(
        np.array(SACRolloutData_TBR_polar.arr_time) / 365.25,
        SACRolloutData_TBR_polar.arr_throttle,
        label="Throttle",
    )
    if ephem_H is not None:
        plt.plot(
            np.array(arr_elapsed_time) / 365.25,
            ephem_H.arr_u,
            label="Hamiltonian Ephem Throttle",
            linestyle="--",
            color="red",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Throttle")
    plt.title("SAC Training Throttle over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Throttle.png"), dpi=300)

    # plot attitude over time
    plt.figure()
    arr_time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25
    plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_alpha_r, label="SAC alpha_r")
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_alpha_theta,
        label="SAC alpha_theta",
    )

    if ephem_H is not None:
        arr_alpha_fpa_cos_H = []
        arr_alpha_fpa_sin_H = []
        for i in range(len(ephem_H.arr_alpha_x)):
            x_i = ephem_H.arr_x[i]
            y_i = ephem_H.arr_y[i]
            vx_i = ephem_H.arr_vx[i]
            vy_i = ephem_H.arr_vy[i]
            alpha_x = ephem_H.arr_alpha_x[i]
            alpha_y = ephem_H.arr_alpha_y[i]

            alpha_fpa_cos, alpha_fpa_sin = convert_alpha_from_cart_to_fpa(
                x_i, y_i, vx_i, vy_i, alpha_x, alpha_y
            )
            arr_alpha_fpa_cos_H.append(alpha_fpa_cos)
            arr_alpha_fpa_sin_H.append(alpha_fpa_sin)

        plt.plot(
            np.array(arr_elapsed_time) / 365.25,
            arr_alpha_fpa_cos_H,
            label="Hamiltonian Ephem alpha_fpa_cos",
            linestyle="--",
            color="red",
        )
        plt.plot(
            np.array(arr_elapsed_time) / 365.25,
            arr_alpha_fpa_sin_H,
            label="Hamiltonian Ephem alpha_fpa_sin",
            linestyle="--",
            color="blue",
        )

    plt.xlabel("Time [years]")
    plt.ylabel("Attitude")
    plt.title("SAC Training Burn Attitude over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Alpha.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_rad, label="r", color="cyan")
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_cos_theta, label="theta_cos", color="magenta")
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_sin_theta, label="theta_sin", color="orange")
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_v, label="v", color="pink")
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_cos_fpa, label="fpa_cos", linestyle="--", color="cyan")
    # plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_sin_fpa, label="fpa_sin", linestyle="--", color="magenta")
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_rad_f,
        label="r_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_cos_theta_f,
        label="theta_cos_target",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_sin_theta_f,
        label="theta_sin_target",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_v_f,
        label="v_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_cos_fpa_f,
        label="fpa_cos_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_sin_fpa_f,
        label="fpa_sin_target",
        linestyle="--",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("ND state")
    plt.title("SAC Training ND Relative State over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_ND_Rel_State.png"), dpi=300)

    plt.figure()
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_rad,
        label="r_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years, SACRolloutData_TBR_polar.arr_cos_theta, label="theta_cos_target"
    )
    plt.plot(
        arr_time_years, SACRolloutData_TBR_polar.arr_sin_theta, label="theta_sin_target"
    )
    plt.plot(
        arr_time_years, SACRolloutData_TBR_polar.arr_v, label="v_target", linestyle="--"
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_cos_fpa,
        label="fpa_cos_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_sin_fpa,
        label="fpa_sin_target",
        linestyle="--",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("ND state")
    plt.title("SAC Training ND State over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_ND_State.png"), dpi=300)

    # plot nd state over time
    if arr_episode_numbers is not None and arr_episode_rs is not None:
        plt.figure()
        plt.plot(
            arr_episode_numbers, arr_episode_rs, label="Training Reward per Episode"
        )
        plt.xlabel("Episode Number")
        plt.ylabel("Reward")
        plt.title("SAC Reward Per Episode During Training")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(path_output, "SAC_Training_reward_per_episode.png"), dpi=300
        )

    # Create 2x1 subplot for actor and critic losses

    if (
        (arr_critic_loss_pt is None)
        or (arr_actor_loss_pt is None)
        or (len(arr_critic_loss_pt) == 0 or len(arr_actor_loss_pt) == 0)
    ):
        print("No pre-training loss data to plot.")
    else:
        print("Plotting pre-training actor and critic losses.")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

        # Plot critic loss on top
        ax1.plot(arr_critic_loss_pt, label="Critic Loss", color="blue")
        ax1.set_xlabel(
            "Iterations (updated every " + str(params["loss_report_rate"]) + " steps)"
        )
        ax1.set_ylabel("Critic Loss")
        ax1.set_title("Pre-Training Critic Loss vs Iterations")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot actor loss on bottom
        ax2.plot(arr_actor_loss_pt, label="Actor Loss", color="orange")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Actor Loss")
        ax2.set_title("Pre-Training Actor Loss vs Iterations")
        ax2.set_xscale("log")
        # ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.savefig(os.path.join(path_output, "SAC_Actor_Critic_Losses.png"), dpi=300)
        plt.close(fig)

    # generate and save figures
    fig_orb = plot_rendezvous_traj(eph, env, params)
    if params.get("flag_gen_H_traj", False) and (ephem_H is not None):
        fig_orb = eph.overlay_ref_orbit(
            ephem=ephem_H, label="Hamiltonian Trajectory", color_in="#f89540"
        )
    fig_orb.savefig(
        os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig_orb)

    # write reward data to csv - handle different length arrays
    sac_time = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25
    sac_reward = SACRolloutData_TBR_polar.arr_reward
    if ephem_H is not None:
        h_time = np.array(arr_elapsed_time) / 365.25
        h_reward = arr_rewards
        h_time = h_time[::10]
        h_reward = h_reward[::10]
        arr_r_pos_H = arr_pos_rewards[::10]
        arr_r_vel_H = arr_vel_rewards[::10]
        arr_r_mass_H = arr_throttle_rewards[::10]

    # downsample to every 10th point for saving
    sac_time = sac_time[::10]
    sac_reward = sac_reward[::10]

    # Find max length AFTER downsampling
    if ephem_H is not None:
        max_len = max(len(sac_time), len(h_time))
    else:
        max_len = len(sac_time)

    if ephem_H is None:
        df = pd.DataFrame(
            {
                "Time_years": list(sac_time) + [np.nan] * (max_len - len(sac_time)),
                "SAC_Reward_per_Step": list(sac_reward)
                + [np.nan] * (max_len - len(sac_reward)),
            }
        )
        df.to_csv(os.path.join(path_output, "SAC_Rewards.csv"), index=False)

    else:
        df = pd.DataFrame(
            {
                "Time_years": list(sac_time) + [np.nan] * (max_len - len(sac_time)),
                "SAC_Reward_per_Step": list(sac_reward)
                + [np.nan] * (max_len - len(sac_reward)),
                "arr_time_H_years": list(h_time) + [np.nan] * (max_len - len(h_time)),
                "arr_r_pos_H_per_step": list(arr_r_pos_H)
                + [np.nan] * (max_len - len(arr_r_pos_H)),
                "arr_r_vel_H_per_step": list(arr_r_vel_H)
                + [np.nan] * (max_len - len(arr_r_vel_H)),
                "arr_r_mass_H_per_step": list(arr_r_mass_H)
                + [np.nan] * (max_len - len(arr_r_mass_H)),
                "Hamiltonian_Ephem_Reward_per_Step": list(h_reward)
                + [np.nan] * (max_len - len(h_reward)),
                "Cumulative_Hamiltonian_Ephem_Reward": list(np.cumsum(h_reward))
                + [np.nan] * (max_len - len(h_reward)),
            }
        )
        df.to_csv(os.path.join(path_output, "SAC_Rewards.csv"), index=False)


def plot_overlay_ballistic_orbit(
    x, y, vx, vy, env, fig, params, eph, label_in, color_in="lime"
):
    # check env type
    if params.get("env_type", "TwoBodyRendezvous_Env") == "TwoBodyRendezvous_Polar_Env":
        flag_use_obs = False
    elif (
        params.get("env_type", "TwoBodyRendezvous_Env")
        == "TwoBodyRendezvous_Polar_Env2"
    ):
        flag_use_obs = False
    else:
        flag_use_obs = True

    obs, info = env.reset()

    state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]

    unwrapped_env = env.unwrapped
    obs, info = unwrapped_env.set_state(state_in)

    T = info["orbital_period_years"] * Constants.YEARS_TO_SEC

    time = 0.0
    flag_done = False
    arr_x = []
    arr_y = []
    max_steps = 10000  # Safety limit to prevent infinite loop
    step_count = 0

    while not flag_done and step_count < max_steps:
        obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.0])

        if flag_use_obs:
            obs = obs
            # dim state
            x_i = obs[0] * params["l_star"]
            y_i = obs[1] * params["l_star"]
        else:
            unwrapped_env = env.unwrapped
            obs = unwrapped_env.get_cartesian_state()
            x_i = obs[0]
            y_i = obs[1]

        if info["Elapsed time"] >= T or done or truncated:
            flag_done = True

        arr_x.append(x_i)
        arr_y.append(y_i)
        step_count += 1

    fig = eph.overlay_ref_orbit(
        ephem=None, label=label_in, color_in=color_in, arr_x=arr_x, arr_y=arr_y
    )

    return fig


def plot_rendezvous_traj(eph, env, params):
    fig_orb = eph.plot_xy(color_in="#7e03a8")

    x_target = eph.arr_x_target[-1]
    y_target = eph.arr_y_target[-1]
    vx_target = eph.arr_vx_target[-1]
    vy_target = eph.arr_vy_target[-1]

    fig_orb = plot_overlay_ballistic_orbit(
        x_target,
        y_target,
        vx_target,
        vy_target,
        env,
        fig_orb,
        params,
        eph,
        label_in="Target Orbit",
        color_in="#cc4778",
    )

    x_0 = eph.arr_x[0]
    y_0 = eph.arr_y[0]
    vx_0 = eph.arr_vx[0]
    vy_0 = eph.arr_vy[0]

    fig_orb = plot_overlay_ballistic_orbit(
        x_0,
        y_0,
        vx_0,
        vy_0,
        env,
        fig_orb,
        params,
        eph,
        label_in="Initial Orbit",
        color_in="#0d0887",
    )

    fig_orb = eph.add_target_icon(x_target, y_target, color_in="#cc4778")

    fig_orb = eph.adjust_plot_limits()

    return fig_orb

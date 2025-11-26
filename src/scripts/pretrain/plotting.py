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


def _save_figure(path, filename, dpi=300):
    """Helper to save current matplotlib figure to the output path."""

    plt.savefig(os.path.join(path, filename), dpi=dpi)


def _plot_time_series(
    x,
    ys,
    labels,
    xlabel,
    ylabel,
    title,
    path_output,
    filename,
    styles=None,
):
    """Generic helper for simple time-series plots.

    Parameters
    ----------
    x : array-like
        Common x-axis values.
    ys : sequence of array-like
        One or more y-series to plot.
    labels : sequence of str
        Legend labels for each y-series.
    xlabel, ylabel, title : str
        Axis labels and title.
    path_output : str
        Directory where the figure will be written.
    filename : str
        Name of the output file.
    styles : sequence of dict | None
        Optional matplotlib style kwargs per series (e.g. color, linestyle).
    """

    plt.figure()
    styles = styles or [{}] * len(ys)
    for y, label, style in zip(ys, labels, styles):
        plt.plot(x, y, label=label, **style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, filename)


def _plot_sac_training_reward(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [SACRolloutData.arr_reward_tot],
        ["Reward"],
        xlabel="Time [days]",
        ylabel="Reward",
        title="SAC Training Reward over Time",
        path_output=path_output,
        filename="SAC_Training_Reward.png",
    )


def _plot_sac_training_reward_components(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [
            SACRolloutData.arr_reward,
            SACRolloutData.arr_reward_mass,
            SACRolloutData.arr_reward_distance,
        ],
        [
            "Reward",
            "Reward Mass Component",
            "Reward Distance Component",
        ],
        xlabel="Time [days]",
        ylabel="Reward per Step",
        title="SAC Training Reward Per Step over Time",
        path_output=path_output,
        filename="SAC_Training_Reward_Per_Step.png",
    )


def _plot_sac_training_throttle(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [SACRolloutData.arr_throttle],
        ["Throttle"],
        xlabel="Time [days]",
        ylabel="Throttle",
        title="SAC Training Throttle over Time",
        path_output=path_output,
        filename="SAC_Training_Throttle.png",
    )


def _plot_sac_training_attitude(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [SACRolloutData.arr_alpha_x, SACRolloutData.arr_alpha_y],
        ["alpha_x", "alpha_y"],
        xlabel="Time [days]",
        ylabel="Attitude",
        title="SAC Training Burn Attitude over Time",
        path_output=path_output,
        filename="SAC_Training_Alpha.png",
    )


def _plot_sac_training_nd_state(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [
            SACRolloutData.arr_x,
            SACRolloutData.arr_y,
            SACRolloutData.arr_vx,
            SACRolloutData.arr_vy,
        ],
        ["x", "y", "vx", "vy"],
        xlabel="Time [days]",
        ylabel="ND state",
        title="SAC Training ND State over Time",
        path_output=path_output,
        filename="SAC_ND_State.png",
    )


def _plot_sac_training_orbital_elements(SACRolloutData, path_output):
    _plot_time_series(
        SACRolloutData.arr_time,
        [SACRolloutData.arr_sma, SACRolloutData.arr_sma_target],
        ["sma", "sma_target"],
        xlabel="Time [days]",
        ylabel="SMA Achieved [m]",
        title="SAC Achieved SMA over Time",
        path_output=path_output,
        filename="SAC_SMA_Achieved.png",
    )

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
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_ECC_Achieved.png")


def _plot_sac_training_reward_per_episode(
    arr_episode_numbers, arr_episode_rs, path_output
):
    _plot_time_series(
        arr_episode_numbers,
        [arr_episode_rs],
        ["Training Reward per Episode"],
        xlabel="Episode Number",
        ylabel="Reward",
        title="SAC Reward Per Episode During Training",
        path_output=path_output,
        filename="SAC_Training_reward_per_episode.png",
    )


def _plot_sac_training_trajectory(eph, eph_h, path_output):
    fig_orb = eph.plot_xy()
    if eph_h is not None:
        fig_orb = eph.overlay_ref_orbit(
            ephem=eph_h, label="Hamiltonian Trajectory", color_in="#f89540"
        )
    fig_orb = eph.plot_xy_ref_orbit(Constants.SMA_MARS, "Mars", "#b7410e")
    fig_orb = eph.plot_xy_ref_orbit(Constants.SMA_EARTH, "Earth")
    fig_orb.savefig(os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300)


def plot_SAC_training(
    SACRolloutData,
    arr_episode_numbers,
    arr_episode_rs,
    path_output,
    eph,
    eph_h=None,
):
    """High-level plotting wrapper for SAC training in Cartesian TBR.

    This composes a series of small helpers to generate all figures
    associated with SAC training diagnostics.
    """

    _plot_sac_training_reward(SACRolloutData, path_output)
    _plot_sac_training_reward_components(SACRolloutData, path_output)
    _plot_sac_training_throttle(SACRolloutData, path_output)
    _plot_sac_training_attitude(SACRolloutData, path_output)
    _plot_sac_training_nd_state(SACRolloutData, path_output)
    _plot_sac_training_orbital_elements(SACRolloutData, path_output)
    _plot_sac_training_reward_per_episode(
        arr_episode_numbers, arr_episode_rs, path_output
    )
    _plot_sac_training_trajectory(eph, eph_h, path_output)


def _calc_rewards_from_h_ephem_if_available(ephem_H, env, params):
    if ephem_H is None:
        return None

    results = calc_rewards_from_H_ephem(ephem_H, env, params)
    (
        arr_elapsed_time,
        arr_rewards,
        arr_pos_r_components,
        arr_vel_r_components,
        arr_mass_r_components,
        arr_throttle_r_components,
        arr_time_r_components,
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
    ) = results

    return {
        "arr_elapsed_time": arr_elapsed_time,
        "arr_rewards": arr_rewards,
        "arr_pos_r_components": arr_pos_r_components,
        "arr_vel_r_components": arr_vel_r_components,
        "arr_mass_r_components": arr_mass_r_components,
        "arr_throttle_r_components": arr_throttle_r_components,
        "arr_r_tot": arr_r_tot,
    }


def _plot_tbr_polar_reward_over_time(
    SACRolloutData_TBR_polar, path_output, h_data=None
):
    time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25

    plt.figure()
    plt.plot(time_years, SACRolloutData_TBR_polar.arr_reward_tot, label="Reward")
    if h_data is not None:
        plt.plot(
            np.array(h_data["arr_elapsed_time"]) / 365.25,
            h_data["arr_r_tot"],
            label="Hamiltonian Ephem Reward",
            linestyle="--",
            color="red",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_Training_Reward.png")


def _plot_tbr_polar_time_to_target(SACRolloutData_TBR_polar, path_output, params):
    time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25
    arr_ttg_days = [
        ttg * params["t_star"] / Constants.DAYS_TO_SEC
        for ttg in SACRolloutData_TBR_polar.arr_ttg
    ]
    arr_zeros = [0.0 for _ in SACRolloutData_TBR_polar.arr_ttg]

    plt.figure()
    plt.plot(
        time_years,
        np.array(arr_ttg_days) / 365.25,
        label="Time to Target",
        color="magenta",
    )
    plt.plot(
        time_years,
        arr_zeros,
        label="Target Reached",
        linestyle="--",
        color="orange",
    )
    plt.xlabel("Time [years]")
    plt.ylabel("Time to Target [years]")
    plt.title("SAC Training Time to Target over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_Training_Time_to_Target.png")


def _plot_tbr_polar_reward_components(
    SACRolloutData_TBR_polar, path_output, h_data=None
):
    time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25

    try:
        # Reward per step
        plt.figure()
        plt.plot(time_years, SACRolloutData_TBR_polar.arr_reward, label="Reward")
        if h_data is not None:
            plt.plot(
                np.array(h_data["arr_elapsed_time"]) / 365.25,
                h_data["arr_rewards"],
                label="Hamiltonian Ephem Reward",
                linestyle="--",
                color="red",
            )
        plt.xlabel("Time [years]")
        plt.ylabel("Reward per Step")
        plt.title("SAC Training Reward Per Step over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_figure(path_output, "SAC_Training_Reward_Per_Step.png")

        # Reward contribution components
        plt.figure()
        plt.plot(
            time_years,
            SACRolloutData_TBR_polar.arr_pos_r_component,
            label="Position r component",
        )
        plt.plot(
            time_years,
            SACRolloutData_TBR_polar.arr_vel_r_component,
            label="Velocity r component",
        )
        plt.plot(
            time_years,
            SACRolloutData_TBR_polar.arr_throttle_r_component,
            label="Throttle r component",
        )
        plt.plot(
            time_years,
            SACRolloutData_TBR_polar.arr_reward,
            label="Composite Reward",
        )
        if h_data is not None:
            h_time_years = np.array(h_data["arr_elapsed_time"]) / 365.25
            plt.plot(
                h_time_years,
                h_data["arr_pos_r_components"],
                label="Hamiltonian Ephem Position r component",
                linestyle="--",
                color="red",
            )
            plt.plot(
                h_time_years,
                h_data["arr_vel_r_components"],
                label="Hamiltonian Ephem Velocity r component",
                linestyle="--",
                color="blue",
            )
            plt.plot(
                h_time_years,
                h_data["arr_throttle_r_components"],
                label="Hamiltonian Ephem Throttle r component",
                linestyle="--",
                color="green",
            )
            plt.plot(
                h_time_years,
                h_data["arr_rewards"],
                label="Hamiltonian Ephem Composite Reward",
                linestyle="--",
                color="orange",
            )
        plt.xlabel("Time [years]")
        plt.ylabel("Reward Contribution")
        plt.title("SAC Training Reward Contribution over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_figure(path_output, "SAC_Training_Reward_Contribution.png")
    except Exception as e:
        print(f"Error saving SAC Training Reward plots: {e}")


def _plot_tbr_polar_throttle(
    SACRolloutData_TBR_polar, path_output, ephem_H=None, h_data=None
):
    time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25

    plt.figure()
    plt.plot(time_years, SACRolloutData_TBR_polar.arr_throttle, label="Throttle")
    if ephem_H is not None and h_data is not None:
        plt.plot(
            np.array(h_data["arr_elapsed_time"]) / 365.25,
            ephem_H.arr_u,
            label="Hamiltonian Ephem Throttle",
            linestyle="--",
            color="red",
        )
    plt.xlabel("Time [years]")
    plt.ylabel("Throttle")
    plt.title("SAC Training Throttle over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_Training_Throttle.png")


def _plot_tbr_polar_attitude(
    SACRolloutData_TBR_polar, path_output, ephem_H=None, h_data=None
):
    arr_time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25

    plt.figure()
    plt.plot(arr_time_years, SACRolloutData_TBR_polar.arr_alpha_r, label="SAC alpha_r")
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_alpha_theta,
        label="SAC alpha_theta",
    )

    if ephem_H is not None and h_data is not None:
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
            np.array(h_data["arr_elapsed_time"]) / 365.25,
            arr_alpha_fpa_cos_H,
            label="Hamiltonian Ephem alpha_fpa_cos",
            linestyle="--",
            color="red",
        )
        plt.plot(
            np.array(h_data["arr_elapsed_time"]) / 365.25,
            arr_alpha_fpa_sin_H,
            label="Hamiltonian Ephem alpha_fpa_sin",
            linestyle="--",
            color="blue",
        )

    plt.xlabel("Time [years]")
    plt.ylabel("Attitude")
    plt.title("SAC Training Burn Attitude over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_Training_Alpha.png")


def _plot_tbr_polar_nd_state(SACRolloutData_TBR_polar, path_output):
    arr_time_years = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25

    # Relative state
    plt.figure()
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
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_ND_Rel_State.png")

    # Absolute state
    plt.figure()
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_rad,
        label="r_target",
        linestyle="--",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_cos_theta,
        label="theta_cos_target",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_sin_theta,
        label="theta_sin_target",
    )
    plt.plot(
        arr_time_years,
        SACRolloutData_TBR_polar.arr_v,
        label="v_target",
        linestyle="--",
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
    plt.grid(True, alpha=0.3)
    _save_figure(path_output, "SAC_ND_State.png")


def _plot_tbr_polar_reward_per_episode(
    arr_episode_numbers, arr_episode_rs, path_output
):
    if arr_episode_numbers is None or arr_episode_rs is None:
        return

    _plot_time_series(
        arr_episode_numbers,
        [arr_episode_rs],
        ["Training Reward per Episode"],
        xlabel="Episode Number",
        ylabel="Reward",
        title="SAC Reward Per Episode During Training",
        path_output=path_output,
        filename="SAC_Training_reward_per_episode.png",
    )


def _plot_pretraining_losses(
    arr_actor_loss_pt, arr_critic_loss_pt, path_output, params
):
    if (
        (arr_critic_loss_pt is None)
        or (arr_actor_loss_pt is None)
        or (len(arr_critic_loss_pt) == 0 or len(arr_actor_loss_pt) == 0)
    ):
        print("No pre-training loss data to plot.")
        return

    print("Plotting pre-training actor and critic losses.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

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


def _plot_tbr_polar_rendezvous_trajectory(eph, env, params, path_output, ephem_H=None):
    fig_orb = plot_rendezvous_traj(eph, env, params)
    if params.get("flag_gen_H_traj", False) and (ephem_H is not None):
        fig_orb = eph.overlay_ref_orbit(
            ephem=ephem_H, label="Hamiltonian Trajectory", color_in="#f89540"
        )
    fig_orb.savefig(
        os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig_orb)


def _write_tbr_polar_rewards_csv(SACRolloutData_TBR_polar, path_output, h_data=None):
    sac_time = np.array(SACRolloutData_TBR_polar.arr_time) / 365.25
    sac_reward = SACRolloutData_TBR_polar.arr_reward

    sac_time = sac_time[::10]
    sac_reward = sac_reward[::10]

    if h_data is not None:
        h_time = np.array(h_data["arr_elapsed_time"]) / 365.25
        h_reward = h_data["arr_rewards"]
        h_time = h_time[::10]
        h_reward = h_reward[::10]
        arr_r_pos_H = h_data["arr_pos_r_components"][::10]
        arr_r_vel_H = h_data["arr_vel_r_components"][::10]
        arr_r_mass_H = h_data["arr_throttle_r_components"][::10]

        max_len = max(len(sac_time), len(h_time))
    else:
        max_len = len(sac_time)

    if h_data is None:
        df = pd.DataFrame(
            {
                "Time_years": list(sac_time) + [np.nan] * (max_len - len(sac_time)),
                "SAC_Reward_per_Step": list(sac_reward)
                + [np.nan] * (max_len - len(sac_reward)),
            }
        )
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
    """High-level plotting wrapper for SAC training in TBR polar coordinates.

    This mirrors :func:`plot_SAC_training` but for the polar rendezvous
    environment, decomposing the original monolithic routine into smaller,
    testable helpers while preserving behavior and outputs.
    """

    h_data = _calc_rewards_from_h_ephem_if_available(ephem_H, env, params)

    _plot_tbr_polar_reward_over_time(SACRolloutData_TBR_polar, path_output, h_data)
    _plot_tbr_polar_time_to_target(SACRolloutData_TBR_polar, path_output, params)
    _plot_tbr_polar_reward_components(SACRolloutData_TBR_polar, path_output, h_data)
    _plot_tbr_polar_throttle(
        SACRolloutData_TBR_polar, path_output, ephem_H=ephem_H, h_data=h_data
    )
    _plot_tbr_polar_attitude(
        SACRolloutData_TBR_polar, path_output, ephem_H=ephem_H, h_data=h_data
    )
    _plot_tbr_polar_nd_state(SACRolloutData_TBR_polar, path_output)
    _plot_tbr_polar_reward_per_episode(arr_episode_numbers, arr_episode_rs, path_output)
    _plot_pretraining_losses(arr_actor_loss_pt, arr_critic_loss_pt, path_output, params)
    _plot_tbr_polar_rendezvous_trajectory(eph, env, params, path_output, ephem_H)
    _write_tbr_polar_rewards_csv(SACRolloutData_TBR_polar, path_output, h_data)


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

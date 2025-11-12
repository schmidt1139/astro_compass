import matplotlib
import matplotlib.pyplot as plt
import os
from constants.constants import Constants
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env


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
        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)


def plot_SAC_training(
    SACRolloutData, arr_episode_numbers, arr_episode_rs, path_output, eph
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
    eph.plot_xy_ref_orbit(Constants.SMA_MARS, "Mars", "#b7410e")
    eph.plot_xy_ref_orbit(Constants.SMA_EARTH, "Earth")
    fig_orb.savefig(os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300)

def plot_SAC_training_TBR(
    SACRolloutData_TBR, arr_episode_numbers, arr_episode_rs, path_output, eph,
    params, env
):

    # plot reward over time
    plt.figure()
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_reward_tot, label="Reward")
    plt.xlabel("Time [days]")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward.png"), dpi=300)

    plt.figure()
    arr_ttg_days = [ttg * params["t_star"] / Constants.DAYS_TO_SEC for ttg in SACRolloutData_TBR.arr_ttg]
    arr_zeros = [0.0 for ttg in SACRolloutData_TBR.arr_ttg]
    plt.plot(SACRolloutData_TBR.arr_time, arr_ttg_days, label="Time to Target", color="magenta")
    plt.plot(SACRolloutData_TBR.arr_time, arr_zeros, label="Target Reached", linestyle="--", color="orange")
    plt.xlabel("Time [days]")
    plt.ylabel("Time to Target [days]")
    plt.title("SAC Training Time to Target over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Time_to_Target.png"), dpi=300)

    # plot reward over time per step
    plt.figure()
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_reward, label="Reward")
    plt.xlabel("Time [days]")
    plt.ylabel("Reward per Step")
    plt.title("SAC Training Reward Per Step over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300)

    # plot throttle over time
    plt.figure()
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_throttle, label="Throttle")
    plt.xlabel("Time [days]")
    plt.ylabel("Throttle")
    plt.title("SAC Training Throttle over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Throttle.png"), dpi=300)

    # plot attitude over time
    plt.figure()
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_alpha_x, label="alpha_x")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_alpha_y, label="alpha_y")
    plt.xlabel("Time [days]")
    plt.ylabel("Attitude")
    plt.title("SAC Training Burn Attitude over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)  # Force grid on with some transparency
    plt.savefig(os.path.join(path_output, "SAC_Training_Alpha.png"), dpi=300)

    # plot nd state over time
    plt.figure()
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_x, label="x", color="cyan")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_y, label="y", color="magenta")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_vx, label="vx", color="orange")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_vy, label="vy", color="pink")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_x_target, label="x_target", linestyle="--", color="cyan")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_y_target, label="y_target", linestyle="--", color="magenta")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_vx_target, label="vx_target", linestyle="--", color="orange")
    plt.plot(SACRolloutData_TBR.arr_time, SACRolloutData_TBR.arr_vy_target, label="vy_target", linestyle="--", color="pink")
    plt.xlabel("Time [days]")
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

    # generate and save figures
    fig_orb = eph.plot_xy()
    x_target = SACRolloutData_TBR.arr_x_target[-1]*params["l_star"]
    y_target = SACRolloutData_TBR.arr_y_target[-1]*params["l_star"]
    vx_target = SACRolloutData_TBR.arr_vx_target[-1]*params["l_star"]/params["t_star"]
    vy_target = SACRolloutData_TBR.arr_vy_target[-1]*params["l_star"]/params["t_star"]
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
        color_in="lime"
    )

    fig_orb.savefig(os.path.join(path_output, "SAC_Test_Traj.png"), dpi=300)

def plot_overlay_ballistic_orbit(x, y, vx, vy, env, fig, params, eph, label_in,
                                 color_in="lime"):

    obs, info = env.reset()

    state_in = [x, y, vx, vy, 1000.0, x, y, vx, vy, Constants.YEARS_TO_SEC * 10.0]

    obs, info = env.set_state(state_in)

    T = info["orbital_period_years"] * Constants.YEARS_TO_SEC

    time = 0.0
    flag_done = False
    arr_x = []
    arr_y = []
    while not flag_done:

        obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.0])

        # dim state
        t_i = info["Elapsed time"]
        x_i = obs[0] * params["l_star"]
        y_i = obs[1] * params["l_star"]
        vx_i = obs[2] * params["l_star"] / params["t_star"]
        vy_i = obs[3] * params["l_star"] / params["t_star"]
        m_i = obs[4] * params["m_star"]

        if info["Elapsed time"] >= T:
            flag_done = True

        arr_x.append(x_i)
        arr_y.append(y_i)

    fig = eph.overlay_ref_orbit(ephem=None, label=label_in, color_in=color_in, arr_x=arr_x, arr_y=arr_y)

    return fig
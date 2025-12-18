import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from astro_compass.utils.buffer_utils import extract_rollouts, plot_episodes
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.path_utils import CONFIG_ROOT, DATA_ROOT, PLOT_ROOT

print("GPU available: ", torch.cuda.is_available())


def format_episode(episodes):
    list_arr_x = []
    list_arr_y = []
    list_arr_x_target = []
    list_arr_y_target = []
    list_arr_action_u = []
    list_arr_action_ar = []
    list_arr_action_at = []
    list_arr_rewards = []

    for episode in episodes:
        # extract obs, action, reward, and done
        obs = episode["obs"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        next_obs = episode["next_obs"]

        arr_x = []
        arr_y = []
        arr_x_target = []
        arr_y_target = []
        arr_rewards = []
        r_tot = 0.0

        for i in range(len(obs)):
            # x, and y
            et = i * params["env_step_size"]
            x = obs[i][6]
            y = obs[i][7]
            x_target = next_obs[i][10]
            y_target = next_obs[i][11]
            r_tot += rewards[i]

            arr_x.append(x)
            arr_y.append(y)
            arr_x_target.append(x_target)
            arr_y_target.append(y_target)
            arr_rewards.append(rewards[i])

        # store action components
        arr_action_u_raw = []
        arr_action_ar = []
        arr_action_at = []

        for i in range(len(actions)):
            arr_action_u_raw.append(actions[i][0])

            a_unit = (actions[i][1] ** 2 + actions[i][2] ** 2) ** 0.5
            a_r = actions[i][1] / a_unit if a_unit != 0 else 0
            a_t = actions[i][2] / a_unit if a_unit != 0 else 0

            arr_action_ar.append(a_r)
            arr_action_at.append(a_t)

        # scale the throttle actions from [-1, 1] to [0, 1]
        rescaled_actions = rescale_actions(
            np.array(arr_action_u_raw), np.array([0.0]), np.array([1.0])
        )
        arr_action_u = rescaled_actions.flatten().tolist()

        list_arr_x.append(arr_x)
        list_arr_y.append(arr_y)
        list_arr_x_target.append(arr_x_target)
        list_arr_y_target.append(arr_y_target)
        list_arr_action_u.append(arr_action_u)
        list_arr_action_ar.append(arr_action_ar)
        list_arr_action_at.append(arr_action_at)
        list_arr_rewards.append(arr_rewards)


def plot_episodes(episodes, params, path_plots):
    # plot x and y trajectories
    # plot velocity residuals over time for all episodes
    plt.figure()
    for i in range(len(list_arr_x)):
        plt.plot(list_arr_x[i], list_arr_y[i], color="blue")
        plt.plot(list_arr_x_target[i], list_arr_y_target[i], color="red")

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory vs Target Trajectory")
    plt.grid(True)
    plt.savefig(f"{path_plots}/trajectory_vs_target.png", dpi=300)
    plt.close()

    # plot throttle
    plt.figure()
    for i in range(len(list_arr_action_u)):
        plt.plot(list_arr_action_u[i], color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Throttle")
    plt.title("Throttle over Time")
    plt.grid(True)
    plt.savefig(f"{path_plots}/throttle_over_time.png", dpi=300)
    plt.close()

    # plot attitude
    plt.figure()
    for i in range(len(list_arr_action_ar)):
        plt.plot(list_arr_action_ar[i], color="blue")
        plt.plot(list_arr_action_at[i], color="orange")
    plt.xlabel("Time Step")
    plt.ylabel("Attitude")
    plt.title("Attitude over Time")
    plt.grid(True)
    plt.savefig(f"{path_plots}/attitude_over_time.png", dpi=300)
    plt.close()

    # plot rewards
    plt.figure()
    for i in range(len(list_arr_rewards)):
        plt.plot(list_arr_rewards[i], color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.savefig(f"{path_plots}/rewards_over_time.png", dpi=300)
    plt.close()


def rescale_actions(actions, low, high):
    """Rescale actions from [-1, 1] to [low, high]"""
    return low + (actions + 1.0) / 2.0 * (high - low)

    # When extracting from buffer:
    low = np.array([0.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    rescaled_actions = rescale_actions(buffer_actions, low, high)


def main(replay_dir, params):
    replay_path = os.path.join(replay_dir, "replay_buffer.pkl")
    with open(replay_path, "rb") as file:
        buffer = pickle.load(file)

    episodes = extract_rollouts(buffer, params)
    plot_episodes(episodes, params, PLOT_ROOT)


if __name__ == "__main__":
    config_toml = "plot_buffer.toml"
    path_config = os.path.join(CONFIG_ROOT, config_toml)
    params = read_toml_config_file(path_config)

    params["config_toml"] = config_toml
    replay_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "replay_buffers")
    main(replay_dir, params)

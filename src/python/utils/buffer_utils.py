import numpy as np
import matplotlib.pyplot as plt

def extract_episodes_from_buffer(replay_buffer, params):
    size = replay_buffer.size()
    obs = replay_buffer.observations
    actions = replay_buffer.actions
    rewards = replay_buffer.rewards
    next_obs = replay_buffer.next_observations
    dones = replay_buffer.dones

    episodes = []
    episode = {"obs": [], "actions": [], "rewards": [], "next_obs": []}

    for i in range(size):
        episode["obs"].append(obs[i][0])
        episode["actions"].append(actions[i][0])
        episode["rewards"].append(rewards[i][0])
        episode["next_obs"].append(next_obs[i][0])
        if np.asarray(dones[i][0]).flatten()[0]:
            episodes.append(episode)
            episode = {"obs": [], "actions": [], "rewards": [], "next_obs": []}

    # Add last episode if not ended with done
    if episode["obs"]:
        episodes.append(episode)

    # Trim down episode list based on input
    plot_n_most_recent_eps = params.get("plot_n_most_recent_eps", 1)

    episodes = episodes[-plot_n_most_recent_eps:]

    return episodes


def plot_episodes(episodes, params, path_plots):

    list_arr_x = []
    list_arr_y = []
    list_arr_x_target = []
    list_arr_y_target = []
    list_arr_action_u = []
    list_arr_action_ar = []
    list_arr_action_at = []

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

        for i in range(len(obs)):
            #x, and y
            et = i * params["env_step_size"]
            x = obs[i][6]
            y = obs[i][7]
            x_target = next_obs[i][10]
            y_target = next_obs[i][11]
            arr_x.append(x)
            arr_y.append(y)
            arr_x_target.append(x_target)
            arr_y_target.append(y_target)

        #store action components
        arr_action_u_raw = []
        arr_action_ar = []
        arr_action_at = []

        for i in range(len(actions)):
            arr_action_u_raw.append(actions[i][0])

            a_unit = (actions[i][1]**2 + actions[i][2]**2)**0.5
            a_r = actions[i][1] / a_unit if a_unit != 0 else 0
            a_t = actions[i][2] / a_unit if a_unit != 0 else 0

            arr_action_ar.append(a_r)
            arr_action_at.append(a_t)

        #scale the throttle actions from [-1, 1] to [0, 1]
        rescaled_actions = rescale_actions(np.array(arr_action_u_raw), np.array([0.0]), np.array([1.0]))
        arr_action_u = rescaled_actions.flatten().tolist()

        list_arr_x.append(arr_x)
        list_arr_y.append(arr_y)
        list_arr_x_target.append(arr_x_target)
        list_arr_y_target.append(arr_y_target)
        list_arr_action_u.append(arr_action_u)
        list_arr_action_ar.append(arr_action_ar)
        list_arr_action_at.append(arr_action_at)

        

    #plot x and y trajectories
    #plot velocity residuals over time for all episodes
    plt.figure()
    for i in range(len(list_arr_x)):
        plt.plot(list_arr_x[i], list_arr_y[i], color='blue')
        plt.plot(list_arr_x_target[i], list_arr_y_target[i], color='red')

    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory vs Target Trajectory')
    plt.grid(True)
    plt.savefig(f"{path_plots}/trajectory_vs_target.png", dpi=300)
    plt.close()

    #plot throttle
    plt.figure()
    for i in range(len(list_arr_action_u)):
        plt.plot(list_arr_action_u[i], color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Throttle')
    plt.title('Throttle over Time')
    plt.grid(True)
    plt.savefig(f"{path_plots}/throttle_over_time.png", dpi=300)
    plt.close()

    #plot attitude
    plt.figure()
    for i in range(len(list_arr_action_ar)):
        plt.plot(list_arr_action_ar[i], color='blue')
        plt.plot(list_arr_action_at[i], color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Attitude')
    plt.title('Attitude over Time')
    plt.grid(True)
    plt.savefig(f"{path_plots}/attitude_over_time.png", dpi=300)
    plt.close()

def rescale_actions(actions, low, high):
    """Rescale actions from [-1, 1] to [low, high]"""
    return low + (actions + 1.0) / 2.0 * (high - low)

    # When extracting from buffer:
    low = np.array([0.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    rescaled_actions = rescale_actions(buffer_actions, low, high)
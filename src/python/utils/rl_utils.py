import torch
import torch.nn as nn
import numpy as np

from utils.log_utils import log
from core.training_data_generation import read_ephems_from_dir
from stable_baselines3.common.callbacks import BaseCallback


def log_training_perf(
    test_log, callback, eval_callback, model, training_steps, flag_verbose
):

    # --- add final training performance metrics to the log ---
    total_episodes = len(callback.episode_rewards)
    total_timesteps_done = getattr(model, "num_timesteps", training_steps)

    if total_episodes > 0:
        avg_reward = sum(callback.episode_rewards) / total_episodes
        best_reward = max(callback.episode_rewards)
        last_reward = callback.episode_rewards[-1]
    else:
        avg_reward = float("nan")
        best_reward = float("nan")
        last_reward = float("nan")

    if len(callback.episode_lengths) > 0:
        avg_length = sum(callback.episode_lengths) / len(callback.episode_lengths)
    else:
        avg_length = 0

    best_eval = getattr(eval_callback, "best_mean_reward", None)

    test_log = log("FINAL TRAINING METRICS", test_log, flag_verbose)
    test_log = log("Total episodes: " + str(total_episodes), test_log, flag_verbose)
    test_log = log(
        "Total timesteps (model.num_timesteps): " + str(total_timesteps_done),
        test_log,
        flag_verbose,
    )
    test_log = log("Average episode reward: " + str(avg_reward), test_log, flag_verbose)
    test_log = log("Best episode reward: " + str(best_reward), test_log, flag_verbose)
    test_log = log("Last episode reward: " + str(last_reward), test_log, flag_verbose)
    test_log = log("Average episode length: " + str(avg_length), test_log, flag_verbose)

    if best_eval is not None:
        test_log = log(
            "Best eval mean reward: " + str(best_eval), test_log, flag_verbose
        )

    return test_log


def copy_linear_layers(src_sequential: nn.Sequential, dst_sequential: nn.Sequential):

    # Extract linear layers from both source and destination sequential models
    src_linears = [m for m in src_sequential.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst_sequential.modules() if isinstance(m, nn.Linear)]

    # Check if the networks sizes are the same
    assert len(src_linears) == len(
        dst_linears
    ), f"Linear layer count mismatch: {len(src_linears)} vs {len(dst_linears)}"

    with torch.no_grad():
        # Step through both sets of linear layers and copy weights and biases
        for s, d in zip(src_linears, dst_linears):

            # ensure the shapes match before copying
            assert (
                s.weight.shape == d.weight.shape
            ), f"Weight shape mismatch {s.weight.shape} vs {d.weight.shape}"

            # Copy weights and biases
            d.weight.copy_(s.weight)

            if s.bias is not None and d.bias is not None:
                assert s.bias.shape == d.bias.shape
                d.bias.copy_(s.bias)


def load_pretrained_nn_into_SAC(supervised_controller, sac_model):

    src_layers = nn.Sequential(
        supervised_controller.fc1,
        supervised_controller.fc2,
        supervised_controller.fc3,
        supervised_controller.fc4,
        supervised_controller.fc5,
    )

    dst_layers = sac_model.policy.actor.latent_pi

    print("Loading pre-trained controller")

    # copy the weights and biases from the supervised controller to the SAC model
    copy_linear_layers(src_layers, dst_layers)

    with torch.no_grad():
        # SB3's actor.mu is a Linear(act_latent_dim -> act_dim)
        assert (
            sac_model.policy.actor.mu.weight.shape
            == supervised_controller.fc6.weight.shape
        )
        sac_model.policy.actor.mu.weight.copy_(supervised_controller.fc6.weight)
        sac_model.policy.actor.mu.bias.copy_(supervised_controller.fc6.bias)

    # Initialize log_std to something small (more deterministic at start)
    with torch.no_grad():
        if hasattr(sac_model.policy.actor, "log_std") and isinstance(
            sac_model.policy.actor.log_std, nn.Parameter
        ):
            sac_model.policy.actor.log_std.fill_(-4.0)  # e.g., std ≈ exp(-4) ~ 0.018


def import_training_into_replay_buffer(
    path_training_data, test_log, model, env, params
):

    test_log = log("Importing training data into replay buffer", test_log, True)

    # read ephemerides from directory
    num_ephems_to_use = params["num_ephems_to_use"]
    test_log = log(
        "Using only first " + str(num_ephems_to_use) + " ephemerides", test_log, True
    )
    set_ephems = read_ephems_from_dir(path_training_data, num_ephems_to_use)
    test_log = log("Reading ephemerides from: " + path_training_data, test_log, True)

    # count number of ephemerides
    num_ephems = len(set_ephems)
    test_log = log(
        "Number of ephemerides loaded into replay buffer: " + str(num_ephems),
        test_log,
        True,
    )
    set_ephems = set_ephems[:num_ephems_to_use]
    num_ephems = len(set_ephems)
    num_states = set_ephems[0].num_vectors * num_ephems
    test_log = log("Number of total state vectors: " + str(num_states), test_log, True)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # collect all states and actions from ephemerides
    for eph in set_ephems:
        for i in range(eph.num_vectors):

            # Extract the vector and separate into state and action components
            state_vec = eph.get_vector_at_index(i)
            x = state_vec[1]
            y = state_vec[2]
            vx = state_vec[3]
            vy = state_vec[4]
            m = state_vec[5]
            alpha_x = state_vec[6]
            alpha_y = state_vec[7]
            u = state_vec[8]

            # non-dim the obs
            x_nd = x / params["l_star"]
            y_nd = y / params["l_star"]
            vx_nd = vx / (params["l_star"] / params["t_star"])
            vy_nd = vy / (params["l_star"] / params["t_star"])
            mass_nd = m / params["m_star"]

            # append to arrays
            env_state = np.array([x, y, vx, vy, m])
            env_state_nd = np.array([x_nd, y_nd, vx_nd, vy_nd, mass_nd])
            env_action = np.array([u, alpha_x, alpha_y])

            # reset env and set state to state vector
            obs, info = env.reset()

            # Unwrap the environment to access the underlying custom environment
            unwrapped_env = env.unwrapped

            # Set the environment state
            unwrapped_env.set_state(env_state)

            # step env with action
            obs, reward, done, truncated, info = unwrapped_env.step(env_action)

            # Store the experience tuple
            add_experience_to_replay_buffer(
                model, env_state_nd, env_action, reward, obs, done
            )

            states.append(env_state_nd)
            actions.append(env_action)
            rewards.append(reward)
            next_states.append(obs)
            dones.append(done)


def add_experience_to_replay_buffer(model, obs, action, reward, next_obs, done):
    """
    Manually add a single experience to the SAC replay buffer

    Args:
        model: SAC model instance
        obs: current observation (numpy array)
        action: action taken (numpy array)
        reward: reward received (float)
        next_obs: next observation (numpy array)
        done: episode termination flag (bool)
    """
    # Ensure arrays are the right shape and type
    obs = np.array(obs, dtype=np.float32).reshape(1, -1)
    action = np.array(action, dtype=np.float32).reshape(1, -1)
    next_obs = np.array(next_obs, dtype=np.float32).reshape(1, -1)
    reward = np.array([reward], dtype=np.float32)
    done = np.array([done], dtype=np.float32)

    # Add to replay buffer
    model.replay_buffer.add(
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=reward,
        done=done,
        infos=[{}],  # Empty info dict
    )


class RewardLoggerCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self._last_ep_buffer_len = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_infos = (
            list(self.model.ep_info_buffer)
            if self.model.ep_info_buffer is not None
            else []
        )  # current buffer snapshot
        # handle deque wrap-around: if buffer shrank, reset last index
        if len(ep_infos) < self._last_ep_buffer_len:
            self._last_ep_buffer_len = 0
        # only process newly added entries
        new_infos = ep_infos[self._last_ep_buffer_len :]
        for ep_info in new_infos:
            # ep_info keys: "r" for reward, "l" for length
            self.episode_rewards.append(ep_info["r"])
            self.episode_lengths.append(ep_info["l"])
        self._last_ep_buffer_len = len(ep_infos)

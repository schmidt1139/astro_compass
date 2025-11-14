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
    test_log = log("Reading ephemerides", test_log, True)

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


def pre_train(test_log, model, params, env):
    
    test_log = log("Pre-training networks...", test_log, True)
    test_log = log("Replay buffer type: " + str(type(model.replay_buffer)), test_log, True)

    # Check if replay buffer is initialized
    if model.replay_buffer is not None:
        test_log = log(
            "Experience buffer size: " + str(model.replay_buffer.size()), test_log, True
        )  # current number of transitions
        test_log = log(
            "Experience buffer capacity: " + str(model.replay_buffer.buffer_size), test_log, True
        )  # capacity
    else:
        test_log = log("Replay buffer is not initialized yet.", test_log, True)

    #check what ephem version to assume
    if "ephem_version" in params:
        ephem_version = params["ephem_version"]
    else:
        ephem_version = 1.0  # default to version 1 if not specified

    test_log = log(
        f"Assuming ephemeris version: {ephem_version}", test_log, True
    )

    # check env state dimensions
    test_log = log(
        f"Environment observation space shape: {env.observation_space.shape}",
        test_log,
        True,
    )

    if (ephem_version == 2.0 and env.observation_space.shape[0] == 10 ):
        test_log = log(
            "Environment observation space matches expected shape for ephemeris version 2.0",
            test_log,
            True,
        )
    else:
        raise ValueError(
            f"Environment observation space shape {env.observation_space.shape} does not match expected shape for ephemeris version {ephem_version}"
        )

    # Import training data into replay buffer
    import_training_into_replay_buffer_v2(
        params["path_training_data"],  # path to directory containing training ephemerides
        test_log,  # log
        model,  # SAC model
        env,
        params,
    )

    test_log = log(
            "Updated experience buffer size: " + str(model.replay_buffer.size()), test_log, True
        )  # current number of transitions
    
    # Train networks on replay buffer
    test_log, critic_loss_reduced, actor_loss_reduced = train_on_replay_buffer(model, params, test_log)

    return test_log, critic_loss_reduced, actor_loss_reduced


def import_training_into_replay_buffer_v2(
    path_training_data, test_log, model, env, params
):
    """
    Import training data from ephemeris v2.0 files into the replay buffer.
    Handles 10-dimensional observation space (x, y, vx, vy, m, x_t, y_t, vx_t, vy_t, ttg).
    """
    from tqdm import tqdm
    
    test_log = log("Importing training data (v2.0) into replay buffer", test_log, True)

    # read ephemerides from directory
    num_ephems_to_use = params.get("num_ephems_to_use", None)
    ephem_version = params.get("ephem_version", 2.0)
    step_size = params.get("ephem_step_size", 1)  # Sample every Nth vector
    
    test_log = log(f"Reading ephemerides (version {ephem_version})", test_log, True)
    set_ephems = read_ephems_from_dir(path_training_data, num_ephems_to_use, version=ephem_version)
    
    num_ephems = len(set_ephems)
    test_log = log(
        f"Number of ephemerides loaded: {num_ephems}",
        test_log,
        True,
    )

    # Count total experiences that will be added (accounting for step_size)
    total_experiences = sum(eph.num_vectors // step_size for eph in set_ephems)
    test_log = log(
        f"Total experiences to import (with step_size={step_size}): {total_experiences}",
        test_log,
        True,
    )

    # Process each ephemeris with progress bar
    for eph in tqdm(set_ephems, desc="Processing ephemerides"):
        for i in range(0, eph.num_vectors - 1, step_size):  # -1 to ensure we have next_obs
            
            # Current state vector
            state_vec = eph.get_vector_at_index(i)
            # Next state vector
            next_state_vec = eph.get_vector_at_index(i + 1)
            
            # Extract current state components (v2.0 format)
            et = state_vec[0]
            x = state_vec[1]
            y = state_vec[2]
            vx = state_vec[3]
            vy = state_vec[4]
            m = state_vec[5]
            x_target = state_vec[6]
            y_target = state_vec[7]
            vx_target = state_vec[8]
            vy_target = state_vec[9]
            ttg = state_vec[10]
            alpha_x = state_vec[11]
            alpha_y = state_vec[12]
            u = state_vec[13]
            
            # Extract next state components
            x_next = next_state_vec[1]
            y_next = next_state_vec[2]
            vx_next = next_state_vec[3]
            vy_next = next_state_vec[4]
            m_next = next_state_vec[5]
            x_target_next = next_state_vec[6]
            y_target_next = next_state_vec[7]
            vx_target_next = next_state_vec[8]
            vy_target_next = next_state_vec[9]
            ttg_next = next_state_vec[10]

            # Non-dimensionalize observations (10-dimensional)
            obs = np.array([
                x / params["l_star"],
                y / params["l_star"],
                vx / (params["l_star"] / params["t_star"]),
                vy / (params["l_star"] / params["t_star"]),
                m / params["m_star"],
                x_target / params["l_star"],
                y_target / params["l_star"],
                vx_target / (params["l_star"] / params["t_star"]),
                vy_target / (params["l_star"] / params["t_star"]),
                ttg / params["t_star"]
            ], dtype=np.float32)
            
            next_obs = np.array([
                x_next / params["l_star"],
                y_next / params["l_star"],
                vx_next / (params["l_star"] / params["t_star"]),
                vy_next / (params["l_star"] / params["t_star"]),
                m_next / params["m_star"],
                x_target_next / params["l_star"],
                y_target_next / params["l_star"],
                vx_target_next / (params["l_star"] / params["t_star"]),
                vy_target_next / (params["l_star"] / params["t_star"]),
                ttg_next / params["t_star"]
            ], dtype=np.float32)
            
            # Action
            action = np.array([u, alpha_x, alpha_y], dtype=np.float32)
            
            # Compute reward using environment's reward function
            # Set the environment state to compute reward
            unwrapped_env = env.unwrapped
            dim_state = np.array([x, y, vx, vy, m, x_target, y_target, vx_target, vy_target, ttg])
            unwrapped_env.set_state(dim_state)
            
            # calc_reward returns (reward, terminated)
            reward, terminated = unwrapped_env.calc_reward()
            
            # Done flag (true at end of trajectory OR if terminated by environment)
            done = (i + step_size >= eph.num_vectors - 1) or terminated
            
            # Add experience to replay buffer
            add_experience_to_replay_buffer(model, obs, action, reward, next_obs, done)


def train_on_replay_buffer(model, params, test_log):
    """
    Train the SAC networks using only the experiences currently in the replay buffer.
    Does not collect any new experiences from the environment.
    
    Args:
        model: SAC model instance
        params: Dictionary containing training parameters
        test_log: Logging list
        
    Returns:
        Updated test_log
    """
    from tqdm import tqdm
    import pandas as pd
    import os
    
    num_gradient_steps = params.get("pretrain_gradient_steps", 10000)
    batch_size = params.get("batch_size", 256)
    
    test_log = log(
        f"Training networks on replay buffer with {num_gradient_steps} gradient steps",
        test_log,
        True
    )
    
    # Check that we have enough samples
    buffer_size = model.replay_buffer.size()
    if buffer_size < batch_size:
        test_log = log(
            f"Warning: Replay buffer has only {buffer_size} samples, less than batch size {batch_size}",
            test_log,
            True
        )
        return test_log
    
    test_log = log(f"Replay buffer size: {buffer_size}", test_log, True)
    test_log = log(f"Batch size: {batch_size}", test_log, True)
    
    # Set up a minimal logger for pre-training if one doesn't exist
    # model.train() requires a logger to record learning rate and other metrics
    # We'll use a simple logger that doesn't interfere with TensorBoard
    if not hasattr(model, '_logger') or model._logger is None:
        from stable_baselines3.common.logger import Logger
        # Create a minimal logger that just stores values without writing anywhere
        model._logger = Logger(folder=None, output_formats=[])
    
    # Set up progress tracking (needed by some internal SB3 methods)
    model._current_progress_remaining = 1.0
    
    # Set the number of timesteps so model.train() knows where we are
    if not hasattr(model, '_total_timesteps'):
        model._total_timesteps = 0
    if not hasattr(model, 'num_timesteps'):
        model.num_timesteps = 0
    
    test_log = log(f"Training on replay buffer", test_log, True)
    
    # Track losses
    critic_losses = []
    actor_losses = []
    
    # Train the networks by sampling from replay buffer
    for step in tqdm(range(num_gradient_steps), desc="Pre-training"):
        # This performs one gradient update on actor and critic networks
        model.train(gradient_steps=1)
        
        # Try to get losses from the model's internal tracking
        # Note: SAC stores these internally but they may not be accessible immediately
        try:
            # Check if _logger attribute exists (not the property)
            if hasattr(model, '_logger') and model._logger is not None:
                # Try to get values from logger's name_to_value dict
                if hasattr(model._logger, 'name_to_value'):
                    critic_loss = model._logger.name_to_value.get("train/critic_loss", None)
                    actor_loss = model._logger.name_to_value.get("train/actor_loss", None)
                    
                    if critic_loss is not None:
                        critic_losses.append(critic_loss)
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
        except (AttributeError, KeyError):
            # Logger not set up or accessible, skip logging
            pass
        
        # Optionally log progress
        log_interval = params.get("pt_log_interval", 1000)
        if (step + 1) % log_interval == 0:
            if critic_losses and actor_losses:
                avg_critic = np.mean(critic_losses[-log_interval:])
                avg_actor = np.mean(actor_losses[-log_interval:])
                tqdm.write(f"Step {step + 1}: Critic Loss: {avg_critic:.4f}, Actor Loss: {avg_actor:.4f}")
            else:
                tqdm.write(f"Completed {step + 1}/{num_gradient_steps} gradient steps")
    
    test_log = log(
        f"Completed {num_gradient_steps} gradient steps on replay buffer",
        test_log,
        True
    )

    # Save arrays of losses to csv files
    if "output_dir_specific" in params:
        path = params["output_dir_specific"]
        
        # Save losses (downsample by 10x for file size)
        critic_loss_reduced = []
        actor_loss_reduced = []
        for i in range(len(critic_losses)):
            if i % 10 == 0:
                critic_loss_reduced.append(critic_losses[i])
                actor_loss_reduced.append(actor_losses[i])

        pd.DataFrame(critic_loss_reduced, columns=["critic_loss"]).to_csv(
            os.path.join(path, "critic_losses.csv"), index=False
        )
        pd.DataFrame(actor_loss_reduced, columns=["actor_loss"]).to_csv(
            os.path.join(path, "actor_losses.csv"), index=False
        )
        
        test_log = log(
            f"Saved loss curves to {path}",
            test_log,
            True
        )

    return test_log, critic_loss_reduced, actor_loss_reduced

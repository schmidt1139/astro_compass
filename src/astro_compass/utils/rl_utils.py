import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.core.rollouts import SACRolloutData_TBR_polar
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.core.training_data_generation import read_ephems
from astro_compass.utils.log_utils import log
from astro_compass.utils.state_vector_utils import (
    cartesian_to_polar,
    convert_attitude_from_cartesian_to_radial,
)


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


def import_training_into_replay_buffer(
    path_training_data, test_log, model, env, params
):
    test_log = log("Importing training data into replay buffer", test_log, True)

    # read ephemerides from directory
    num_ephems_to_use = params["num_ephems_to_use"]
    test_log = log(
        "Using only first " + str(num_ephems_to_use) + " ephemerides", test_log, True
    )
    set_ephems = read_ephems(path_training_data, num_ephems_to_use)
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
    for eph in tqdm(set_ephems, desc="Processing ephemerides"):
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


def add_experience_to_replay_buffer(
    model, obs, action, reward, next_obs, done, params=None
):
    """
    Add single or batched experiences to the SAC replay buffer.

    Args:
        model: SAC model instance
        obs: current observation(s), shape (obs_dim,) or (batch, obs_dim)
        action: action(s) taken, shape (action_dim,) or (batch, action_dim)
        reward: reward(s), float or array-like
        next_obs: next observation(s), shape (obs_dim,) or (batch, obs_dim)
        done: done flag(s), bool or array-like
    """
    obs = np.array(obs, dtype=np.float32)
    action = np.array(action, dtype=np.float32)
    next_obs = np.array(next_obs, dtype=np.float32)
    reward = np.array(reward, dtype=np.float32)
    done = np.array(done, dtype=np.float32)

    # If single experience, reshape to (1, dim)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)
    if reward.ndim == 0 or (reward.ndim == 1 and reward.shape == ()):
        reward = np.array([reward], dtype=np.float32)
    if reward.ndim == 1 and reward.shape[0] != obs.shape[0]:
        reward = reward.reshape(-1, 1)
    if done.ndim == 0 or (done.ndim == 1 and done.shape == ()):
        done = np.array([done], dtype=np.float32)
    if done.ndim == 1 and done.shape[0] != obs.shape[0]:
        done = done.reshape(-1, 1)

    # Info dicts: one per experience
    infos = [{} for _ in range(obs.shape[0])]

    model.replay_buffer.add(
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=reward,
        done=done,
        infos=infos,
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
    test_log = log(
        "Replay buffer type: " + str(type(model.replay_buffer)), test_log, True
    )

    # Check if replay buffer is initialized
    if model.replay_buffer is not None:
        test_log = log(
            "Experience buffer size: " + str(model.replay_buffer.size()), test_log, True
        )  # current number of transitions
        test_log = log(
            "Experience buffer capacity: " + str(model.replay_buffer.buffer_size),
            test_log,
            True,
        )  # capacity
    else:
        test_log = log("Replay buffer is not initialized yet.", test_log, True)

    # check what ephem version to assume
    if "ephem_version" in params:
        ephem_version = params["ephem_version"]
    else:
        ephem_version = 1.0  # default to version 1 if not specified

    test_log = log(f"Assuming ephemeris version: {ephem_version}", test_log, True)

    # check env state dimensions
    test_log = log(
        f"Environment observation space shape: {env.observation_space.shape}",
        test_log,
        True,
    )

    if ephem_version == 2.0 and (
        env.observation_space.shape[0] == 10 or env.observation_space.shape[0] == 14
    ):
        test_log = log(
            "Environment observation space matches expected shape for ephemeris version 2.0",
            test_log,
            True,
        )
    elif ephem_version == 1.0 and env.observation_space.shape[0] == 5:
        test_log = log(
            "Environment observation space matches expected shape for ephemeris version 1.0",
            test_log,
            True,
        )
    elif ephem_version == 3.0 and env.observation_space.shape[0] == 22:
        test_log = log(
            "Environment observation space matches expected shape for ephemeris version 3.0",
            test_log,
            True,
        )
    elif ephem_version == 3.0 and (env.observation_space.shape[0] == 10):
        test_log = log(
            "Environment observation space matches expected shape for ephemeris version 3.0",
            test_log,
            True,
        )
    else:
        raise ValueError(
            f"Environment observation space shape {env.observation_space.shape} does not match expected shape for ephemeris version {ephem_version}"
        )

    if not params.get("read_replay_buffer", False):
        if ephem_version == 2.0:
            # Import training data into replay buffer
            test_log = log(
                "Importing training data into replay buffer v2.0", test_log, True
            )
            import_training_into_replay_buffer_v2(
                params[
                    "path_training_data"
                ],  # path to directory containing training ephemerides
                test_log,  # log
                model,  # SAC model
                env,
                params,
            )
        elif ephem_version == 1.0:
            # Import training data into replay buffer
            test_log = log(
                "Importing training data into replay buffer v1.0", test_log, True
            )
            import_training_into_replay_buffer(
                params[
                    "path_training_data"
                ],  # path to directory containing training ephemerides
                test_log,  # log
                model,  # SAC model
                env,
                params,
            )

    else:
        test_log = log(
            "Buffer read in from: " + params["path_replay_buffer"], test_log, True
        )

    test_log = log(
        "Updated experience buffer size: " + str(model.replay_buffer.size()),
        test_log,
        True,
    )  # current number of transitions

    # Train networks on replay buffer
    test_log, critic_loss_reduced, actor_loss_reduced = train_on_replay_buffer(
        model, params, test_log, env
    )

    return test_log, actor_loss_reduced, critic_loss_reduced


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
    num_vec_envs = params.get("num_vec_envs", 1)

    test_log = log(f"Reading ephemerides (version {ephem_version})", test_log, True)
    set_ephems = read_ephems(
        path_training_data, num_ephems_to_use, version=ephem_version, params=params
    )

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
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = (
        [],
        [],
        [],
        [],
        [],
    )
    for eph in tqdm(set_ephems, desc="Processing ephemerides"):
        for i in range(
            0, eph.num_vectors - 1, step_size
        ):  # -1 to ensure we have next_obs
            # Current state vector
            state_vec = eph.get_vector_at_index(i)

            env_type = params["env_type"]

            if env_type == "TwoBodyRendezvous_Polar_Env2":
                obs, action, reward, next_obs, done = (
                    package_ephem_state_into_polar_SART(
                        state_vec, env, i, eph.num_vectors, params
                    )
                )
            elif env_type == "TwoBodyRendezvous_Polar_Env":
                obs, action, reward, next_obs, done = (
                    package_ephem_state_into_polar_SART(
                        state_vec, env, i, eph.num_vectors, params
                    )
                )
            elif env_type == "TwoBodyRendezvous_Env":
                obs, action, reward, next_obs, done = (
                    package_ephem_state_into_cart_SART(
                        state_vec, env, i, eph.num_vectors, params
                    )
                )
            else:
                raise NotImplementedError(
                    f"Environment type {env_type} not supported in v2.0 import."
                )

            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)

            if len(obs_batch) == num_vec_envs:
                add_experience_to_replay_buffer(
                    model,
                    np.stack(obs_batch),
                    np.stack(action_batch),
                    np.array(reward_batch, dtype=np.float32).reshape(-1),
                    np.stack(next_obs_batch),
                    np.array(done_batch, dtype=np.float32).reshape(-1),
                )
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

    # optionally save the updated replay buffer to disk when complete
    if params.get("save_pre_training_only_replay_buffer", False):
        path_replay_buffer = os.path.join(
            params["output_dir_specific"], "replay_buffer_pre_training_only.pkl"
        )
        model.save_replay_buffer(path_replay_buffer)
        test_log = log(f"Replay buffer saved to {path_replay_buffer}", test_log, True)


def train_on_replay_buffer(model, params, test_log, env):
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
    import os

    import pandas as pd
    from tqdm import tqdm

    num_gradient_steps = params.get("pretrain_gradient_steps", 10000)
    batch_size = params.get("batch_size", 256)

    test_log = log(
        f"Training networks on replay buffer with {num_gradient_steps} gradient steps",
        test_log,
        True,
    )

    # Check that we have enough samples
    buffer_size = model.replay_buffer.size()
    if buffer_size < batch_size:
        test_log = log(
            f"Warning: Replay buffer has only {buffer_size} samples, less than batch size {batch_size}",
            test_log,
            True,
        )
        # Return empty loss arrays if we can't train
        return test_log, [], []

    test_log = log(f"Replay buffer size: {buffer_size}", test_log, True)
    test_log = log(f"Batch size: {batch_size}", test_log, True)

    # Set up a minimal logger for pre-training if one doesn't exist
    # model.train() requires a logger to record learning rate and other metrics
    # We'll use a simple logger that doesn't interfere with TensorBoard
    if not hasattr(model, "_logger") or model._logger is None:
        from stable_baselines3.common.logger import Logger

        # Create a minimal logger that just stores values without writing anywhere
        model._logger = Logger(folder=None, output_formats=[])

    # Set up progress tracking (needed by some internal SB3 methods)
    model._current_progress_remaining = 1.0

    # Set the number of timesteps so model.train() knows where we are
    if not hasattr(model, "_total_timesteps"):
        model._total_timesteps = 0
    if not hasattr(model, "num_timesteps"):
        model.num_timesteps = 0

    test_log = log("Training on replay buffer", test_log, True)

    # Track losses
    critic_losses = []
    actor_losses = []
    mean_rewards = []
    std_rewards = []

    # Optionally log progress
    log_interval = params.get("pt_log_interval", 1000)
    checkpoint_interval = params.get("checkpoint_freq", 10_000)
    gradient_batch_size = params.get("gradient_batch_size", 100)

    # Train the networks by sampling from replay buffer
    for step in tqdm(
        range(0, num_gradient_steps, gradient_batch_size), desc="Pre-training batches"
    ):
        # This performs multiple gradient updates on actor and critic networks at once
        steps_to_train = min(gradient_batch_size, num_gradient_steps - step)
        model.train(gradient_steps=steps_to_train)

        # Try to get losses from the model's internal tracking
        # Note: SAC stores these internally but they may not be accessible immediately
        try:
            # Check if _logger attribute exists (not the property)
            if hasattr(model, "_logger") and model._logger is not None:
                # Try to get values from logger's name_to_value dict
                if hasattr(model._logger, "name_to_value"):
                    critic_loss = model._logger.name_to_value.get(
                        "train/critic_loss", None
                    )
                    actor_loss = model._logger.name_to_value.get(
                        "train/actor_loss", None
                    )

                    if critic_loss is not None:
                        critic_losses.append(critic_loss)
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
        except (AttributeError, KeyError):
            # Logger not set up or accessible, skip logging
            pass

        if (step) % log_interval == 0:
            if critic_losses and actor_losses:
                avg_critic = np.mean(critic_losses[-log_interval:])
                avg_actor = np.mean(actor_losses[-log_interval:])
                tqdm.write(
                    f"Step {step + 1}: Critic Loss: {avg_critic:.4f}, Actor Loss: {avg_actor:.4f}"
                )
            else:
                tqdm.write(f"Completed {step}/{num_gradient_steps} gradient steps")

        # also checkpoint the model
        if (step) % checkpoint_interval == 0:
            # evaluate current policy
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=params.get("n_eval_episodes", 10),
                deterministic=True,
            )

            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            tqdm.write(
                f"Evaluation after {step} steps: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}"
            )

            if mean_reward >= max(mean_rewards, default=-np.inf):
                path = os.path.join(params["output_dir_specific"], "checkpoints")
                checkpoint_path = os.path.join(
                    path, f"sac_pretrained_step_{step + 1}.zip"
                )
                model.save(checkpoint_path)
                tqdm.write(f"Saved model checkpoint to {checkpoint_path}")

    test_log = log(
        f"Completed {num_gradient_steps} gradient steps on replay buffer",
        test_log,
        True,
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

        pd.DataFrame({"mean_reward": mean_rewards, "std_reward": std_rewards}).to_csv(
            os.path.join(path, "eval_rewards.csv"), index=True
        )

        test_log = log(f"Saved loss curves to {path}", test_log, True)

    return test_log, critic_loss_reduced, actor_loss_reduced


def package_ephem_state_into_polar_SART(
    state_vec, env, current_index, num_vectors, params
):
    env.reset()

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

    state_input = np.array(
        [x, y, vx, vy, m, x_target, y_target, vx_target, vy_target, ttg]
    )

    # manually set the state of the environment
    unwrapped_env = env.unwrapped

    obs, info = unwrapped_env.set_state(state_input)

    # Verify that the state was set correctly
    state_check = unwrapped_env.get_cartesian_state()
    if not np.allclose(state_check, state_input, atol=1e-5):
        raise ValueError(
            "Environment state does not match expected state after setting."
        )

    # Convert alpha_x, alpha_y to flight path angle components
    alpha_vr, alpha_theta = convert_attitude_from_cartesian_to_radial(
        x, y, alpha_x, alpha_y
    )

    # Action in terms of (u, alpha_cos_fpa, alpha_sin_fpa)
    action = np.array([u, alpha_vr, alpha_theta], dtype=np.float32)

    # Step the environment to get next_obs, reward, done
    next_obs, reward, done, truncated, info = unwrapped_env.step(action)

    if current_index + 1 >= num_vectors:
        done = True

    return obs, action, reward, next_obs, done


def package_ephem_state_into_cart_SART(
    state_vec, env, current_index, num_vectors, params
):
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

    state_input = np.array(
        [x, y, vx, vy, m, x_target, y_target, vx_target, vy_target, ttg]
    )

    # manually set the state of the environment
    unwrapped_env = env.unwrapped

    obs, info = unwrapped_env.set_state(state_input)

    # Verify that the state was set correctly
    state_check = unwrapped_env.get_cartesian_state()
    if not np.allclose(state_check, state_input, atol=1e-5):
        raise ValueError(
            "Environment state does not match expected state after setting."
        )

    # Action in terms of (u, alpha_cos_fpa, alpha_sin_fpa)
    action = np.array([u, alpha_x, alpha_y], dtype=np.float32)

    # Step the environment to get next_obs, reward, done
    next_obs, reward, done, truncated, info = unwrapped_env.step(action)

    if current_index + 1 >= num_vectors:
        done = True

    return obs, action, reward, next_obs, done


def rollout_model(env, params, model, test_log):
    flag_report_live = params.get("flag_report_live", False)

    # reset the env
    obs, info = env.reset(seed=params.get("seed_traj", 42))
    eph = Ephemeris_v2()  # create new ephemeris object

    sum_reward = 0.0

    rollout_data = SACRolloutData_TBR_polar()

    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    while flag_continue:
        # step the env
        action, _states = model.predict(obs, deterministic=True)
        unwrapped_env = env.unwrapped
        state_cart = unwrapped_env.get_cartesian_state()
        throttle = action[0]
        alpha_fpa_cos = action[1]
        alpha_fpa_sin = action[2]

        # dim state
        t_i = info["Elapsed time"]
        t_i_days = t_i / (3600 * 24)
        x_i = state_cart[0]
        y_i = state_cart[1]
        vx_i = state_cart[2]
        vy_i = state_cart[3]
        m_i = state_cart[4]
        x_target_i = state_cart[5]
        y_target_i = state_cart[6]
        vx_target_i = state_cart[7]
        vy_target_i = state_cart[8]
        ttg_i = state_cart[9]

        # info of interest
        pos_reward = info.get("pos_reward", None)
        vel_reward = info.get("vel_reward", None)
        mass_reward = info.get("mass_reward", None)
        throttle_reward = info.get("throttle_reward", None)
        alpha_x = info.get("alpha_x", None)
        alpha_y = info.get("alpha_y", None)
        position_res = info.get("pos_residual", None)
        velocity_res = info.get("vel_residual", None)

        # log data to ephemeris
        eph.add_data(
            t_i,
            x_i,
            y_i,
            vx_i,
            vy_i,
            m_i,
            x_target_i,
            y_target_i,
            vx_target_i,
            vy_target_i,
            ttg_i,
            alpha_x,
            alpha_y,
            throttle,
        )

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        obs, reward, done, truncated, info = env.step(action)
        state_cart = env.get_cartesian_state()

        # get relevant information
        pos_reward = info["pos_reward"]
        vel_reward = info["vel_reward"]
        mass_reward = info["mass_reward"]
        throttle_reward = info["throttle_reward"]
        v_current_nd = info["v_current_nd"]
        v_target_nd = info["v_target_nd"]
        v_r_unit = info["v_r_unit"]
        v_t_unit = info["v_t_unit"]
        delta_cos_eta = obs[4] - obs[1]
        delta_sin_eta = obs[5] - obs[2]
        delta_target_v_nd = v_target_nd - v_current_nd
        d_v_r_unit = info["v_r_target_unit"] - info["v_r_unit"]
        d_v_t_unit = info["v_t_target_unit"] - info["v_t_unit"]

        count_step = count_step + 1

        # store the results
        rollout_data.add_step(
            info["Elapsed time"] / 86400,  # elapsed time in days #1
            reward,  # reward #2
            action[0],  # throttle #3
            action[1],  # alpha_r #4
            action[2],  # alpha_theta #5
            obs[0],  # r_nd #6
            obs[1],  # eta_cos_nd #7
            obs[2],  # eta_sin_nd #8
            v_current_nd,  # v_nd #9
            v_r_unit,  # v_r_unit #10
            v_t_unit,  # v_t_unit #11
            obs[21],  # mass_nd #12
            obs[18],  # delta target_r_nd #13
            delta_cos_eta,  # delta target_eta_cos_nd #14
            delta_sin_eta,  # delta target_eta_sin_nd #15
            delta_target_v_nd,  # delta target_v_nd #16
            d_v_r_unit,  # delta v_r_unit #17
            d_v_t_unit,  # delta v_t_unit #18
            obs[20],  # TTG_nd #19
            pos_reward,  # position reward #20
            vel_reward,  # velocity reward #21
            mass_reward,  # mass reward #22
            throttle_reward,  # throttle reward #23
            position_res,
            velocity_res,
        )

        if done or truncated:
            break

    test_log = log("Test trajectory complete", test_log, flag_report_live)
    test_log = log("Steps taken: " + str(count_step), test_log, flag_report_live)
    test_log = log("Total reward: " + str(sum_reward), test_log, flag_report_live)
    test_log = log("Final x: " + str(obs[0]) + " ", test_log, flag_report_live)
    test_log = log("Final y: " + str(obs[1]) + " ", test_log, flag_report_live)
    test_log = log("Final vx: " + str(obs[2]) + " ", test_log, flag_report_live)
    test_log = log("Final vy: " + str(obs[3]) + " ", test_log, flag_report_live)
    test_log = log("Final m: " + str(obs[4]) + " ", test_log, flag_report_live)
    test_log = log("Final sma: " + str(obs[6]) + " ", test_log, flag_report_live)
    test_log = log("Final ecc: " + str(arr_OE[1]) + " ", test_log, flag_report_live)
    test_log = log("terminated: " + str(terminated) + " ", test_log, flag_report_live)
    test_log = log("truncated: " + str(truncated) + " ", test_log, flag_report_live)
    # final env info
    for key, value in info.items():
        if key != "ODE Solution":
            test_log = log(f"{key}: {value}", test_log, flag_report_live)
    return test_log, eph, rollout_data


def extract_experiences_from_ephem(eph, params):
    # Step through ephem and gather observations, actions, rewards, next_obs, dones
    obs_batch = []
    action_batch = []
    reward_batch = []
    next_obs_batch = []
    done_batch = []

    num_steps_per_SART = params["ephem_step_size"]

    # Only add every num_steps_per_SART-th vector
    for i in range(
        0, eph.num_vectors - 1, num_steps_per_SART
    ):  # -1 to ensure we have next_obs
        state_vec = eph.get_vector_at_index(i)

        env_type = params["env_type"]

        # unpack state
        t = state_vec[0]
        current_state = state_vec[1:6]  # x, y, vx, vy, m
        target_state = state_vec[6:10]  # x_t, y_t, vx, vy
        ttg = state_vec[10]
        alpha_x = state_vec[11]
        alpha_y = state_vec[12]
        u = state_vec[13]

        # compute polar observation
        if (
            env_type == "TwoBodyRendezvous_Polar_Env2"
            or env_type == "TwoBodyRendezvous_Polar_Env"
        ):
            obs, _ = create_relative_polar_observation_fast(
                params, current_state, target_state, ttg
            )
        else:
            raise NotImplementedError("Check env type")

        # polar action
        # Convert alpha_x, alpha_y to polar form
        x = current_state[0]
        y = current_state[1]
        alpha_vr, alpha_theta = convert_attitude_from_cartesian_to_radial(
            x, y, alpha_x, alpha_y
        )
        action = np.array([u, alpha_vr, alpha_theta], dtype=np.float32)
        u = action[0]

        # compute the reward
        # compute polar observation
        if (
            env_type == "TwoBodyRendezvous_Polar_Env2"
            or env_type == "TwoBodyRendezvous_Polar_Env"
        ):
            reward, terminated, truncated, _ = compute_reward_fast(
                params, current_state, ttg, target_state, u
            )
        else:
            raise NotImplementedError("Check env type")

        # check if terminal
        done = terminated or truncated

        # next observation
        next_t = t + params["env_step_size"]
        if next_t > eph.get_vector_at_index(eph.num_vectors - 1)[0]:
            break
        else:
            next_state_vec = eph.get_interpolated_vector_at_time(next_t)
            next_current_state = next_state_vec[1:6]
            next_target_state = next_state_vec[6:10]
            next_ttg = next_state_vec[10]

            # compute polar observation
            if (
                env_type == "TwoBodyRendezvous_Polar_Env2"
                or env_type == "TwoBodyRendezvous_Polar_Env"
            ):
                next_obs, _ = create_relative_polar_observation_fast(
                    params, next_current_state, next_target_state, next_ttg
                )
            else:
                raise NotImplementedError("Check env type")

        obs_batch.append(obs)
        action_batch.append(action)
        reward_batch.append(reward)
        next_obs_batch.append(next_obs)
        done_batch.append(done)

    # update the last done to be terminal
    if len(done_batch) > 0:
        done_batch[-1] = True

    # After the loop, before buffer import:
    obs_batch = np.stack(obs_batch)  # shape (N, obs_dim)
    action_batch = np.stack(action_batch)  # shape (N, action_dim)
    reward_batch = np.array(reward_batch).reshape(-1, 1)  # shape (N, 1)
    next_obs_batch = np.stack(next_obs_batch)  # shape (N, obs_dim)
    done_batch = np.array(done_batch).reshape(-1, 1)  # shape (N, 1)

    return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch


def create_relative_polar_observation_fast(
    params, current_state_t, target_state_t, TTG
):
    l_star = params["l_star"]
    t_star = params["t_star"]
    m_star = params["m_star"]

    x_current_nd = current_state_t[0] / l_star
    y_current_nd = current_state_t[1] / l_star
    vx_current_nd = current_state_t[2] / (l_star / t_star)
    vy_current_nd = current_state_t[3] / (l_star / t_star)
    mass_current_nd = current_state_t[4] / m_star

    x_target_nd = target_state_t[0] / l_star
    y_target_nd = target_state_t[1] / l_star
    vx_target_nd = target_state_t[2] / (l_star / t_star)
    vy_target_nd = target_state_t[3] / (l_star / t_star)

    TTG_nd = TTG / t_star

    # convert to polar coordinates
    r_nd_0, eta_nd_0, v_r_nd_0, v_eta_nd_0 = cartesian_to_polar(
        x_current_nd, y_current_nd, vx_current_nd, vy_current_nd
    )

    r_nd_target, eta_nd_target, v_r_nd_target, v_eta_nd_target = cartesian_to_polar(
        x_target_nd, y_target_nd, vx_target_nd, vy_target_nd
    )

    cos_eta = np.cos(eta_nd_0)
    sin_eta = np.sin(eta_nd_0)
    cos_eta_target = np.cos(eta_nd_target)
    sin_eta_target = np.sin(eta_nd_target)

    # determine angular momentum
    h_nd_0 = r_nd_0 * v_eta_nd_0
    h_nd_target = r_nd_target * v_eta_nd_target
    # total velocity magnitudes
    v_comp = (vx_current_nd**2 + vy_current_nd**2) ** 0.5
    v_comp_target = (vx_target_nd**2 + vy_target_nd**2) ** 0.5

    # transpose velocites
    v_t_nd = h_nd_0 / r_nd_0
    v_t_target_nd = h_nd_target / r_nd_target

    v_r_nd = (max(v_comp**2 - v_t_nd**2, 1e-6)) ** 0.5
    v_r_target_nd = (max(v_comp_target**2 - v_t_target_nd**2, 1e-6)) ** 0.5

    v_r_unit = v_r_nd / v_comp if v_comp != 0 else 0.0
    v_t_unit = v_t_nd / v_comp if v_comp != 0 else 0.0
    v_r_target_unit = v_r_target_nd / v_comp_target if v_comp_target != 0 else 0.0
    v_t_target_unit = v_t_target_nd / v_comp_target if v_comp_target != 0 else 0.0

    # construct relative target vector
    delta_r = r_nd_target - r_nd_0
    delta_eta_cos = cos_eta_target - cos_eta
    delta_eta_sin = sin_eta_target - sin_eta
    delta_v = v_comp_target - v_comp
    delta_v_r = v_r_target_unit - v_r_unit
    delta_v_t = v_t_target_unit - v_t_unit

    polar_observation = np.array(
        [
            # Spherical Position SC
            r_nd_0,  # 0
            cos_eta,  # 1
            sin_eta,  # 2
            # Spherical Position Planet
            r_nd_target,  # 3
            cos_eta_target,  # 4
            sin_eta_target,  # 5
            # Cartesian Position SC
            x_current_nd,  # 6
            y_current_nd,  # 7
            vx_current_nd,  # 8
            vy_current_nd,  # 9
            # Cartesian Position Planet
            x_target_nd,  # 10
            y_target_nd,  # 11
            vx_target_nd,  # 12
            vy_target_nd,  # 13
            # Differences Cartesian
            x_target_nd - x_current_nd,  # 14
            y_target_nd - y_current_nd,  # 15
            vx_target_nd - vx_current_nd,  # 16
            vy_target_nd - vy_current_nd,  # 17
            # Differences Magnitudes
            r_nd_target - r_nd_0,  # 18
            v_comp_target - v_comp,  # 19
            # Time to Go
            TTG_nd,  # 20
            mass_current_nd,  # 21
        ],
        dtype=np.float32,
    )

    env_data = {
        "arr_r_polar_nd": [r_nd_0, eta_nd_0],
        "arr_v_polar_nd": [v_r_nd_0, v_eta_nd_0],
        "arr_rf_polar_nd": [r_nd_target, eta_nd_target],
        "arr_vf_polar_nd": [v_r_nd_target, v_eta_nd_target],
        "h_nd_0": h_nd_0,
        "h_nd_target": h_nd_target,
        "cos_eta": cos_eta,
        "sin_eta": sin_eta,
        "cos_eta_target": cos_eta_target,
        "sin_eta_target": sin_eta_target,
        "fpa_nd": 0.0,
        "fpa_target_nd": 0.0,
        "v_r_unit": v_r_unit,
        "v_t_unit": v_t_unit,
        "v_r_target_unit": v_r_target_unit,
        "v_t_target_unit": v_t_target_unit,
        "v_current_nd": v_comp,
        "v_target_nd": v_comp_target,
    }

    return polar_observation, env_data


def compute_reward_fast(
    params,
    current_state_t,
    TTG,
    target_state_t,
    u,
    step_count=0,
    timesteps_in_prop=1000,
):
    # non-dimensionalize states
    x_nd = current_state_t[0] / params["l_star"]
    y_nd = current_state_t[1] / params["l_star"]
    vx_nd = current_state_t[2] / (params["l_star"] / params["t_star"])
    vy_nd = current_state_t[3] / (params["l_star"] / params["t_star"])
    x_target_nd = target_state_t[0] / params["l_star"]
    y_target_nd = target_state_t[1] / params["l_star"]
    vx_target_nd = target_state_t[2] / (params["l_star"] / params["t_star"])
    vy_target_nd = target_state_t[3] / (params["l_star"] / params["t_star"])

    TTG_nd = TTG / params["t_star"]
    r_nd = (x_nd**2 + y_nd**2) ** 0.5

    mass = current_state_t[4] / params["m_star"]

    # extract orbital elements
    r_0, eta_0, v_r_0, v_eta_0 = cartesian_to_polar(
        current_state_t[0], current_state_t[1], current_state_t[2], current_state_t[3]
    )

    sc = Spacecraft(
        r_0, eta_0, v_r_0, v_eta_0, current_state_t[4], params["max_T"], params["ISP"]
    )
    sc.update_state_cartesian(
        current_state_t[0],
        current_state_t[1],
        current_state_t[2],
        current_state_t[3],
        current_state_t[4],
    )
    arr_OE = sc.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, Constants.MU_SUN_M)
    e = arr_OE[1]

    # extract the current target state
    x_target_nd_i = target_state_t[0] / params["l_star"]
    y_target_nd_i = target_state_t[1] / params["l_star"]
    vx_target_nd_i = target_state_t[2] / (params["l_star"] / params["t_star"])
    vy_target_nd_i = target_state_t[3] / (params["l_star"] / params["t_star"])

    # calculate the current distance to target
    # MUST DO, otherwise you can get two big bumps rather than one bump.
    d_r_nd = np.sqrt((x_nd - x_target_nd_i) ** 2 + (y_nd - y_target_nd_i) ** 2)
    d_v_nd = np.sqrt((vx_nd - vx_target_nd_i) ** 2 + (vy_nd - vy_target_nd_i) ** 2)

    # these rewards get bigger (max 1) the closer you get to the target
    pos_reward = np.exp(-params["r_dist_weight"] * d_r_nd**2)
    vel_reward = np.exp(-params["v_dist_weight"] * d_v_nd**2)

    # These values will be negative, approaching zero as you get closer to the target
    pos_reward = (-1 + pos_reward) * params["pos_r_weight"]
    vel_reward = (-1 + vel_reward) * params["vel_r_weight"]

    # ttg weighting
    time_weight = np.exp(-params["time_dist_weight"] * TTG_nd**2)
    pos_reward *= time_weight
    vel_reward *= time_weight

    # This value will always be negative
    throttle_reward = -u * params["throttle_r_weight"]

    reward = pos_reward + vel_reward + throttle_reward

    terminated = False

    # terminal rewards
    exceeded_min = r_nd < 0.1  # too close to sun
    exceeded_max = r_nd > 5.0  # too far from sun
    # episode_timeout = self.step_count >= self.max_episode_steps
    episode_timeout = TTG_nd <= 0.0  # out of time
    fuel_exceeded = mass <= 0.01  # out of fuel
    eccentricity_exceeded = e >= 1.0

    # Failure Conditions
    terminated = False
    truncated = False
    if exceeded_min or exceeded_max:
        reward -= 10.0
        terminated = True

    if params.get("flag_hyperbolic_termination", False):
        if eccentricity_exceeded:
            reward -= 100.0
            terminated = True

    if fuel_exceeded:
        reward -= 10.0
        terminated = True

    if episode_timeout:
        # Nothing good or bad assigned to this
        truncated = True

    # pack additional env info
    env_info = {
        "pos_residual": d_r_nd,
        "vel_residual": d_v_nd,
        "time_r_component": time_weight,
        "throttle_r_component": throttle_reward,
        "pos_r_component": pos_reward,
        "vel_r_component": vel_reward,
        "terminated": terminated,
        "truncated": truncated,
    }

    return reward, terminated, truncated, env_info

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.core.rollouts import SACRolloutData_TBR_polar
from astro_compass.core.spacecraft import Spacecraft
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
    num_ephems = params["num_ephems"]
    test_log = log(
        "Using only first " + str(num_ephems) + " ephemerides", test_log, True
    )
    set_ephems = read_ephems(path_training_data, Ephemeris_v2)
    test_log = log("Reading ephemerides", test_log, True)

    # count number of ephemerides
    num_ephems = len(set_ephems)
    test_log = log(
        "Number of ephemerides loaded into replay buffer: " + str(num_ephems),
        test_log,
        True,
    )
    set_ephems = set_ephems[:num_ephems]
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
    num_ephems = params.get("num_ephems", None)
    ephem_version = params.get("ephem_version", 2.0)
    step_size = params.get("ephem_step_size", 1)  # Sample every Nth vector
    num_vec_envs = params.get("num_vec_envs", 1)

    test_log = log(f"Reading ephemerides (version {ephem_version})", test_log, True)
    set_ephems = read_ephems(path_training_data, Ephemeris_v2)

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

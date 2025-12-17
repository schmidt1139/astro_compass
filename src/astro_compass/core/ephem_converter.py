import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env_target import (
    TwoBody_Orb2Orb_Transfer_Env_target,
)
from astro_compass.utils.log_utils import log
from astro_compass.utils.state_vector_utils import (
    convert_attitude_from_cartesian_to_radial,
)


def _read_single_ephem(path, eph_class):
    eph = eph_class()
    eph.read(path)
    return eph


def read_ephems(
    ephem_dir,
    eph_class=None,
    num_workers=4,
):
    filenames = os.listdir(ephem_dir)
    paths = [os.path.join(ephem_dir, file) for file in filenames]

    list_ephems = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _read_single_ephem,
                path,
                eph_class,
            )
            for path in paths
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            list_ephems.append(f.result())

    return list_ephems, filenames


def extract_experiences_from_ephem_TBT(eph, params):
    # Step through ephem and gather observations, actions, rewards, next_obs, dones
    obs_batch = []
    action_batch = []
    reward_batch = []
    next_obs_batch = []
    done_batch = []

    num_steps_per_SART = params["ephem_step_size"]
    env = TwoBody_Orb2Orb_Transfer_Env_target

    # Only add every num_steps_per_SART-th vector
    for i in range(
        0, eph.num_vectors - 1, num_steps_per_SART
    ):  # -1 to ensure we have next_obs
        state_vec = eph.get_vector_at_index(i)

        # unpack state
        t = state_vec[0]
        current_state = state_vec[1:6]  # x, y, vx, vy, m
        target_state = state_vec[6:10]  # x_t, y_t, vx, vy
        ttg = state_vec[10]
        alpha_x = state_vec[11]
        alpha_y = state_vec[12]
        u = state_vec[13]

        # compute polar observation
        state = np.concatenate((current_state, target_state))
        state_dict = env.decode_state(state)
        obs, _ = env.compute_obs_fast_TBT(state_dict, params, ttg)

        # polar action
        # Convert alpha_x, alpha_y to polar form
        x = current_state[0]
        y = current_state[1]
        alpha_vr, alpha_theta = convert_attitude_from_cartesian_to_radial(
            x, y, alpha_x, alpha_y
        )
        action = np.array([u, alpha_vr, alpha_theta], dtype=np.float32)
        u = action[0]

        reward, terminated, truncated, _ = env.compute_reward_fast_TBT(
            state_dict,
            params,
            u,
            ttg,
        )

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
            state = np.concatenate((next_current_state, next_target_state))
            state_dict = env.decode_state(state)
            next_obs, _ = env.compute_obs_fast_TBT(state, params, next_ttg)

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


def import_training_into_replay_buffer_v3(set_ephems, test_log, model, params):
    """
    Import training data from ephemeris v3.0 files into the replay buffer.
    Handles 10-dimensional observation space (x, y, vx, vy, m, x_t, y_t, vx_t, vy_t, ttg).

    Does this without env overhead by directly setting states in the replay buffer.

    Assumes ephemeris v3.0 format, which contains a moving target with full state info.
    """

    cores = params.get("cores", 1)
    batches = []
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [
            executor.submit(extract_experiences_from_ephem_TBT, eph, params)
            for eph in set_ephems
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            batches.append(f.result())

    # Now add all experiences to replay buffer in batches
    for obs_batch, action_batch, reward_batch, next_obs_batch, done_batch in tqdm(
        batches, total=len(batches), desc="Adding experiences to replay buffer"
    ):
        batch_size = obs_batch.shape[0]
        for i in range(batch_size):
            obs = np.array(obs_batch[i], dtype=np.float32)
            action = np.array(action_batch[i], dtype=np.float32)
            next_obs = np.array(next_obs_batch[i], dtype=np.float32)
            reward = np.array(reward_batch[i], dtype=np.float32)
            done = np.array(done_batch[i], dtype=np.float32)

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

    # report updated buffer size
    test_log = log(
        "Updated experience buffer size: " + str(model.replay_buffer.size()),
        test_log,
        True,
    )  # current number of transitions

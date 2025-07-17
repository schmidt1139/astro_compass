import numpy as np
import gymnasium as gym
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)

from Ephemeris import Ephemeris


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")


steps_per_traj = 86400 * 365 * 1.1 / 3600
num_traj = 1


def test_runnable_env(env, num_trajectories, num_steps_per_traj):
    count_traj = 0
    arr_episodes = np.array([])
    arr_reward_totals = np.array([])
    total_steps_in_env = 0

    for count_traj in range(0, num_trajectories):
        # reset the environment
        steps = 0
        r_tot = 0.0

        eph = Ephemeris()
        observation, info = env.reset(seed=42)

        while steps < steps_per_traj:
            # Arbitrary test action
            action = env.action_space.sample()
            action[0] = 1.0
            action[1] = -1.0 + np.tanh(steps / steps_per_traj)
            action[2] = -1.0 + np.tanh(2 * steps / steps_per_traj)

            observation, reward, terminated, truncated, info = env.step(action)

            r_tot = r_tot + reward

            elapsed_time = info["Elapsed time"]

            if terminated:
                break

            if truncated:
                break

            eph.add_data(
                elapsed_time,
                observation[0],
                observation[1],
                observation[2],
                observation[3],
                observation[4],
                action[1],
                action[2],
                action[0],
            )

            # print( elapsed_time, a, e, reward )
            steps = steps + 1

        arr_episodes = np.append(arr_episodes, count_traj)
        arr_reward_totals = np.append(arr_reward_totals, r_tot)

        total_steps_in_env = total_steps_in_env + steps

        print(
            "Episode count: "
            + str(count_traj + 1)
            + " of "
            + str(num_traj)
            + "   Total steps: "
            + str(steps)
        )
        print("Total steps in environment: " + str(total_steps_in_env))

        if count_traj == num_traj - 1:
            print("Plotting last trajectory...")
            eph.plot_xy(info["planet_radii"])
            eph.plot_xy_ref_orbit(observation[6], "Earth Orbit")
            eph.plot_all_ephemeris_data()

    print("Test successful")


test_runnable_env(env, num_traj, steps_per_traj)

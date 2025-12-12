import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plot
import numpy as np
from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from astro_compass.Ephemeris import Ephemeris

# register the environment if it isn't registered
if "HohmannTransferEnv-v0" not in envs.registry.keys():
    register(
        id="HohmannTransferEnv-v0",
        entry_point="Hohmann_Transfer_Env:HohmannTransferEnv",
    )


# initialize the environment
env = gym.make("HohmannTransferEnv-v0")


steps_per_traj = 10000
num_traj = 10


def test_runnable_env(env, num_trajectories, num_steps_per_traj):
    count_traj = 0
    arr_episodes = np.array([])
    arr_reward_totals = np.array([])
    total_steps_in_env = 0

    for count_traj in range(0, num_traj):
        # reset the environment
        steps = 0
        r_tot = 0.0

        eph = Ephemeris()
        observation, info = env.reset()

        while steps < steps_per_traj:
            # Sample randomly from the action space. Since the action is a delta-V
            # magnitude in km/s, and the action space is unbounded (-inf to inf) the
            # test maneuver that is returned will be sampled from a Gaussian normal
            # distribution with a mean of 0 and a standard deviation of 1. We
            # devide by 1000 in this test case to give relatively small maneuvers.
            action = env.action_space.sample() / 100

            observation, reward, terminated, truncated, info = env.step(action)

            r_tot = r_tot + reward

            elapsed_time = info["Elapsed time"]

            if terminated:
                break

            eph.add_data(
                elapsed_time,
                observation[0],
                observation[1],
                observation[2],
                observation[3],
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
            fig = eph.plot_xy(info["planet_radii"])
            plot.show(fig)

    fig_reward, ax = plot.subplots(figsize=(6, 6))

    ax.plot(arr_episodes, arr_reward_totals, label="Total Reward")

    # Customize the figure
    ax.set_title("Total Reward Per Episode")
    ax.set_xlabel("Episode Count")
    ax.set_ylabel("Total R")
    ax.legend()
    ax.grid(False)
    plot.show(fig_reward)

    print("Test successful")


test_runnable_env(env, num_traj, steps_per_traj)

import os

import matplotlib.pyplot as plt
import numpy as np


class RolloutPlotter:
    def __init__(self, rollout_data, path_output=None):
        self.rollout_data = rollout_data
        self.path_output = path_output

    def plot(self):
        self.plot_return()
        self.plot_reward()
        self.plot_throttle()
        self.plot_attitude()

    def plot_return(self):
        # plot reward
        plt.figure()

        total_reward = np.cumsum(self.rollout_data.rewards)
        plt.plot(
            total_reward,
            label="Cumulative Reward",
        )
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.title("SAC Training Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_Training_Reward.png"), dpi=300)

    def plot_reward(self):
        # plot reward per step
        plt.figure()
        plt.plot(
            self.rollout_data.rewards,
            label="Reward",
        )

        # TODO
        # get reward components by looking at keys in the info dict
        # plt.plot(
        #     self.rollout_data.arr_reward_mass,
        #     label="Reward Mass Component",
        # )
        # plt.plot(
        #     self.rollout_data.arr_reward_distance,
        #     label="Reward Distance Component",
        # )
        plt.xlabel("Steps")
        plt.ylabel("Reward per Step")
        plt.title("SAC Training Reward Per Step")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(self.path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300
        )

    def plot_throttle(self):
        plt.figure()
        plt.plot(
            np.array(self.rollout_data.actions)[:, 0],
            label="Throttle",
        )
        plt.xlabel("Steps")
        plt.ylabel("Throttle")
        plt.title("SAC Training Throttle")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(self.path_output, "SAC_Training_Throttle.png"), dpi=300
        )

    def plot_attitude(self):
        plt.figure()
        plt.plot(
            np.array(self.rollout_data.actions)[:, 1],
            label="alpha_r",
        )
        plt.plot(
            np.array(self.rollout_data.actions)[:, 0],
            label="alpha_r_theta",
        )
        plt.xlabel("Steps")
        plt.ylabel("Attitude")
        plt.title("SAC Training Burn Attitude")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_Training_Alpha.png"), dpi=300)

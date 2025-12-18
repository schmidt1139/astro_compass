import os

import matplotlib.pyplot as plt
from astro_compass.vis.ephem_plotter import EphemPlotter

from astro_compass.constants.constants import Constants


class RolloutPlotter:
    def __init__(self, rollout_data, path_output=None):
        self.rollout_data = rollout_data
        self.path_output = path_output

    def plot(
        self,
        eph=None,
        ephem_H=None,
    ):
        self.plot_return_vs_time()
        self.plot_reward_vs_time()
        self.plot_throttle_vs_time()
        self.plot_attitude_vs_time()
        self.plot_state_vs_time()
        self.plot_sma_vs_time()
        self.plot_ecc_vs_time()
        if eph is not None:
            self.plot_ephem(eph, ephem_H)

    def plot_return_vs_time(self):
        # plot reward over time
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_reward_tot,
            label="Reward",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("Reward")
        plt.title("SAC Training Reward over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_Training_Reward.png"), dpi=300)

    def plot_reward_vs_time(self):
        # plot reward over time per step
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_reward,
            label="Reward",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_reward_mass,
            label="Reward Mass Component",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_reward_distance,
            label="Reward Distance Component",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("Reward per Step")
        plt.title("SAC Training Reward Per Step over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(self.path_output, "SAC_Training_Reward_Per_Step.png"), dpi=300
        )

    def plot_throttle_vs_time(self):
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_throttle,
            label="Throttle",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("Throttle")
        plt.title("SAC Training Throttle over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(
            os.path.join(self.path_output, "SAC_Training_Throttle.png"), dpi=300
        )

    def plot_attitude_vs_time(self):
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_alpha_x,
            label="alpha_x",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_alpha_y,
            label="alpha_y",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("Attitude")
        plt.title("SAC Training Burn Attitude over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_Training_Alpha.png"), dpi=300)

    def plot_state_vs_time(self):
        plt.figure()
        plt.plot(self.rollout_data.arr_time, self.rollout_data.arr_x, label="x")
        plt.plot(self.rollout_data.arr_time, self.rollout_data.arr_y, label="y")
        plt.plot(self.rollout_data.arr_time, self.rollout_data.arr_vx, label="vx")
        plt.plot(self.rollout_data.arr_time, self.rollout_data.arr_vy, label="vy")
        plt.xlabel("Time [days]")
        plt.ylabel("ND state")
        plt.title("SAC Training ND State over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_ND_State.png"), dpi=300)

    def plot_sma_vs_time(self):
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_sma,
            label="sma",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_sma_target,
            label="sma_target",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("SMA Achieved [m]")
        plt.title("SAC Achieved SMA over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_SMA_Achieved.png"), dpi=300)

    def plot_ecc_vs_time(self):
        plt.figure()
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_ecc,
            label="ecc",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_ecc_target,
            label="ecc_target",
        )
        plt.plot(
            self.rollout_data.arr_time,
            self.rollout_data.arr_ecc_max,
            label="ecc_max",
            linestyle="--",
            color="red",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("ECC Achieved")
        plt.title("SAC Achieved ECC over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Force grid on with some transparency
        plt.savefig(os.path.join(self.path_output, "SAC_ECC_Achieved.png"), dpi=300)

    # generate and save figures
    def plot_ephem(self, eph, eph_h=None):
        vis = EphemPlotter(eph)
        fig_orb = vis.plot_xy()
        if eph_h is not None:
            fig_orb = vis.overlay_ref_orbit(
                ephem=eph_h, label="Hamiltonian Trajectory", color_in="#f89540"
            )
        fig_orb = vis.plot_xy_ref_orbit(Constants.SMA_MARS, "Mars", "#b7410e")
        fig_orb = vis.plot_xy_ref_orbit(Constants.SMA_EARTH, "Earth")
        fig_orb.savefig(os.path.join(self.path_output, "SAC_Test_Traj.png"), dpi=300)

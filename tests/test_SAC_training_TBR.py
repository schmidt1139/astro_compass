import os
import random
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from astro_compass.utils.log_utils import log
from astro_compass.utils.path_utils import DATA_ROOT, PROJECT_ROOT
from astro_compass.utils.plotting_utils import plot_SAC_training
from astro_compass.utils.rl_utils import log_training_perf
from astro_compass.utils.state_vector_utils import cartesian_to_polar
from astro_compass.utils.test_utils import compare_log_files_with_tolerance

os.chdir(PROJECT_ROOT)
print("Now working in:", os.getcwd())

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))


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
        ep_infos = list(self.model.ep_info_buffer)  # current buffer snapshot
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


def test_SAC_training_TBR(flag_report_live=False, seed_in=42):
    # set random seeds for reproducibility
    random.seed(seed_in)
    np.random.seed(seed_in)
    import torch

    torch.manual_seed(seed_in)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_in)

    # define normalization parameters (for NN)
    params = {
        "mu": Constants.MU_SUN * 10 ** (9),  # sun mu [m^3/s^2]
        "max_T": 1.33,  # max spacecraft thrust [N]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # assumed time of flight [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (149598023000**3 / (Constants.MU_SUN * 10 ** (9)))
        ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
        "env_step_size": 3600 * 24,  # environment step size [s]
    }

    # initialize the environment
    env = TwoBodyRendezvous_Env(
        mu=params["mu"],  # solar gravitational parameter in m^3/s^2
        max_T=params["max_T"],  # max thrust in N
        ISP=params["ISP"],  # ISP in seconds
        TOF=params["TOF"],  # time of flight in seconds
        l_star=params["l_star"],  # characteristic length in m
        m_star=params["m_star"],  # characteristic mass in kg
        t_star=params["t_star"],  # characteristic time in s
        g0=params["g0"],  # gravitational acceleration at Earth surface in m/s^2
        step_size=params["env_step_size"],  # environment step size in seconds
    )

    eval_env = TwoBodyRendezvous_Env(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        TOF=params["TOF"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
    )

    max_episode_steps_in = 500
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    training_steps = max_episode_steps_in * 10

    obs, info = env.reset(seed=seed_in)
    obs, info = eval_env.reset(seed=seed_in)

    plt.style.use("data/support_files/dark_scientific.mplstyle")

    test_log = []
    test_log = log("SAC Training Script", test_log, flag_report_live)
    # print("GPU available: ", torch.cuda.is_available())  # Should print True if GPU is available)

    # paths
    path_nns = os.path.normpath(os.path.join(DATA_ROOT, "neural_networks"))
    path_output = os.path.normpath(
        os.path.join(DATA_ROOT, "test_data", "test_SAC_training_TBR")
    )
    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    path_output_log = os.path.join(path_output, "SAC_Training_Log.txt")
    path_output_log_truth = os.path.join(path_output, "SAC_Training_TBR_Log_truth.txt")

    # reset the environment
    observation, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, flag_report_live)
    test_log = log("Seed: " + str(seed_in), test_log, flag_report_live)

    # Create the SAC model
    model = SAC("MlpPolicy", env, verbose=1, device="cpu", seed=seed_in)

    # Train the agent
    callback = RewardLoggerCallback(print_freq=10000)
    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=1000,  # adjust frequency
        n_eval_episodes=5,  # episodes per evaluation
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([eval_callback, callback])

    model.learn(
        total_timesteps=training_steps, progress_bar=True, callback=callback_list
    )

    test_log = log("Training complete", test_log, flag_report_live)
    test_log = log_training_perf(
        test_log, callback, eval_callback, model, training_steps, flag_report_live
    )

    # Save the model
    model.save(path_SAC_model)

    arr_time = []
    arr_reward = []
    arr_reward_tot = []
    arr_throttle = []
    arr_alpha_x = []
    arr_alpha_y = []
    arr_x = []
    arr_y = []
    arr_vx = []
    arr_vy = []
    arr_x_target = []
    arr_y_target = []
    arr_vx_target = []
    arr_vy_target = []
    arr_ttg = []
    arr_sma = []
    arr_sma_target = []
    arr_ecc = []
    arr_ecc_target = []
    arr_ecc_max = []
    sum_reward = 0.0

    test_log = log("Plotting test trajectory...", test_log, True)
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    obs, info = env.reset(seed=42)
    obs, info = eval_env.reset(seed=42)
    eph = Ephemeris()  # create new ephemeris object

    while flag_continue:
        # step the env
        action, _states = model.predict(obs, deterministic=True)
        throttle = action[0]
        alpha_x = action[1]
        alpha_y = action[2]

        # dim state
        t_i = info["Elapsed time"]
        x_i = obs[0] * params["l_star"]
        y_i = obs[1] * params["l_star"]
        vx_i = obs[2] * params["l_star"] / params["t_star"]
        vy_i = obs[3] * params["l_star"] / params["t_star"]
        m_i = obs[4] * params["m_star"]
        x_t_i = obs[5] * params["l_star"]
        y_t_i = obs[6] * params["l_star"]
        vx_t_i = obs[7] * params["l_star"] / params["t_star"]
        vy_t_i = obs[8] * params["l_star"] / params["t_star"]
        ttg_i = obs[9] * params["t_star"]

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        obs, reward, terminated, truncated, info = env.step(action)

        count_step = count_step + 1

        # log data
        sum_reward = sum_reward + float(reward)
        arr_time.append(info["Elapsed time"] / (3600 * 24))  # time in days
        arr_reward.append(reward)
        arr_throttle.append(throttle)
        arr_alpha_x.append(alpha_x)
        arr_alpha_y.append(alpha_y)
        arr_reward_tot.append(sum_reward)
        arr_x.append(obs[0])
        arr_y.append(obs[1])
        arr_vx.append(obs[2])
        arr_vy.append(obs[3])
        arr_x_target.append(x_t_i)
        arr_y_target.append(y_t_i)
        arr_vx_target.append(vx_t_i)
        arr_vy_target.append(vy_t_i)
        arr_ttg.append(ttg_i)
        arr_sma.append(arr_OE[0])
        arr_ecc.append(arr_OE[1])
        arr_ecc_target.append(0.0)
        arr_ecc_max.append(1.0)

        if terminated or truncated:
            break

    test_log = log("Test trajectory complete", test_log, flag_report_live)
    test_log = log("Steps taken: " + str(count_step), test_log, flag_report_live)
    test_log = log("Total reward: " + str(sum_reward), test_log, flag_report_live)
    test_log = log("Final x: " + str(obs[0]) + " ", test_log, flag_report_live)
    test_log = log("Final y: " + str(obs[1]) + " ", test_log, flag_report_live)
    test_log = log("Final vx: " + str(obs[2]) + " ", test_log, flag_report_live)
    test_log = log("Final vy: " + str(obs[3]) + " ", test_log, flag_report_live)
    test_log = log("Final m: " + str(obs[4]) + " ", test_log, flag_report_live)
    test_log = log("Final x_target: " + str(x_t_i) + " ", test_log, flag_report_live)
    test_log = log("Final y_target: " + str(y_t_i) + " ", test_log, flag_report_live)
    test_log = log("Final vx_target: " + str(vx_t_i) + " ", test_log, flag_report_live)
    test_log = log("Final vy_target: " + str(vy_t_i) + " ", test_log, flag_report_live)
    test_log = log("Final ttg: " + str(ttg_i) + " ", test_log, flag_report_live)
    test_log = log("terminated: " + str(terminated) + " ", test_log, flag_report_live)
    test_log = log("truncated: " + str(truncated) + " ", test_log, flag_report_live)

    # plot the results
    plot_SAC_training(
        rollout_data1,
        arr_epsisode_numbers,
        arr_epsisode_rs,
        path_output,
        eph,
    )

    env.close()

    # save ephemeris to file
    eph.write_to_file(
        os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"),
        mod_vector_write_frequency=1,
    )

    fig_xy = eph.plot_xy()
    fig_xy.savefig(os.path.join(path_output, "SAC_Test_Traj.png"))

    test_log = log("Complete!", test_log, flag_report_live)
    test_log = log("Plots saved!", test_log, flag_report_live)

    # save log to file
    with open(os.path.join(path_output, "SAC_Training_Log.txt"), "w") as f:
        for line in test_log:
            f.write(line + "\n")

    # Compare log files with numerical tolerance for cross-platform compatibility
    are_same = compare_log_files_with_tolerance(
        path_output_log, path_output_log_truth, flag_report_live=flag_report_live
    )

    if flag_report_live:
        print("Log files match truth (with numerical tolerance):", are_same)

    return are_same


if __name__ == "__main__":
    test_SAC_training_TBR()

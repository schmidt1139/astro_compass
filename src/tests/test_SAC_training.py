import gymnasium as gym
import sys
import os
import torch
import matplotlib.pyplot as plt
import random
import filecmp

from datetime import datetime
from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# Adding python src code directory
# Adding python src code directory
os.chdir("C:/Users/micha/MSI_Data/Masters_Thesis/astro_compass")
print("Now working in:", os.getcwd())

sys.path.append(os.path.relpath("src/python/"))
sys.path.append(os.path.relpath("src/scripts/"))

from NN_Utils import query_NN_at_state
from Constants import Constants
from Neural_Net_Controllers import NN_TBT_Controller
from Log_Utils import log
from Ephemeris import Ephemeris
from Spacecraft import Spacecraft
from StateVectorUtilities import cartesian_to_polar, polar_to_cartesian
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from Plotting_Utils import plot_SAC_training
from RL_Utils import log_training_perf


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env_nd-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env_nd-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env_nd:TwoBody_Orb2Orb_Transfer_Env_nd",
        max_episode_steps=500  # <-- set your desired max steps per episode
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

def test_SAC_training(flag_report_live=False, seed_in=42):

    # set random seed
    random.seed(seed_in)

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
        "env_step_size": 3600*24,  # environment step size [s]
    }

    # initialize the environment
    env = gym.make("TwoBody_Orb2Orb_Transfer_Env_nd-v0",
                    mu=params["mu"],                    #solar gravitational parameter in m^3/s^2
                    max_T=params["max_T"],              #max thrust in N
                    ISP=params["ISP"],                  #ISP in seconds
                    TOF=params["TOF"],                  #time of flight in seconds
                    l_star=params["l_star"],            #characteristic length in m
                    m_star=params["m_star"],            #characteristic mass in kg
                    t_star=params["t_star"],            #characteristic time in s
                    g0=params["g0"],                    #gravitational acceleration at Earth surface in m/s^2
                    step_size=params["env_step_size"]   #environment step size in seconds
    )

    eval_env = gym.make("TwoBody_Orb2Orb_Transfer_Env_nd-v0",
                    mu=params["mu"], max_T=params["max_T"], ISP=params["ISP"],
                    TOF=params["TOF"], l_star=params["l_star"], m_star=params["m_star"],
                    t_star=params["t_star"], g0=params["g0"], step_size=params["env_step_size"])

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
    #print("GPU available: ", torch.cuda.is_available())  # Should print True if GPU is available)

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(os.getcwd(), "data\\neural_networks\\"))
    path_plots = os.path.normpath(os.path.join(os.getcwd(), "data\\plots\\SAC_plots\\"))
    path_output = os.path.normpath(os.path.join(os.getcwd(), "data\\test_data\\test_SAC_training\\"))
    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    path_output_log = os.path.join(path_output, "SAC_Training_Log.txt")
    path_output_log_truth = os.path.join(path_output, "SAC_Training_Log_truth.txt")


    # reset the environment
    observation, info = env.reset( seed=seed_in )
    test_log = log("Environment has been reset", test_log, flag_report_live)
    test_log = log("Seed: " + str(seed_in), test_log, flag_report_live)
    test_log = log("Max steps per episode: " + str(env.spec.max_episode_steps), test_log, flag_report_live)

    # Create the SAC model
    model = SAC("MlpPolicy", env, verbose=1, device="cpu", seed=seed_in)

    # Train the agent
    callback = RewardLoggerCallback(print_freq=10000)
    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=1000,             # adjust frequency
        n_eval_episodes=5,          # episodes per evaluation
        deterministic=True,
        render=False
    )
    callback_list = CallbackList([eval_callback, callback])

    model.learn(total_timesteps=training_steps, progress_bar=True , callback=callback_list)

    # After training:
    arr_epsisode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_epsisode_rs = callback.episode_rewards
    #print("Episodes:", len(callback.episode_rewards))
    #print("Timesteps:", model.num_timesteps)
    test_log = log("Training complete", test_log, flag_report_live)
    test_log = log_training_perf(test_log, callback, eval_callback, model, training_steps, flag_report_live)

    # Save the model
    model.save(path_SAC_model)

    # Optionally, test the trained agent
    eph = Ephemeris()  # create new ephemeris object

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

    while flag_continue:

        #step the env
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
        sma_t_i = obs[6] * params["l_star"]

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"] )
        arr_OE = SC.calc_Planar_OE( 0.0, 0.0, 0.0, 0.0, params["mu"] )

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
        arr_sma.append(arr_OE[0])
        arr_sma_target.append(sma_t_i)
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
    test_log = log("Final sma: " + str(obs[6]) + " ", test_log, flag_report_live)
    test_log = log("Final ecc: " + str(arr_OE[1]) + " ", test_log, flag_report_live)
    test_log = log("terminated: " + str(terminated) + " ", test_log, flag_report_live)
    test_log = log("truncated: " + str(truncated) + " ", test_log, flag_report_live)

    # plot the results
    #plot_SAC_training(arr_time, arr_reward_tot, arr_reward, arr_throttle, arr_alpha_x, arr_alpha_y, arr_x, arr_y, arr_vx, arr_vy, arr_sma, arr_sma_target, arr_ecc, arr_ecc_target, arr_ecc_max, arr_epsisode_numbers, arr_epsisode_rs, path_output, eph)

    env.close()

    # save ephemeris to file
    eph.write_to_file(os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"), mod_vector_write_frequency=1)

    test_log = log("Complete!", test_log, flag_report_live)
    test_log = log("Plots saved to: " + path_output, test_log, flag_report_live)

    # save log to file
    with open(os.path.join(path_output, "SAC_Training_Log.txt"), "w") as f:
        for line in test_log:
            f.write(line + "\n")

    # compare the two files
    are_same = filecmp.cmp(path_output_log, path_output_log_truth, shallow=False)

    if flag_report_live:
        print("Log files match truth:", are_same)

    return are_same


test_SAC_training(flag_report_live=False, seed_in=42)

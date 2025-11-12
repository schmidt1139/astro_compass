import gymnasium as gym
import sys
import os
import torch
import matplotlib.pyplot as plt
import random

from datetime import datetime
from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


print("Now working in:", os.getcwd())

from constants.constants import Constants
from utils.log_utils import log
from core.ephemeris_v2 import Ephemeris_v2 as Ephemeris
from core.spacecraft import Spacecraft
from utils.log_utils import write_log_to_file, write_config_file, read_config_file
from utils.state_vector_utils import cartesian_to_polar
from utils.plotting_utils import SACRolloutData_TBR, plot_SAC_training, SACRolloutData, plot_SAC_training_TBR
from utils.rl_utils import log_training_perf, RewardLoggerCallback
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env


def SAC_training_TBR(seed_in=42):

    # set random seed
    random.seed(seed_in)

    # config path
    path_config = os.path.join("data", "config", "SAC_training_TBR_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)



    # initialize the environment
    env = TwoBodyRendezvous_Env(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
        a_min_init_env_nd=params["a_min_init_env_nd"],
        a_max_init_env_nd=params["a_max_init_env_nd"],
        e_min_init_env=params["e_min_init_env"],
        e_max_init_env=params["e_max_init_env"],
        w_min_init_env_deg=params["w_min_init_env_deg"],
        w_max_init_env_deg=params["w_max_init_env_deg"],
        a_min_final_env_nd=params["a_min_final_env_nd"],
        a_max_final_env_nd=params["a_max_final_env_nd"],
        e_min_final_env=params["e_min_final_env"],
        e_max_final_env=params["e_max_final_env"],
        w_min_final_env_deg=params["w_min_final_env_deg"],
        w_max_final_env_deg=params["w_max_final_env_deg"]
    )

    eval_env = TwoBodyRendezvous_Env(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
        a_min_init_env_nd=params["a_min_init_env_nd"],
        a_max_init_env_nd=params["a_max_init_env_nd"],
        e_min_init_env=params["e_min_init_env"],
        e_max_init_env=params["e_max_init_env"],
        w_min_init_env_deg=params["w_min_init_env_deg"],
        w_max_init_env_deg=params["w_max_init_env_deg"],
        a_min_final_env_nd=params["a_min_final_env_nd"],
        a_max_final_env_nd=params["a_max_final_env_nd"],
        e_min_final_env=params["e_min_final_env"],
        e_max_final_env=params["e_max_final_env"],
        w_min_final_env_deg=params["w_min_final_env_deg"],
        w_max_final_env_deg=params["w_max_final_env_deg"]
    )

    max_episode_steps_in = 5000
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    training_steps = 100_000

    plt.style.use("data/support_files/dark_scientific.mplstyle")

    test_log = []
    test_log = log("SAC Training Script", test_log, True)
    print(
        "GPU available: ", torch.cuda.is_available()
    )  # Should print True if GPU is available)

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(os.getcwd(), "data\\neural_networks\\"))
    path_output = os.path.normpath(
        os.path.join(
            os.getcwd(), "data\\script_output\\SAC_training_" + time_tag + "\\"
        )
    )
    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    os.makedirs(path_output, exist_ok=True)

    # reset the environment
    observation, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)
    test_log = log(
        "Max steps per episode: " + str(max_episode_steps_in), test_log, True
    )

    # Create the SAC model with TensorBoard logging
    model = SAC("MlpPolicy", env, verbose=1, device="cpu", seed=seed_in, tensorboard_log=path_output)

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

    # After training:
    arr_epsisode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_epsisode_rs = callback.episode_rewards
    print("Episodes:", len(callback.episode_rewards))
    print("Timesteps:", model.num_timesteps)
    test_log = log("Training complete", test_log, True)
    test_log = log_training_perf(
        test_log, callback, eval_callback, model, training_steps, True
    )

    # Save the model
    model.save(path_SAC_model)

    # Optionally, test the trained agent
    obs, info = env.reset(seed=42)
    eph = Ephemeris()  # create new ephemeris object

    rollout_data1 = SACRolloutData_TBR()
    sum_reward = 0.0

    test_log = log("Plotting test trajectory...", test_log, True)
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    while flag_continue:

        # step the env
        action, _states = model.predict(obs, deterministic=True)
        throttle = action[0]
        alpha_x = action[1]
        alpha_y = action[2]

        # dim state
        t_i = info["Elapsed time"]
        t_i_days = t_i / (3600 * 24)
        x_i = obs[0] * params["l_star"]
        y_i = obs[1] * params["l_star"]
        vx_i = obs[2] * params["l_star"] / params["t_star"]
        vy_i = obs[3] * params["l_star"] / params["t_star"]
        m_i = obs[4] * params["m_star"]
        x_target_i = obs[5] * params["l_star"]
        y_target_i = obs[6] * params["l_star"]
        vx_target_i = obs[7] * params["l_star"] / params["t_star"]
        vy_target_i = obs[8] * params["l_star"] / params["t_star"]
        ttg_i = obs[9] * params["t_star"]

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, x_target_i, y_target_i, vx_target_i, vy_target_i, ttg_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        obs, reward, terminated, truncated, info = env.step(action)

        count_step = count_step + 1

        # log data
        rollout_data1.add_step(
            t_i_days,
            reward,
            throttle,
            alpha_x,
            alpha_y,
            obs[0],
            obs[1],
            obs[2],
            obs[3],
            obs[4],
            obs[5],
            obs[6],
            obs[7],
            obs[8],
            obs[9]
        )

        if terminated or truncated:
            break

    test_log = log("Test trajectory complete", test_log, True)
    test_log = log("Steps taken: " + str(count_step), test_log, True)
    test_log = log("Total reward: " + str(sum_reward), test_log, True)
    test_log = log("Final x: " + str(obs[0]) + " ", test_log, True)
    test_log = log("Final y: " + str(obs[1]) + " ", test_log, True)
    test_log = log("Final vx: " + str(obs[2]) + " ", test_log, True)
    test_log = log("Final vy: " + str(obs[3]) + " ", test_log, True)
    test_log = log("Final m: " + str(obs[4]) + " ", test_log, True)
    test_log = log("Final sma: " + str(obs[6]) + " ", test_log, True)
    test_log = log("Final ecc: " + str(arr_OE[1]) + " ", test_log, True)
    test_log = log("terminated: " + str(terminated) + " ", test_log, True)
    test_log = log("truncated: " + str(truncated) + " ", test_log, True)

    # plot the results
    plot_SAC_training_TBR(
        rollout_data1,
        arr_epsisode_numbers,
        arr_epsisode_rs,
        path_output,
        eph,
        params,
        env.unwrapped  # Unwrap to get the base TwoBodyRendezvous_Env
    )

    env.close()

    # save ephemeris to file
    eph.write_to_file(
        os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"),
        mod_vector_write_frequency=1,
    )

    test_log = log("Complete!", test_log, True)
    test_log = log("Plots saved to: " + path_output, test_log, True)

    # save log to file
    with open(os.path.join(path_output, "SAC_Training_Log.txt"), "w") as f:
        for line in test_log:
            f.write(line + "\n")


SAC_training_TBR()

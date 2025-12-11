import os
import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Update ALL imports to use the new package structure (remove 'python.' prefix):
from constants.constants import Constants
from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.utils.log_utils import log, log_parameters
from astro_compass.utils.plotting_utils import SACRolloutData, plot_SAC_training
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    import_training_into_replay_buffer,
    log_training_perf,
)
from astro_compass.utils.state_vector_utils import cartesian_to_polar

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env_nd_obs5-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env_nd_obs5-v0",
        entry_point="envs.TwoBody_Orb2Orb_Transfer_Env_nd_obs5:TwoBody_Orb2Orb_Transfer_Env_nd_obs5",
        max_episode_steps=500,  # <-- set your desired max steps per episode
    )


def SAC_seeded_training(seed_in=42):
    test_log = []
    test_log = log("SAC Training Script", test_log, True)

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
        "env_step_size": 3600 * 24,  # environment step size [s]
        "flag_seed_replay_buffer": True,  # flag to seed the replay buffer
        "num_ephems_to_use": 100,  # number of ephemerides to use for seeding
        "include_callbacks_in_learn": True,  # flag to include callbacks in learn() method
        "training_steps": 500,  # number of training steps
        "max_episode_steps_in": 500,  # max steps per episode
        "print_freq": 2500,  # frequency of evaluation and printing/logging rewards
        "n_eval_episodes": 16,  # number of episodes per evaluation
        "plot_style": "light",  # "light" or "dark"
    }

    test_log = log_parameters(params, test_log, True)

    # initialize the training environment
    env = gym.make(
        "TwoBody_Orb2Orb_Transfer_Env_nd_obs5-v0",
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

    # initialize the evaluation environment
    eval_env = gym.make(
        "TwoBody_Orb2Orb_Transfer_Env_nd_obs5-v0",
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

    # wrap envs
    env = gym.wrappers.TimeLimit(env, max_episode_steps=params["max_episode_steps_in"])
    eval_env = gym.wrappers.TimeLimit(
        eval_env, max_episode_steps=params["max_episode_steps_in"]
    )
    env = Monitor(env)
    eval_env = Monitor(eval_env)

    if params["plot_style"] == "light":
        plt.style.use("data/support_files/light_paper.mplstyle")
    elif params["plot_style"] == "dark":
        plt.style.use("data/support_files/dark_scientific.mplstyle")
    else:
        print("Invalid plot style selected in parameters dictionary.")
        print("Using default 'light' style.")
        plt.style.use("data/support_files/light_paper.mplstyle")

    print(
        "GPU available: ", torch.cuda.is_available()
    )  # Should print True if GPU is available)

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(os.getcwd(), "data", "neural_networks"))
    path_training_data = os.path.normpath(
        os.path.join(os.getcwd(), "data", "training_ephems", "test_set_bang_bang")
    )
    path_output = os.path.normpath(
        os.path.join(
            os.getcwd(),
            os.path.join("data", "script_output", "SAC_seeded_training_")
            + time_tag
            + "\\",
        )
    )
    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    os.makedirs(path_output, exist_ok=True)

    # reset the environment
    _, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)
    test_log = log(
        "Max steps per episode: " + str(params["max_episode_steps_in"]), test_log, True
    )

    # define the policy architecture
    policy_kwargs = dict(
        net_arch=[32, 32, 32, 32, 32],  # four hidden layers with 32 units each
        activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
    )

    # Create the blank slate SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=params[
            "include_callbacks_in_learn"
        ],  # Changed from 1 to 0 to suppress status updates
        device="cpu",
        seed=seed_in,
        policy_kwargs=policy_kwargs,
    )

    # Seed replay buffer if enabled
    if params["flag_seed_replay_buffer"]:
        print(type(model.replay_buffer))  # ReplayBuffer or DictReplayBuffer (goal envs)
        if model.replay_buffer is not None:
            print(
                "Experience buffer size: ", model.replay_buffer.size()
            )  # current number of transitions
            print(
                "Experience buffer capacity: ", model.replay_buffer.buffer_size
            )  # capacity
        else:
            print("Replay buffer is not initialized yet.")

        import_training_into_replay_buffer(
            path_training_data,  # path to directory containing training ephemerides
            test_log,  # log
            model,  # SAC model
            env,
            params,
        )

        if model.replay_buffer is not None:
            print(
                "Seeded experience buffer size: ", model.replay_buffer.size()
            )  # current number of transitions
            print(
                "Seeded experience buffer capacity: ", model.replay_buffer.buffer_size
            )  # capacity
        else:
            print("Replay buffer is not initialized yet.")

    # Setup callbacks
    callback = RewardLoggerCallback(print_freq=params["print_freq"])
    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=params["print_freq"],  # adjust frequency
        n_eval_episodes=params["n_eval_episodes"],  # episodes per evaluation
        deterministic=True,
        render=False,
    )

    if params["include_callbacks_in_learn"]:
        callback_list = CallbackList([eval_callback, callback])
    else:
        callback_list = None

    model.learn(
        total_timesteps=params["training_steps"],
        progress_bar=True,
        callback=callback_list,
    )

    print(type(model.replay_buffer))  # ReplayBuffer or DictReplayBuffer (goal envs)
    if model.replay_buffer is not None:
        print(
            "Experience buffer size: ", model.replay_buffer.size()
        )  # current number of transitions
        print(
            "Experience buffer capacity: ", model.replay_buffer.buffer_size
        )  # capacity
    else:
        print("Replay buffer is not initialized yet.")

    # After training:
    arr_epsisode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_epsisode_rs = callback.episode_rewards
    print("Episodes:", len(callback.episode_rewards))
    print("Timesteps:", model.num_timesteps)
    test_log = log("Training complete", test_log, True)
    test_log = log_training_perf(
        test_log, callback, eval_callback, model, params["training_steps"], True
    )

    # Save the model
    model.save(path_SAC_model)

    # Optionally, test the trained agent
    obs, info = env.reset(seed=42)
    eph = Ephemeris()  # create new ephemeris object

    rollout_data1 = SACRolloutData()

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
        t_i_days = t_i / (3600 * 24)  # time in days
        x_i = obs[0] * params["l_star"]
        y_i = obs[1] * params["l_star"]
        vx_i = obs[2] * params["l_star"] / params["t_star"]
        vy_i = obs[3] * params["l_star"] / params["t_star"]
        m_i = obs[4] * params["m_star"]
        sma_t_i = Constants.SMA_EARTH

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        # take the env step
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
            arr_OE[0],
            sma_t_i,
            arr_OE[1],
            0.0,
            1.0,
        )

        if terminated or truncated:
            break

    test_log = log("Test trajectory complete", test_log, True)
    test_log = log("Steps taken: " + str(count_step), test_log, True)
    test_log = log("Total reward: " + str(rollout_data1.sum_reward), test_log, True)
    test_log = log("Final x: " + str(obs[0]) + " ", test_log, True)
    test_log = log("Final y: " + str(obs[1]) + " ", test_log, True)
    test_log = log("Final vx: " + str(obs[2]) + " ", test_log, True)
    test_log = log("Final vy: " + str(obs[3]) + " ", test_log, True)
    test_log = log("Final m: " + str(obs[4]) + " ", test_log, True)
    test_log = log("Final sma: " + str(arr_OE[0]) + " ", test_log, True)
    test_log = log("Final ecc: " + str(arr_OE[1]) + " ", test_log, True)
    test_log = log("terminated: " + str(terminated) + " ", test_log, True)
    test_log = log("truncated: " + str(truncated) + " ", test_log, True)

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

    test_log = log("Complete!", test_log, True)
    test_log = log("Plots saved to: " + path_output, test_log, True)

    # save log to file
    with open(os.path.join(path_output, "SAC_Training_Log.txt"), "w") as f:
        for line in test_log:
            f.write(line + "\n")


SAC_seeded_training()

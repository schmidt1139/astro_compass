from Training_Data_Generation import generate_nn_training_data
from core.hamiltonian_control import Hamiltonian_Controller_TBT
import gymnasium as gym
import os
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn

from datetime import datetime
from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.monitor import Monitor
from constants.constants import Constants
from utils.log_utils import log
from core.ephemeris import Ephemeris as Ephemeris
from core.spacecraft import Spacecraft
from utils.log_utils import write_log_to_file, write_config_file, read_config_file
from utils.state_vector_utils import cartesian_to_polar
from utils.plotting_utils import plot_SAC_training, SACRolloutData
from utils.rl_utils import log_training_perf, RewardLoggerCallback, pre_train
from envs.TwoBody_Orb2Orb_Transfer_Env_nd_obs5 import (
    TwoBody_Orb2Orb_Transfer_Env_nd_obs5,
)
from core.process_single_trajectory import process_single_trajectory


def SAC_training_TBR(seed_in=42):
    test_log = []
    test_log = log("SAC Training Script", test_log, True)

    # set random seed
    random.seed(seed_in)

    # config path
    path_config = os.path.join("data", "config", "SAC_training_TBT_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)

    # initialize the environment
    env = TwoBody_Orb2Orb_Transfer_Env_nd_obs5(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
        mass_penalty=params["mass_penalty"],
    )

    eval_env = TwoBody_Orb2Orb_Transfer_Env_nd_obs5(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
    )

    sma_t_i = Constants.SMA_EARTH

    plt.style.use("data/support_files/light_paper.mplstyle")

    print(
        "GPU available: ", torch.cuda.is_available()
    )  # Should print True if GPU is available)

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(os.getcwd(), "data\\neural_networks\\"))

    # Handle both absolute and relative paths for output_dir
    output_base = params["output_dir"]
    if not os.path.isabs(output_base):
        output_base = os.path.join(os.getcwd(), output_base)
    path_output = os.path.normpath(
        os.path.join(output_base, "SAC_training_TBT_" + time_tag)
    )

    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    os.makedirs(path_output, exist_ok=True)
    params["output_dir_specific"] = path_output

    # env wrappers
    max_episode_steps_in = params["max_episode_steps"]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    training_steps = params["training_steps"]

    # make a subdir for checkpoints
    path_checkpoints = os.path.normpath(os.path.join(path_output, "checkpoints"))
    path_ephems = os.path.normpath(os.path.join(path_output, "ephems"))
    path_plots = os.path.normpath(os.path.join(path_output, "plots"))
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_ephems, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)

    # reset the environment
    observation, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)
    test_log = log(
        "Max steps per episode: " + str(max_episode_steps_in), test_log, True
    )

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # load model if specified, otherwise create new
    if params["load_model_checkpoint"]:
        test_log = log(
            "Loading SAC model from: " + params["path_SAC_model_load"], test_log, True
        )
        model = SB3_SAC.load(
            params["path_SAC_model_load"],
            env=env,
            device="cpu",
            seed=seed_in,
            tensorboard_log=path_output,
        )  # Use path_output so SB3 creates SAC_1/ subdirectory
    else:
        # Implement custom NN architectures
        nn_arch_type = params.get("nn_arch_type", "default")
        if nn_arch_type == "custom":
            # use default architecture
            # define the policy architecture
            policy_kwargs = dict(
                net_arch=[32, 32, 32, 32, 32],  # four hidden layers with 32 units each
                activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
                optimizer_kwargs=dict(eps=1e-5),
            )
            model = SB3_SAC(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                verbose=1,
                device="cpu",
                seed=seed_in,
                tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
                buffer_size=buffer_size,
                tau=params.get("tau", 0.005),
                train_freq=params.get("train_freq", 1),
                gradient_steps=params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
            )

        else:
            # use default architecture
            policy_kwargs = dict(
                optimizer_kwargs=dict(eps=1e-5)  # More stable Adam optimizer
            )
            model = SB3_SAC(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                verbose=1,
                device="cpu",
                seed=seed_in,
                tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
                buffer_size=buffer_size,
                tau=params.get("tau", 0.005),
                train_freq=params.get("train_freq", 1),
                gradient_steps=params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
            )

    if params["read_replay_buffer"]:
        test_log = log(
            "Loading replay buffer from: " + params["path_replay_buffer"],
            test_log,
            True,
        )
        model.load_replay_buffer(params["path_replay_buffer"])

    # pre-train networks if specified
    if params["pre_train_networks"]:
        test_log, arr_actor_loss_pt, arr_critic_loss_pt = pre_train(
            test_log, model, params, env
        )
        # Remove the minimal logger from pre-training so model.learn() can set up TensorBoard properly
        # This ensures TensorBoard logging works correctly during actual training
        if hasattr(model, "_logger"):
            delattr(model, "_logger")

    callback = RewardLoggerCallback(print_freq=params["print_freq"])
    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=params["eval_freq"],  # adjust frequency
        n_eval_episodes=params["n_eval_episodes"],  # episodes per evaluation
        deterministic=True,
        render=False,
    )
    # callback_list = CallbackList([eval_callback, callback])
    callback_list = CallbackList([callback])

    # Train the agent
    model.learn(
        total_timesteps=training_steps,
        progress_bar=True,
        callback=callback_list,
        tb_log_name=params["tb_log_name"],
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
    obs, info = env.reset(seed=params.get("seed_traj", 42))
    eph = Ephemeris()  # create new ephemeris object

    rollout_data1 = SACRolloutData()
    sum_reward = 0.0

    test_log = log("Plotting test trajectory...", test_log, True)
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    # optionally generate hamiltonian trajectory off of ephem
    if params.get("flag_gen_H_traj", False):
        test_log = log(
            "Generating Hamiltonian trajectory for comparison...", test_log, True
        )
        params["data_path"] = path_output
        params["scenario_index"] = 0
        params["flag_plot_traj"] = False

        init_observation = []
        init_observation.append(obs[0] * params["l_star"] / 1000)
        init_observation.append(obs[1] * params["l_star"] / 1000)
        init_observation.append(obs[2] * params["l_star"] / params["t_star"] / 1000)
        init_observation.append(obs[3] * params["l_star"] / params["t_star"] / 1000)
        init_observation.append(obs[4] * params["m_star"])
        init_observation.append(Constants.MU_SUN)
        init_observation.append(Constants.SMA_EARTH / 1000)

        input_TOF = 1.1 * 365.25 * 24 * 60 * 60

        unwrapped_env = env.unwrapped

        H_controller = Hamiltonian_Controller_TBT(
            unwrapped_env, init_observation, info, input_TOF
        )

        # modify parameters
        H_controller.eps_threshold = params.get("eps_final", 0.0004)

        # compute solution
        flag_solved, h_sol, eps, sol, h_log = H_controller.hamiltonian_solution_finder()

        ephem_H = Ephemeris()
        ephem_path = os.path.join(path_ephems, "Hamiltonian_Traj_Ephem.txt")

        if flag_solved:
            # write output ephemeris
            eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
                H_controller.generate_output_ephemeris(eph)
            )
            eph_out.write_to_file(ephem_path, mod_vector_write_frequency=1)

        try:
            test_log = log(
                "Generated Hamiltonian trajectory for comparison...", test_log, True
            )
            ephem_H.read_from_file(ephem_path)
        except Exception as e:
            test_log = log(
                "Error generating Hamiltonian trajectory file: " + str(e),
                test_log,
                True,
            )
            params["flag_gen_H_traj"] = False

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

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        obs, reward, terminated, truncated, info = env.step(action)
        reward_mass_component = info.get("reward_mass_component", 0.0)
        reward_distance_component = info.get("reward_distance_component", 0.0)

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
            reward_mass_component,
            reward_distance_component,
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

    # final env info
    for key, value in info.items():
        if key != "ODE Solution":
            test_log = log(f"{key}: {value}", test_log, True)

    # plot the results
    plot_SAC_training(
        rollout_data1,
        arr_epsisode_numbers,
        arr_epsisode_rs,
        path_output,
        eph,
        ephem_H if ephem_H is not None else None,
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
    write_log_to_file(os.path.join(path_output, "SAC_Training_Log.txt"), test_log)

    # write config to output dir
    write_config_file(params, os.path.join(path_output, "SAC_Training_Config.txt"))


SAC_training_TBR()

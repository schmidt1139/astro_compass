import os
import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.ephemeris_v2 import Ephemeris_v2 as Ephemeris
from core.process_single_trajectory import process_single_trajectory
from core.spacecraft import Spacecraft
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.utils.log_utils import (
    log,
    read_config_file,
    write_config_file,
    write_log_to_file,
)
from astro_compass.utils.plotting_utils import (
    SACRolloutData_TBR,
    plot_SAC_training_TBR,
)
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    log_training_perf,
    pre_train,
)
from astro_compass.utils.state_vector_utils import cartesian_to_polar


def SAC_training_TBR(seed_in=42):
    # set random seed
    random.seed(seed_in)

    # config path
    path_config = os.path.join("data", "config", "SAC_training_TBR_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)

    plt.style.use("data/support_files/light_paper.mplstyle")

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
        w_max_final_env_deg=params["w_max_final_env_deg"],
        pos_r_weight=params.get("pos_r_weight", 1.0),
        vel_r_weight=params.get("vel_r_weight", 1.0),
        mass_r_weight=params.get("mass_r_weight", 1.0),
        tof_scale=params.get("tof_scale", 1.0),
        r_dist_weight=params.get("r_dist_weight", 1.0),
        v_dist_weight=params.get("v_dist_weight", 1.0),
        success_threshold_pos=params.get("success_threshold_pos", 0.01),
        success_threshold_vel=params.get("success_threshold_vel", 0.01),
        terminal_bonus=params.get("terminal_bonus", 100.0),
        precision_mult=params.get("precision_mult", 10.0),
        tof_weight=params.get("tof_weight", 1.0),
        time_dist_weight=params.get("time_dist_weight", 1.0),
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
        w_max_final_env_deg=params["w_max_final_env_deg"],
        pos_r_weight=params.get("pos_r_weight", 1.0),
        vel_r_weight=params.get("vel_r_weight", 1.0),
        mass_r_weight=params.get("mass_r_weight", 1.0),
        tof_scale=params.get("tof_scale", 1.0),
        r_dist_weight=params.get("r_dist_weight", 1.0),
        v_dist_weight=params.get("v_dist_weight", 1.0),
        success_threshold_pos=params.get("success_threshold_pos", 0.01),
        success_threshold_vel=params.get("success_threshold_vel", 0.01),
        terminal_bonus=params.get("terminal_bonus", 100.0),
        precision_mult=params.get("precision_mult", 10.0),
        tof_weight=params.get("tof_weight", 1.0),
        time_dist_weight=params.get("time_dist_weight", 1.0),
    )

    # env wrappers
    max_episode_steps_in = params["max_episode_steps"]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    training_steps = params["training_steps"]

    plt.style.use("data/support_files/light_paper.mplstyle")

    test_log = []
    test_log = log("SAC Training Script", test_log, True)
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
        os.path.join(output_base, "SAC_training_" + time_tag)
    )

    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbt_model"))
    os.makedirs(path_output, exist_ok=True)
    params["output_dir_specific"] = path_output

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
            # define the policy architecture
            policy_kwargs = dict(
                net_arch=[32, 32, 32, 32, 32],  # four hidden layers with 32 units each
                activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
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

    else:
        arr_actor_loss_pt = []
        arr_critic_loss_pt = []

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
    callback_list = CallbackList([eval_callback, callback])

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

    rollout_data1 = SACRolloutData_TBR()
    sum_reward = 0.0

    test_log = log("Plotting test trajectory...", test_log, True)
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    # optionally generate hamiltonian trajectory
    if params.get("flag_gen_H_traj", False):
        test_log = log(
            "Generating Hamiltonian trajectory for comparison...", test_log, True
        )
        params["data_path"] = path_output
        params["scenario_index"] = 0
        params["flag_plot_traj"] = False
        results = process_single_trajectory(params)
        ephem_path = results[1]
        ephem_H = Ephemeris()
        try:
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
        x_target_i = obs[5] * params["l_star"]
        y_target_i = obs[6] * params["l_star"]
        vx_target_i = obs[7] * params["l_star"] / params["t_star"]
        vy_target_i = obs[8] * params["l_star"] / params["t_star"]
        ttg_i = obs[9] * params["t_star"]

        # info of interest
        pos_reward = info.get("pos_reward", None)
        vel_reward = info.get("vel_reward", None)
        mass_reward = info.get("mass_reward", None)

        # log data to ephemeris
        eph.add_data(
            t_i,
            x_i,
            y_i,
            vx_i,
            vy_i,
            m_i,
            x_target_i,
            y_target_i,
            vx_target_i,
            vy_target_i,
            ttg_i,
            alpha_x,
            alpha_y,
            throttle,
        )

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
            obs[9],
            pos_reward,
            vel_reward,
            mass_reward,
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

    # final env info
    for key, value in info.items():
        if key != "ODE Solution":
            test_log = log(f"{key}: {value}", test_log, True)

    # plot the results
    plot_SAC_training_TBR(
        rollout_data1,
        arr_epsisode_numbers,
        arr_epsisode_rs,
        path_output,
        eph,
        params,
        env.unwrapped,  # Unwrap to get the base TwoBodyRendezvous_Env
        arr_actor_loss_pt,
        arr_critic_loss_pt,
        ephem_H if params.get("flag_gen_H_traj", False) else None,
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

    print("\n\n\n")


SAC_training_TBR()

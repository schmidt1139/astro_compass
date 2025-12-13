import os
import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.hamiltonian_control import Hamiltonian_Controller_TBT
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_config_file,
    write_config_file,
)
from astro_compass.utils.path_utils import CONFIG_ROOT, RUNS_ROOT
from astro_compass.utils.plotting_utils import SACRolloutData, plot_SAC_training
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    pre_train,
)
from astro_compass.utils.state_vector_utils import cartesian_to_polar

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def SAC_training_TBR(params, output_dir, seed_in=42):
    test_log = []
    print("SAC Training Script")

    # set random seed
    random.seed(seed_in)

    # initialize the environment
    env = gen_rl_environment(params)
    eval_env = gen_rl_environment(params)

    sma_t_i = Constants.SMA_EARTH

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_output = os.path.join(output_dir, time_tag)
    path_SAC_model = os.path.join(path_output, "model")
    path_checkpoints = os.path.join(path_output, "checkpoints")
    path_ephems = os.path.join(path_output, "ephems")
    path_plots = os.path.join(path_output, "plots")
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_ephems, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)

    # Handle both absolute and relative paths for output_dir
    params["output_dir_specific"] = path_output

    # env wrappers
    max_episode_steps_in = params["max_episode_steps"]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    training_steps = params["training_steps"]

    # reset the environment
    observation, info = env.reset(seed=seed_in)
    print("Environment has been reset")
    print("Seed: " + str(seed_in))
    print("Max steps per episode: " + str(max_episode_steps_in))

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # load model if specified, otherwise create new
    if params["load_model_checkpoint"]:
        print(
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
        print(
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

    callback = RewardLoggerCallback(log_freq=params["log_freq"])
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
    print("Training complete")

    # Save the model
    model.save(path_SAC_model)

    # Optionally, test the trained agent
    obs, info = env.reset(seed=params.get("seed_traj", 42))
    eph = Ephemeris()  # create new ephemeris object

    rollout_data1 = SACRolloutData()
    sum_reward = 0.0

    print("Plotting test trajectory...")
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    # optionally generate hamiltonian trajectory off of ephem
    if params.get("flag_gen_H_traj", False):
        print("Generating Hamiltonian trajectory for comparison...", test_log, True)
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
            print("Generated Hamiltonian trajectory for comparison...", test_log, True)
            ephem_H.read_from_file(ephem_path)
        except Exception as e:
            print(
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

    print("Test trajectory complete")
    print("Steps taken: " + str(count_step))
    print("Total reward: " + str(rollout_data1.sum_reward))
    print("Final x: " + str(obs[0]) + " ")
    print("Final y: " + str(obs[1]) + " ")
    print("Final vx: " + str(obs[2]) + " ")
    print("Final vy: " + str(obs[3]) + " ")
    print("Final m: " + str(obs[4]) + " ")
    print("Final sma: " + str(arr_OE[0]) + " ")
    print("Final ecc: " + str(arr_OE[1]) + " ")
    print("terminated: " + str(terminated) + " ")
    print("truncated: " + str(truncated) + " ")

    # final env info
    for key, value in info.items():
        if key != "ODE Solution":
            print(f"{key}: {value}")

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

    print("Complete!")
    print("Plots saved to: " + path_output)

    # write config to output dir
    write_config_file(params, os.path.join(path_output, "SAC_Training_Config.txt"))


if __name__ == "__main__":
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.txt")
    params = read_config_file(path_config)

    output_dir = os.path.join(RUNS_ROOT, "SAC_training_TBT")
    # HACK FOR Legacy
    params["output_dir"] = output_dir

    SAC_training_TBR(params, output_dir)

import os
import random
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from astro_compass.core.ephemeris_v2 import Ephemeris_v2 as Ephemeris
from astro_compass.core.process_single_trajectory import process_single_trajectory
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    log,
    read_config_file,
    write_config_file,
    write_log_to_file,
)
from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.utils.plotting_utils import plot_SAC_training_TBR_polar
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    log_training_perf,
    pre_train,
    rollout_model,
)


def SAC_training_TBR(seed_in=42):
    # start time
    start_time = datetime.now()

    # set random seed
    random.seed(seed_in)

    # config path
    path_config = os.path.join(
        DATA_ROOT, "config", "SAC_training_TBR_polar__env2_config.txt"
    )

    # define parameters
    params = read_config_file(path_config)

    # initialize the training and evaluation environments
    # env = gen_rl_environment(params)
    # eval_env = gen_rl_environment(params)

    # set up vectorized environments
    max_episode_steps_in = params["max_episode_steps"]

    def make_env(params, seed):
        def _init():
            env = gen_rl_environment(params)
            env.seed(seed)
            env = gym.wrappers.TimeLimit(
                env, max_episode_steps=params["max_episode_steps"]
            )
            env = Monitor(env)
            return env

        return _init

    num_envs = params.get("num_vec_envs", 1)
    env = SubprocVecEnv([make_env(params, i) for i in range(num_envs)])

    # establish eval environment
    eval_env = gen_rl_environment(params)
    pre_train_env = gen_rl_environment(params)
    pre_train_env.seed(seed_in)
    pre_train_env.reset()
    pre_train_env = gym.wrappers.TimeLimit(
        pre_train_env, max_episode_steps=max_episode_steps_in
    )
    pre_train_env = Monitor(pre_train_env)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)
    eval_env = Monitor(eval_env)

    training_steps = params["training_steps"]

    test_log = []
    test_log = log("SAC Training Script", test_log, True)
    print(
        "GPU available: ", torch.cuda.is_available()
    )  # Should print True if GPU is available)
    test_log = log(
        "NN Eval Device: " + str(params.get("eval_device", "cpu")), test_log, True
    )

    # paths
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250928_143005"
    path_nns = os.path.normpath(os.path.join(os.getcwd(), "data\\neural_networks\\"))

    # Handle both absolute and relative paths for output_dir
    output_base = params["output_dir"]
    if not os.path.isabs(output_base):
        output_base = os.path.join(os.getcwd(), output_base)
    path_output = os.path.normpath(
        os.path.join(output_base, "SAC_training_TBR_polar" + time_tag)
    )

    path_SAC_model = os.path.normpath(os.path.join(path_nns, "sac_tbr_polar_model"))
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
    observations = env.reset()
    test_log = log("Environment has been reset", test_log, True)
    test_log = log(
        "Max steps per episode: " + str(max_episode_steps_in), test_log, True
    )

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # Number of vectorized environments
    num_vec_envs = params.get("num_vec_envs", 1)
    test_log = log(f"Number of vectorized environments: {num_vec_envs}", test_log, True)

    # load model if specified, otherwise create new
    if params["load_model_checkpoint"]:
        test_log = log(
            "Loading SAC model from: " + params["path_SAC_model_load"], test_log, True
        )
        model = SB3_SAC.load(
            params["path_SAC_model_load"],
            env=env,
            device=params.get("eval_device", "cpu"),
            seed=seed_in,
            tensorboard_log=path_output,
        )  # Use path_output so SB3 creates SAC_1/ subdirectory
    else:
        # Implement custom NN architectures
        nn_arch_type = params.get("nn_arch_type", "default")
        if nn_arch_type == "custom":
            test_log = log("Using custom neural network architecture", test_log, True)
            test_log = log(
                f"Network architecture: {params['net_arch']}", test_log, True
            )
            # define the policy architecture
            policy_kwargs = dict(
                net_arch=params["net_arch"],  # four hidden layers with 32 units each
                activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
            )
            model = SB3_SAC(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                verbose=1,
                device=params.get("eval_device", "cpu"),
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
                device=params.get("eval_device", "cpu"),
                seed=seed_in,
                tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
                buffer_size=buffer_size,
                tau=params.get("tau", 0.005),
                train_freq=params.get("train_freq", 1),
                gradient_steps=params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
            )

    # report number of trainable parameters
    num_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    test_log = log(
        f"Number of trainable parameters in the model: {num_params}", test_log, True
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
            test_log, model, params, pre_train_env
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
    arr_episode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_episode_rs = callback.episode_rewards
    print("Episodes:", len(callback.episode_rewards))
    print("Timesteps:", model.num_timesteps)
    test_log = log("Training complete", test_log, True)
    test_log = log_training_perf(
        test_log, callback, eval_callback, model, training_steps, True
    )

    # Save the model
    model.save(path_SAC_model)

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
            ephem_H.read(ephem_path)
        except Exception as e:
            test_log = log(
                "Error generating Hamiltonian trajectory file: " + str(e),
                test_log,
                True,
            )
            params["flag_gen_H_traj"] = False

    rollout_env = gen_rl_environment(params)
    rollout_env.seed(seed_in)

    test_log, eph, rollout_data = rollout_model(rollout_env, params, model, test_log)

    # render training plots
    plot_SAC_training_TBR_polar(
        rollout_data,
        path_plots,
        eph,
        params,
        rollout_env,
        arr_episode_numbers=arr_episode_numbers,
        arr_episode_rs=arr_episode_rs,
        arr_actor_loss_pt=arr_actor_loss_pt,
        arr_critic_loss_pt=arr_critic_loss_pt,
        ephem_H=ephem_H if params.get("flag_gen_H_traj", False) else None,
    )

    # save replay buffer if enabled
    if params.get("save_final_replay_buffer", False):
        path_replay_buffer = os.path.join(
            params["output_dir_specific"],
            "replay_buffer.pkl",
        )
        model.save_replay_buffer(path_replay_buffer)
        test_log = log(
            f"Final replay buffer saved to {path_replay_buffer}", test_log, True
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

    # elapsed time
    elapsed_time = (datetime.now() - start_time).total_seconds()
    test_log = log(f"Elapsed time: {elapsed_time:.2f} seconds", test_log, True)

    print("\n\n\n")


if __name__ == "__main__":
    SAC_training_TBR()

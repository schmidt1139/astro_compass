import os
import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_config_file,
    write_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import CONFIG_ROOT, RUNS_ROOT
from astro_compass.utils.plotting_utils import plot_reward_per_episode
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    pre_train,
)

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def SAC_training_TBT(params, output_dir, seed_in=42):
    test_log = []
    print("SAC Training Script")

    # set random seed
    random.seed(seed_in)

    # initialize the environment
    env = gen_rl_environment(params)
    eval_env = gen_rl_environment(params)

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

    model = get_model(params, env, seed_in, path_output)

    if params["read_replay_buffer"]:
        print("Loading replay buffer from: " + params["path_replay_buffer"])
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

    # Save the model
    model.save(path_SAC_model)

    # write config to output dir
    write_config_file(params, os.path.join(path_output, "SAC_Training_Config.txt"))

    arr_epsisode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_epsisode_rs = callback.episode_rewards
    plot_reward_per_episode(
        arr_epsisode_numbers,
        arr_epsisode_rs,
        os.path.join(path_plots, "reward_per_episode.png"),
    )


if __name__ == "__main__":
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.txt")
    params = read_config_file(path_config)

    output_dir = os.path.join(RUNS_ROOT, "SAC_training_TBT")
    # HACK FOR Legacy
    params["output_dir"] = output_dir

    SAC_training_TBT(params, output_dir)

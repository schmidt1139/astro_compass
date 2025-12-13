import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import CONFIG_ROOT, RUNS_ROOT, get_run_paths
from astro_compass.utils.plotting_utils import plot_reward_per_episode
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
)

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def SAC_training_TBT(params, output_dir, seed_in=42):
    random.seed(seed_in)

    # initialize the environment
    env = gen_rl_environment(params)
    eval_env = gen_rl_environment(params)

    # paths
    run_paths = get_run_paths(output_dir)
    path_output = run_paths["path_output"]

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

    model = get_model(params, env, seed_in, run_paths["path_SAC_model"])

    if params["read_replay_buffer"]:
        model.load_replay_buffer(params["path_replay_buffer"])

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
        tb_log_name=run_paths["path_output"].split(os.sep)[-1],  # time stamp
    )

    # Save the model
    model.save(run_paths["path_SAC_model"])

    arr_epsisode_numbers = list(range(1, len(callback.episode_rewards) + 1))
    arr_epsisode_rs = callback.episode_rewards
    plot_reward_per_episode(
        arr_epsisode_numbers,
        arr_epsisode_rs,
        run_paths["path_plots"],
    )


if __name__ == "__main__":
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.txt")
    params = read_config_file(path_config)

    output_dir = os.path.join(RUNS_ROOT, "SAC_training_TBT")
    # HACK FOR Legacy
    params["output_dir"] = output_dir

    SAC_training_TBT(params, output_dir)

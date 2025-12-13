import random

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import get_run_paths
from astro_compass.utils.rl_utils import (
    pre_train,
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

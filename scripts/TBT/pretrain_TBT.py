import os
import random

import gymnasium as gym
import torch
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    LOGS_ROOT,
    get_run_paths,
)
from astro_compass.utils.pre_train_utils import train_on_replay_buffer

print("GPU available: ", torch.cuda.is_available())


def pre_train(params, seed_in=42):
    random.seed(seed_in)

    # initialize the environment
    env = gen_rl_environment(params)
    eval_env = gen_rl_environment(params)

    # paths
    paths = get_run_paths(params)

    # env wrappers
    max_episode_steps_in = params["max_episode_steps"]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)

    env = Monitor(env)
    eval_env = Monitor(eval_env)

    env.reset(seed=seed_in)

    model = get_model(params, env, seed_in, LOGS_ROOT, model_id=None)

    model.load_replay_buffer(params["path_replay_buffer"])

    model = train_on_replay_buffer(model, params, env, paths)

    # Remove the minimal logger from pre-training so model.learn() can set up TensorBoard properly
    # This ensures TensorBoard logging works correctly during actual training
    if hasattr(model, "_logger"):
        delattr(model, "_logger")

    # save model
    model_path = paths["path_SAC_model"]
    model.save(model_path)


def main():
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(path_config)
    pre_train(params)


if __name__ == "__main__":
    main()

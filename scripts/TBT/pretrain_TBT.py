import os
import random

import gymnasium as gym
import pandas as pd
import torch
import tqdm
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    DATA_ROOT,
    LOGS_ROOT,
    RUNS_ROOT,
    get_run_paths,
)

print("GPU available: ", torch.cuda.is_available())


def pre_train(params, steps, batch_size, buffer_path, seed_in=42):
    random.seed(seed_in)

    # initialize the environment
    env = gen_rl_environment(params)
    eval_env = gen_rl_environment(params)

    # paths
    paths = get_run_paths(RUNS_ROOT)

    # env wrappers
    max_episode_steps_in = params["max_episode_steps"]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_in)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps_in)

    env = Monitor(env)
    eval_env = Monitor(eval_env)

    env.reset(seed=seed_in)

    model = get_model(params, env, seed_in, LOGS_ROOT, model_id=None)

    model._logger = Logger(folder=None, output_formats=[])

    model.load_replay_buffer(buffer_path)

    # model._logger = Logger(folder=None, output_formats=[])
    critic_losses = []
    actor_losses = []
    # make this a progress bar
    for step in tqdm.tqdm(range(steps)):
        model.train(gradient_steps=1, batch_size=batch_size)

        critic_losses.append(model._logger.name_to_value.get("train/critic_loss", []))
        actor_losses.append(model._logger.name_to_value.get("train/actor_loss", []))

    # Remove the minimal logger from pre-training so model.learn() can set up TensorBoard properly
    # This ensures TensorBoard logging works correctly during actual training
    if hasattr(model, "_logger"):
        delattr(model, "_logger")

    # save model
    model_path = paths["path_SAC_model"]
    model.save(model_path)

    df = pd.DataFrame(
        {
            "critic_loss": critic_losses,
            "actor_loss": actor_losses,
        }
    )

    df.to_csv(os.path.join(paths["path_output"], "pretrain_losses.csv"))

    # df.plot(y="critic_loss", title="Pre-training Critic Loss")
    # df.plot(y="actor_loss", title="Pre-training Actor Loss")
    # plt.show()

    return


def main():
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(path_config)
    steps = 50
    batch_size = 256
    buffer_path = os.path.join(
        DATA_ROOT,
        "pre-training-data",
        "TBT",
        "replay_buffers",
        "replay_buffer",
    )
    pre_train(params, steps, batch_size, buffer_path, seed_in=42)


if __name__ == "__main__":
    main()

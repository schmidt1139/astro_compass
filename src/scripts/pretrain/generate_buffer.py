import os
import random

import torch
import utils
from pretrain_utils import generate_env, generate_paths
from stable_baselines3 import SAC as SB3_SAC
from utils.log_utils import read_toml_config_file
from utils.rl_utils import (
    import_training_into_replay_buffer,
)

print("GPU available: ", torch.cuda.is_available())

# HACK
PROJECT_ROOT = os.path.dirname(os.path.dirname(utils.__file__)) + "/../.."


def main(params, training_data, seed_in=42):
    random.seed(seed_in)

    # initialize the training and evaluation environments
    env, eval_env, pre_train_env, single_env = generate_env(params, seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)

    # reset the environment
    single_env.reset()

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    model = SB3_SAC(
        "MlpPolicy",
        single_env,
        learning_rate=params["learning_rate"],
        verbose=1,
        device=params.get("eval_device", "cpu"),
        seed=seed_in,
        tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
        buffer_size=buffer_size,
        tau=params.get("tau", 0.005),
        train_freq=params.get("train_freq", 1),
        gradient_steps=params.get("gradient_steps", 1),
        policy_kwargs=dict(
            optimizer_kwargs=dict(eps=1e-5),  # More stable Adam optimizer
        ),
    )

    # loads data into model replay buffer
    test_log = []
    path_training_data = os.path.join(
        PROJECT_ROOT, "data", "pre-training-data", training_data
    )
    import_training_into_replay_buffer(
        path_training_data,  # path to directory containing training ephemerides
        test_log,  # log
        model,  # SAC model
        single_env,
        params,
    )

    path_replay_buffer = os.path.join(
        PROJECT_ROOT,
        "data",
        "pre-training",
        "replay_buffers",
        f"{training_data}_replay_buffer.pkl",
    )
    model.save_replay_buffer(path_replay_buffer)


if __name__ == "__main__":
    config_toml = "SAC_training_TBR_polar__JM_config.toml"
    training_data = "training-TBR-overfit"

    config_toml = "SAC_training_TBR_polar__JM_config.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)

    main(params, training_data)

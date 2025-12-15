import os
import random
import shutil

import torch
from stable_baselines3 import SAC as SB3_SAC

from astro_compass.core.training_data_generation import read_ephems_from_dir
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import log, read_toml_config_file
from astro_compass.utils.path_utils import CONFIG_ROOT, DATA_ROOT, PROJECT_ROOT
from astro_compass.utils.rl_utils import (
    import_training_into_replay_buffer_v3,
)

print("GPU available: ", torch.cuda.is_available())


def main(params, training_data_dir, output_dir, seed_in=42):
    random.seed(seed_in)

    # generate the environment
    env = gen_rl_environment(params)

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    model = SB3_SAC(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        verbose=1,
        device=params.get("eval_device", "cpu"),
        seed=seed_in,
        tensorboard_log=output_dir,  # Use output_dir so SB3 creates SAC_1/ subdirectory
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

    # read the ephems from directory
    params["ephem_version"] = 3.0  # ephem version to read
    params["cores"] = params.get("cores", 4)  # number of cores to use for reading
    set_ephems, filenames = read_ephems_from_dir(
        training_data_dir,
        params["num_ephems_to_use"],
        version=params["ephem_version"],
        flag_return_filenames=True,
        params=params,
    )

    import torch.nn as nn

    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
    )
    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed_in,
        tensorboard_log=output_dir,  # Use output_dir so SB3 creates SAC_1/ subdirectory
        buffer_size=buffer_size,
        policy_kwargs=policy_kwargs,
    )

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log("Starting size: " + str(model.replay_buffer.size()), test_log, True)

    import_training_into_replay_buffer_v3(set_ephems, test_log, model, env, params)

    path_replay_buffer = os.path.join(output_dir, "replay_buffer.pkl")
    model.save_replay_buffer(path_replay_buffer)

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log(f"Ending buffer size: {model.replay_buffer.size()}", test_log, True)

    test_log = log(f"\nSaved replay buffer to: {path_replay_buffer}", test_log, True)

    # copy the config file
    path_config_src = os.path.join(CONFIG_ROOT, params["config_toml"])
    path_config_dst = os.path.join(output_dir, params["config_toml"])
    shutil.copyfile(path_config_src, path_config_dst)


if __name__ == "__main__":
    config_toml = "gen_buffer_config.toml"
    path_config = os.path.join(CONFIG_ROOT, config_toml)
    params = read_toml_config_file(path_config)
    params["config_toml"] = config_toml

    training_data_dir = os.path.join(
        DATA_ROOT, "pre-train-data", "training_TBR_overfit", "ephems"
    )
    output_dir = os.path.join(PROJECT_ROOT, "replay_buffers")
    main(params, training_data_dir, output_dir)

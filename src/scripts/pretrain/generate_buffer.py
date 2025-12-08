import random
from pathlib import Path

import torch
from core.training_data_generation import read_ephems_from_dir
from stable_baselines3 import SAC as SB3_SAC
from utils.config_utils import load_config, write_config_sources
from utils.env_utils import gen_rl_environment
from utils.log_utils import log
from utils.pretrain_utils import generate_paths
from utils.rl_utils import (
    import_training_into_replay_buffer_v3,
)

print("GPU available: ", torch.cuda.is_available())


def main(params, seed_in=42, config_meta=None):
    random.seed(seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)

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
    training_data_path = params["path_training_data"]

    # read the ephems from directory
    params["ephem_version"] = 3.0  # ephem version to read
    params["cores"] = params.get("cores", 4)  # number of cores to use for reading
    set_ephems, filenames = read_ephems_from_dir(
        training_data_path,
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
        tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
        buffer_size=buffer_size,
        policy_kwargs=policy_kwargs,
    )

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log("Starting size: " + str(model.replay_buffer.size()), test_log, True)

    import_training_into_replay_buffer_v3(set_ephems, test_log, model, env, params)

    path_replay_buffer = params["path_replay_buffer"]
    model.save_replay_buffer(path_replay_buffer)

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log(
        "Ending buffer size: " + str(model.replay_buffer.size()), test_log, True
    )

    test_log = log(f"\nSaved replay buffer to: {path_replay_buffer}", test_log, True)

    # persist config provenance for traceability
    if config_meta:
        write_config_sources(config_meta, Path(path_output))


if __name__ == "__main__":
    base_files = [
        "common.toml",
        "envs.toml",
        "models.toml",
        "pretraining.toml",
    ]
    experiment_file = "experiments/generate_buffer_default.toml"
    params, meta = load_config(base_files=base_files, experiment_file=experiment_file)
    main(params, seed_in=0, config_meta=meta)

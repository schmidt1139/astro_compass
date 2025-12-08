import random
from pathlib import Path

import torch
from core.training_data_generation import read_ephems_from_dir
from utils.config_utils import load_config, write_config_sources
from utils.env_utils import gen_rl_environment
from utils.log_utils import log
from utils.model_utils import build_sac
from utils.pretrain_utils import generate_paths
from utils.rl_utils import (
    import_training_into_replay_buffer_v3,
)

print("GPU available: ", torch.cuda.is_available())


def main(config, seed_in=42, config_meta=None):
    random.seed(seed_in)

    paths_cfg = config["paths"]
    env_cfg = config["environment"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    general_cfg = config.get("general", {})

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(paths_cfg)

    # generate the environment
    env = gen_rl_environment(config)

    # Create the SAC model with TensorBoard logging
    buffer_size = training_cfg.get("buffer_size", 1000000)  # Default 1M transitions
    model = build_sac(
        env=env,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        seed=seed_in,
        tensorboard_log=path_output,
    )

    # loads data into model replay buffer
    test_log = []
    training_data_path = paths_cfg["path_training_data"]

    # read the ephems from directory
    cores = general_cfg.get("cores", env_cfg.get("vectorization", {}).get("cores", 4))
    set_ephems, filenames = read_ephems_from_dir(
        training_data_path,
        training_cfg["num_ephems_to_use"],
        version=general_cfg.get("ephem_version", 3.0),
        flag_return_filenames=True,
        params={"cores": cores},
    )

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log("Starting size: " + str(model.replay_buffer.size()), test_log, True)

    import_training_into_replay_buffer_v3(set_ephems, test_log, model, env, config)

    path_replay_buffer = paths_cfg["path_replay_buffer"]
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
    config, meta = load_config(base_files=base_files, experiment_file=experiment_file)
    main(config, seed_in=0, config_meta=meta)

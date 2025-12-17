import os
import random
import shutil

import torch
from stable_baselines3 import SAC as SB3_SAC

from astro_compass.core.ephem_converter import (
    import_training_into_replay_buffer_v3,
    read_ephems,
)
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.path_utils import CONFIG_ROOT, DATA_ROOT, PROJECT_ROOT

print("GPU available: ", torch.cuda.is_available())


def main(params, training_data_dir, output_dir, seed_in=42):
    random.seed(seed_in)

    # generate the environment
    env = gen_rl_environment(params)

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # read the ephems from directory
    params["ephem_version"] = 3.0  # ephem version to read
    params["cores"] = params.get("cores", 4)  # number of cores to use for reading
    set_ephems, filenames = read_ephems(
        training_data_dir,
        params["num_ephems"],
        version=params["ephem_version"],
        return_filenames=True,
    )

    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed_in,
        buffer_size=buffer_size,
    )
    test_log = []
    import_training_into_replay_buffer_v3(set_ephems, test_log, model, env, params)

    path_replay_buffer = os.path.join(output_dir, "replay_buffer.pkl")
    model.save_replay_buffer(path_replay_buffer)

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

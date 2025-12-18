import os
import pickle
import random
import shutil

import torch
from stable_baselines3 import SAC as SB3_SAC

from astro_compass.core.ephem_converter import generate_rollouts
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.path_utils import CONFIG_ROOT, DATA_ROOT

print("GPU available: ", torch.cuda.is_available())


def main(params, ephems_dir, output_dir, seed_in=42):
    random.seed(seed_in)

    # generate the environment
    env = gen_rl_environment(params)

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # read the ephems from directory
    params["ephem_version"] = 3.0  # ephem version to read
    params["cores"] = params.get("cores", 4)  # number of cores to use for reading

    ephem_paths = os.listdir(ephems_dir)
    ephems = []
    for ephem_path in ephem_paths:
        full_path = os.path.join(ephems_dir, ephem_path)
        with open(full_path, "rb") as f:
            ephem = pickle.load(f)
            ephems.append(ephem)

    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed_in,
        buffer_size=buffer_size,
    )
    rollouts = generate_rollouts(ephems, params)

    for rollout in rollouts:
        for i in range(len(rollout.obs)):
            model.replay_buffer.add(
                obs=rollout.obs[i].reshape(1, -1),
                action=rollout.action[i].reshape(1, -1),
                reward=rollout.reward[i].reshape(1, -1),
                next_obs=rollout.next_obs[i].reshape(1, -1),
                done=rollout.done[i].reshape(1, -1),
                infos=[rollout.info[i]],
            )

    path_replay_buffer = os.path.join(output_dir, "replay_buffer.pkl")
    model.save_replay_buffer(path_replay_buffer)

    # copy the config file
    path_config_src = os.path.join(CONFIG_ROOT, params["config_toml"])
    path_config_dst = os.path.join(output_dir, params["config_toml"])
    shutil.copyfile(path_config_src, path_config_dst)


if __name__ == "__main__":
    config_toml = "SAC_training_TBT_config.toml"
    path_config = os.path.join(CONFIG_ROOT, config_toml)
    params = read_toml_config_file(path_config)
    params["config_toml"] = config_toml

    ephems_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")
    output_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "replay_buffers")
    main(params, ephems_dir, output_dir)

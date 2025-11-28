import os
import random

from core.training_data_generation import read_ephems_from_dir
import torch
import utils
from pretrain_utils import generate_env, generate_paths
from stable_baselines3 import SAC as SB3_SAC
from utils.log_utils import read_toml_config_file, log
from utils.rl_utils import (
    import_training_into_replay_buffer,
    import_training_into_replay_buffer_v3,
)
from utils.env_utils import gen_rl_environment

print("GPU available: ", torch.cuda.is_available())

# HACK
PROJECT_ROOT = os.path.dirname(os.path.dirname(utils.__file__)) + "/../.."


def main(params, training_data, seed_in=42):
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
    path_training_data = os.path.join(
        PROJECT_ROOT, "data", "pre-training-data", training_data
    )

    # read the ephems from directory
    params["ephem_version"] = 3.0  # ephem version to read
    params["cores"] = params.get("cores", 4)  # number of cores to use for reading
    set_ephems, filenames = read_ephems_from_dir( training_data, 
                                                  params["num_ephems_to_use"],  
                                                  version=params["ephem_version"], 
                                                  flag_return_filenames=True,
                                                  params=params )

    import torch.nn as nn
    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
    )
    model = SB3_SAC("MlpPolicy", 
                    env, 
                    verbose=1, 
                    seed=seed_in,
                    tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
                    buffer_size=buffer_size,
                    policy_kwargs=policy_kwargs)
    
    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log("Starting size: " + str(model.replay_buffer.size()), test_log, True)

    import_training_into_replay_buffer_v3( set_ephems, test_log, model, env, params )

    path_replay_buffer = os.path.join(path_output, "replay_buffer.pkl")
    model.save_replay_buffer(path_replay_buffer)

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, True)
    test_log = log("Ending buffer size: " + str(model.replay_buffer.size()), test_log, True)
    
    test_log = log(f"\nSaved replay buffer to: {path_replay_buffer}", test_log, True)

if __name__ == "__main__":
    config_toml = "SAC_training_TBR_polar__JM_config.toml"

    config_toml = "gen_buffer_config.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)
    training_data = params["path_training_data"]
    main(params, training_data)

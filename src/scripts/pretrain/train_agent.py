import os
import random

import tomli as toml
import torch
import utils
from pretrain_utils import generate_env, generate_paths
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from utils.callbacks import ReplayBufferCheckpointCallback
from utils.model_utils import load_sac_model, save_sac_model
from utils.rl_utils import (
    RewardLoggerCallback,
)

print("GPU available: ", torch.cuda.is_available())
# HACK


def configure_callbacks(params, path_output, eval_env):
    callback = RewardLoggerCallback(print_freq=params["print_freq"])

    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=params["eval_freq"],  # adjust frequency
        n_eval_episodes=params["n_eval_episodes"],  # episodes per evaluation
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([eval_callback, callback])
    save_freq_adj = int(
        params["n_freq_checkpoint_replay_buffer"] / params["num_vec_envs"]
    )
    if params.get("flag_checkpoint_replay_buffer", False):
        replay_buffer_callback = ReplayBufferCheckpointCallback(
            save_freq=save_freq_adj, save_path=path_output
        )
        callback_list = CallbackList([eval_callback, callback, replay_buffer_callback])

    return callback_list


def main(env_params, model_params, seed_in=42):
    random.seed(seed_in)

    env, eval_env, pre_train_env, single_env = generate_env(env_params, seed_in)
    path_output, path_SAC_model, path_plots = generate_paths(model_params)

    env.reset()

    model = load_sac_model(model_params, env, path_output, seed_in)
    callback_list = configure_callbacks(env_params, path_output, eval_env)

    # Train the agent
    model.learn(
        total_timesteps=env_params["training_steps"],
        progress_bar=True,
        callback=callback_list,
        tb_log_name=env_params["tb_log_name"],
    )

    save_sac_model(model, model_params, path_output, path_SAC_model)


if __name__ == "__main__":
    model_config = "model_base.toml"
    env_config = "env_TBR_base.toml"
    hamiltonian_config = "hamiltonian_TBR_base.toml"
    PROJECT_ROOT = os.path.dirname(os.path.dirname(utils.__file__)) + "/../.."

    def read_toml(config_file):
        path_config = os.path.join(PROJECT_ROOT, "data", "config", config_file)
        with open(path_config, "rb") as f:
            config_params = toml.load(f)
        return config_params

    model_params = read_toml(model_config)
    env_params = read_toml(env_config)
    hamiltonian_params = read_toml(hamiltonian_config)

    # Optional user modifications
    model_params["read_replay_buffer"] = False
    model_params["load_model_checkpoint"] = False

    main(env_params, model_params)

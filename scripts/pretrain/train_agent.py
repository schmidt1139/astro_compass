import os
import random
import shutil

import torch
import utils
from pretrain_utils import generate_env, generate_paths
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from astro_compass.utils.callbacks import ReplayBufferCheckpointCallback
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.rl_utils import (
    RewardLoggerCallback,
    log_training_perf,
)

print("GPU available: ", torch.cuda.is_available())
# HACK
PROJECT_ROOT = os.path.dirname(os.path.dirname(utils.__file__)) + "/../.."


def main(params, seed_in=42):
    random.seed(seed_in)

    # initialize the training and evaluation environments
    env, eval_env, pre_train_env, single_env = generate_env(params, seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)

    # reset the environment
    env.reset()

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    if params["load_model_checkpoint"]:
        model = SB3_SAC.load(
            params["path_SAC_model_load"],
            env=env,
            device=params.get("eval_device", "cpu"),
            seed=seed_in,
            tensorboard_log=path_output,
        )  # Use path_output so SB3 creates SAC_1/ subdirectory
    else:
        if params.get("nn_arch_type", "default") == "custom":
            policy_kwargs = dict(
                net_arch=params["net_arch"],  # four hidden layers with 32 units each
                activation_fn=torch.nn.LeakyReLU,  # LeakyReLU activation function
            )
        else:
            # use default architecture
            policy_kwargs = dict(
                optimizer_kwargs=dict(eps=1e-5),  # More stable Adam optimizer
            )

        model = SB3_SAC(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            verbose=1,
            device=params.get("eval_device", "cpu"),
            seed=seed_in,
            learning_starts=params.get("learning_starts", 50000),
            tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
            buffer_size=buffer_size,
            tau=params.get("tau", 0.005),
            train_freq=(
                params.get("train_freq", 1),
                params.get("train_freq_unit", "step"),
            ),
            gradient_steps=params.get("gradient_steps", 1),
            policy_kwargs=policy_kwargs,
        )

    if params["read_replay_buffer"]:
        model.load_replay_buffer(params["path_replay_buffer"])

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

    # Train the agent
    model.learn(
        total_timesteps=params["training_steps"],
        progress_bar=True,
        callback=callback_list,
        tb_log_name=params["tb_log_name"],
    )

    test_log = []
    test_log = log_training_perf(
        test_log,
        callback,
        eval_callback,
        model,
        params["training_steps"],
        True,
    )

    # Save the model
    model.save(path_SAC_model)

    # optionally save the replay buffer
    if params.get("save_final_replay_buffer", False):
        path_replay_buffer = os.path.join(path_output, "replay_buffer.pkl")
        model.save_replay_buffer(path_replay_buffer)
        print("Replay buffer saved to: ", path_replay_buffer)

    # copy the config file
    path_config_src = os.path.join(
        PROJECT_ROOT, "data", "config", params["config_toml"]
    )
    path_config_dst = os.path.join(path_output, params["config_toml"])
    shutil.copyfile(path_config_src, path_config_dst)

    print("Model saved to: ", path_SAC_model)
    print("Output saved to: ", path_output)


if __name__ == "__main__":
    config_toml = "train_agent_config.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)

    params["read_replay_buffer"] = False
    params["load_model_checkpoint"] = False
    params["config_toml"] = config_toml

    main(params)

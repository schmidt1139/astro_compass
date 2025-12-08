import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import SAC as SB3_SAC
from utils.config_utils import load_config, write_config_sources
from utils.pretrain_utils import generate_env, generate_paths
from utils.rl_utils import pre_train

print("GPU available: ", torch.cuda.is_available())


def main(params, seed_in=42, config_meta=None):
    random.seed(seed_in)

    # initialize the training and evaluation environments
    env, eval_env, pre_train_env, single_env = generate_env(params, seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)

    # reset the environment
    env.reset()

    # Create the SAC model with TensorBoard logging
    buffer_size = params.get("buffer_size", 1000000)  # Default 1M transitions

    # Implement custom NN architectures
    if params.get("nn_arch_type", "default") == "custom":
        policy_kwargs = dict(
            net_arch=params["net_arch"],  # four hidden layers with 32 units each
            activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
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
        tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
        buffer_size=buffer_size,
        tau=params.get("tau", 0.005),
        train_freq=params.get("train_freq", 1),
        gradient_steps=params.get("gradient_steps", 1),
        policy_kwargs=policy_kwargs,
    )

    model.load_replay_buffer(params["path_replay_buffer"])

    # pre-train networks if specified
    test_log = []

    test_log, arr_actor_loss_pt, arr_critic_loss_pt = pre_train(
        test_log,
        model,
        params,
        pre_train_env,
    )
    # Remove the minimal logger from pre-training so model.learn() can set up TensorBoard properly
    # This ensures TensorBoard logging works correctly during actual training
    if hasattr(model, "_logger"):
        delattr(model, "_logger")

    plt.figure()
    if len(arr_actor_loss_pt) > 0:
        if max(arr_actor_loss_pt) > 0:
            plt.semilogy(arr_actor_loss_pt, label="Actor Loss")
        else:
            plt.plot(arr_actor_loss_pt, label="Actor Loss")
        plt.legend()
        plt.savefig(os.path.join(path_plots, "pretrain_actor_loss.png"), dpi=300)

        plt.figure()
        if max(arr_critic_loss_pt) > 0:
            plt.semilogy(arr_critic_loss_pt, label="Critic Loss")
        else:
            plt.plot(arr_critic_loss_pt, label="Critic Loss")
        plt.legend()
        plt.savefig(os.path.join(path_plots, "pretrain_critic_loss.png"), dpi=300)

    # Save the model
    model.save(path_SAC_model)
    model.save(params["path_SAC_model_save"])

    if config_meta:
        write_config_sources(config_meta, Path(path_output))

    plt.show()


if __name__ == "__main__":
    base_files = [
        "common.toml",
        "envs.toml",
        "models.toml",
        "pretraining.toml",
    ]
    experiment_file = "experiments/pretrain_default.toml"
    params, meta = load_config(base_files=base_files, experiment_file=experiment_file)
    main(params, seed_in=0, config_meta=meta)

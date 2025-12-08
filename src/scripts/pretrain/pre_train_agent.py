import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from utils.config_utils import load_config, write_config_sources
from utils.model_utils import build_sac
from utils.pretrain_utils import generate_env, generate_paths
from utils.rl_utils import pre_train

print("GPU available: ", torch.cuda.is_available())


def main(config, seed_in=42, config_meta=None):
    random.seed(seed_in)

    paths_cfg = config["paths"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    # initialize the training and evaluation environments
    env, eval_env, pre_train_env, single_env = generate_env(config, seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(paths_cfg)
    paths_cfg["path_plots"] = path_plots

    # reset the environment
    env.reset()

    # Create the SAC model with TensorBoard logging
    model = build_sac(
        env=env,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        seed=seed_in,
        tensorboard_log=path_output,
    )

    if training_cfg.get("read_replay_buffer", True):
        model.load_replay_buffer(paths_cfg["path_replay_buffer"])

    # pre-train networks if specified
    test_log = []

    test_log, arr_actor_loss_pt, arr_critic_loss_pt = pre_train(
        test_log,
        model,
        config,
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
    model.save(paths_cfg["path_SAC_model_save"])

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
    config, meta = load_config(base_files=base_files, experiment_file=experiment_file)
    main(config, seed_in=0, config_meta=meta)

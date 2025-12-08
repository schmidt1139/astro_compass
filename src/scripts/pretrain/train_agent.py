import os
import random

import torch
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from utils.callbacks import ReplayBufferCheckpointCallback
from utils.config_utils import load_config, write_config_sources
from utils.model_utils import build_sac
from utils.pretrain_utils import generate_env, generate_paths
from utils.rl_utils import RewardLoggerCallback, log_training_perf

print("GPU available: ", torch.cuda.is_available())


def main(config, seed_in=42, config_meta=None):
    random.seed(seed_in)

    paths_cfg = config["paths"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    env_cfg = config["environment"]

    env, eval_env, pre_train_env, single_env = generate_env(config, seed_in)

    path_output, path_SAC_model, path_plots = generate_paths(paths_cfg)
    paths_cfg["path_plots"] = path_plots

    env.reset()

    if training_cfg.get("load_model_checkpoint", False):
        model = SB3_SAC.load(
            paths_cfg["path_SAC_model_load"],
            env=env,
            device=model_cfg.get("eval_device", "cpu"),
            seed=seed_in,
            tensorboard_log=path_output,
        )
    else:
        model = build_sac(
            env=env,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            seed=seed_in,
            tensorboard_log=path_output,
        )

    if training_cfg.get("read_replay_buffer", True):
        model.load_replay_buffer(paths_cfg["path_replay_buffer"])

    callback = RewardLoggerCallback(print_freq=training_cfg["print_freq"])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=training_cfg["eval_freq"],
        n_eval_episodes=training_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([eval_callback, callback])

    if training_cfg.get("flag_checkpoint_replay_buffer", False):
        save_freq_adj = int(
            training_cfg["n_freq_checkpoint_replay_buffer"]
            / env_cfg.get("vectorization", {}).get("num_vec_envs", 1)
        )
        replay_buffer_callback = ReplayBufferCheckpointCallback(
            save_freq=save_freq_adj, save_path=path_output
        )
        callback_list = CallbackList([eval_callback, callback, replay_buffer_callback])

    model.learn(
        total_timesteps=training_cfg["training_steps"],
        progress_bar=True,
        callback=callback_list,
        tb_log_name=paths_cfg.get("tb_log_name", "SAC_training"),
    )

    test_log = []
    test_log = log_training_perf(
        test_log, callback, eval_callback, model, training_cfg["training_steps"], True
    )

    model.save(path_SAC_model)

    if training_cfg.get("save_final_replay_buffer", False):
        path_replay_buffer = os.path.join(path_output, "replay_buffer.pkl")
        model.save_replay_buffer(path_replay_buffer)
        print("Replay buffer saved to: ", path_replay_buffer)

    if config_meta:
        from pathlib import Path

        write_config_sources(config_meta, Path(path_output))

    print("Model saved to: ", path_SAC_model)
    print("Output saved to: ", path_output)


if __name__ == "__main__":
    base_files = [
        "common.toml",
        "envs.toml",
        "models.toml",
        "training.toml",
    ]
    experiment_file = "experiments/train_default.toml"
    config, meta = load_config(base_files=base_files, experiment_file=experiment_file)

    config["training"]["read_replay_buffer"] = False
    config["training"]["load_model_checkpoint"] = False

    main(config, seed_in=0, config_meta=meta)

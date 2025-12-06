import os
import shutil

import torch
from stable_baselines3 import SAC as SB3_SAC

import python

PROJECT_ROOT = os.path.dirname(os.path.dirname(python.__file__)) + "/../.."


def load_sac_model(params, env, path_output, seed_in=42):
    if params["load_model_checkpoint"]:
        model = SB3_SAC.load(
            params["path_SAC_model_load"],
            env=env,
            device=params.get("eval_device", "cpu"),
            seed=seed_in,
            tensorboard_log=path_output,
        )
    else:
        policy_kwargs = dict(optimizer_kwargs=dict(eps=1e-5))

        if params.get("nn_arch_type", "default") == "custom":
            policy_kwargs = dict(
                net_arch=params["net_arch"],
                activation_fn=torch.nn.LeakyReLU,
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
            buffer_size=params.get("buffer_size", 1_000_000),
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

    return model


def save_sac_model(model, params, path_output, path_SAC_model):
    # Save the model
    model.save(path_SAC_model)
    print("Model saved to: ", path_SAC_model)

    # optionally save the replay buffer
    if params.get("save_final_replay_buffer", False):
        path_replay_buffer = os.path.join(path_output, "replay_buffer.pkl")
        model.save_replay_buffer(path_replay_buffer)
        print("Replay buffer saved to: ", path_replay_buffer)

    # copy the used config file to confirm it was the same
    path_config_src = os.path.join(
        PROJECT_ROOT, "data", "config", params["config_toml"]
    )
    path_config_dst = os.path.join(path_output, params["config_toml"] + ".saved")
    shutil.copyfile(path_config_src, path_config_dst)
    print("Config file saved to: ", path_config_dst)
    print("Output products saved to: ", path_output)

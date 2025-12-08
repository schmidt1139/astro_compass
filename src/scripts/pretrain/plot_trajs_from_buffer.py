import os

import torch
import torch.nn as nn
from pretrain_utils import generate_paths
from stable_baselines3 import SAC as SB3_SAC
from utils.buffer_utils import extract_episodes_from_buffer, plot_episodes
from utils.env_utils import gen_rl_environment
from utils.log_utils import read_toml_config_file
from utils.path_utils import PROJECT_ROOT

print("GPU available: ", torch.cuda.is_available())


def main(params):
    print("Plot Trajectories from Replay Buffer Script")
    print("")

    seed_in = params.get("seed", 42)

    env = gen_rl_environment(params)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)

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
        verbose=0,
        device=params.get("eval_device", "cpu"),
        seed=seed_in,
        tensorboard_log=path_output,  # Use path_output so SB3 creates SAC_1/ subdirectory
        tau=params.get("tau", 0.005),
        train_freq=params.get("train_freq", 1),
        gradient_steps=params.get("gradient_steps", 1),
        policy_kwargs=policy_kwargs,
    )

    print("Loading the replay buffer: " + params["path_replay_buffer"])
    model.load_replay_buffer(params["path_replay_buffer"])
    print("Replay buffer loaded.")
    transitions = model.replay_buffer.size()
    print(f"Number of transitions in replay buffer: {transitions}")

    print("Extracting episodes from the replay buffer...")
    episodes = extract_episodes_from_buffer(model.replay_buffer, params)
    print(f"Extracted {len(episodes)} episodes from the replay buffer.")

    plot_episodes(episodes, params, path_plots)
    print("Plots saved to: " + path_plots)


if __name__ == "__main__":
    config_toml = "plot_buffer.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)
    params["config_toml"] = config_toml

    main(params)
